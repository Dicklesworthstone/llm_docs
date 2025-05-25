"""
Module for distilling documentation using LLMs via aisuite.
"""

import asyncio
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aisuite as ai
import tiktoken
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from llm_docs.config import config
from llm_docs.storage.models import Package
from llm_docs.utils.token_tracking import update_aisuite_response_tracking

# Initialize console
console = Console()

class LLMAPIError(Exception):
    """Exception raised when LLM API calls fail after retries."""
    pass

class DocumentationDistiller:
    """Distills documentation using LLMs via aisuite."""
    
    def __init__(self, 
                 output_dir: str = "distilled_docs", 
                 llm_config: Optional[Dict[str, Any]] = None,
                 max_chunk_tokens: int = 80000,
                 retry_delay: int = 5,
                 max_retries: int = 3):
        """
        Initialize the documentation distiller.
        
        Args:
            output_dir: Directory to store distilled documentation
            llm_config: LLM provider configuration (defaults to config.llm.distillation or config.llm.default)
            max_chunk_tokens: Maximum number of tokens per chunk
            retry_delay: Initial delay between retries in seconds
            max_retries: Maximum number of retries for API calls
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up cache directory
        self.cache_dir = Path("cache/distillation")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Use provided config or get from global config
        if llm_config:
            self.provider = llm_config.get("provider", "anthropic")
            self.provider_model = llm_config.get("model", "claude-3-7-sonnet-20250219")
            self.temperature = llm_config.get("temperature", 0.1)
            self.max_tokens = llm_config.get("max_tokens", 4000)
        else:
            # Use distillation config if available, otherwise use default
            llm_conf = config.llm.distillation or config.llm.default
            self.provider = llm_conf.provider
            self.provider_model = llm_conf.model
            self.temperature = llm_conf.temperature
            self.max_tokens = llm_conf.max_tokens
        
        # Initialize aisuite client
        self.client = ai.Client()
        
        # Full model name for aisuite (provider:model)
        self.model = f"{self.provider}:{self.provider_model}"
        
        self.max_chunk_tokens = max_chunk_tokens
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        
        # Load prompt templates from prompt_templates.py if available
        try:
            from llm_docs.distillation.prompt_templates import get_prompt_for_chunk
            self.get_prompt_for_chunk = get_prompt_for_chunk
        except ImportError:
            # Fallback to built-in templates
            self.main_prompt_template = """
I attached the COMPLETE documentation for {package_name}, the python library {package_description}.

The documentation contains a lot of "fluff" and references to things which we do NOT care about. I pasted the docs manually from the documentation website.  

Your goal is to meticulously read and understand everything and then distill it down into the most compact and clear form that would still be easily intelligible to YOU in a new conversation where you had never seen the original documentation text. It's too long to do in one shot, so let's do it in {num_parts} sections, starting now with the first part. 

Feel free to rearrange the order of sections so that it makes most sense to YOU since YOU will be the one consuming the output! Don't worry if it would be confusing or hard to follow by a human, it's not for a human, it's for YOU to quickly and efficiently convey all the relevant information in the documentation.

{chunk_content}
"""
            
            self.follow_up_prompt_template = """
continue with part {part_num} of {num_parts}

{chunk_content}
"""
            
            self.final_prompt_template = """
ok now make a part {part_num} with all the important stuff that you left out from the full docs in the {prev_parts_count} parts you already wrote above
"""

    def split_into_chunks(self, markdown_content: str, num_chunks: int = 5) -> List[Tuple[str, int]]:
        """
        Split the markdown content into chunks for processing.
        
        Args:
            markdown_content: Full markdown content
            num_chunks: Number of chunks to split into
            
        Returns:
            List of tuples with (markdown_chunk, estimated_tokens)
        """
        estimated_tokens = self.estimate_tokens(markdown_content)
        
        if estimated_tokens <= self.max_chunk_tokens:
            return [(markdown_content, estimated_tokens)]
        
        # Split by markdown headings
        sections = re.split(r'(#+\s+[^\n]+\n)', markdown_content)
        
        # Combine header with its content
        combined_sections = []
        current = ""
        
        for section in sections:
            if re.match(r'#+\s+[^\n]+\n', section):
                if current:
                    combined_sections.append(current)
                current = section
            else:
                current += section
                
        if current:
            combined_sections.append(current)
            
        # Now distribute sections into chunks
        chunks = [[] for _ in range(num_chunks)]
        section_tokens = [(section, len(section) / 4) for section in combined_sections]
        
        # Try to balance token count across chunks
        chunk_token_counts = [0] * num_chunks
        
        for section, tokens in section_tokens:
            # Find the chunk with the lowest token count
            min_index = chunk_token_counts.index(min(chunk_token_counts))
            chunks[min_index].append(section)
            chunk_token_counts[min_index] += tokens
            
        # Combine sections in each chunk
        return [("".join(chunk), count) for chunk, count in zip(chunks, chunk_token_counts, strict=False)]
    
    async def distill_chunk(self, 
                        package_name: str, 
                        chunk_content: str, 
                        part_num: int, 
                        num_parts: int, 
                        package_description: str = "",
                        doc_type: str = "general") -> str:
        """
        Distill a chunk of documentation using LLM.
        
        Args:
            package_name: Name of the package
            chunk_content: Content of the chunk to distill
            part_num: Part number (1-based)
            num_parts: Total number of parts
            package_description: Optional description of the package
            doc_type: Type of documentation (general, api, tutorial)
            
        Returns:
            Distilled documentation for the chunk
        """
        console.print(f"[cyan]Distilling chunk {part_num}/{num_parts} for {package_name}...[/cyan]")
        
        # Check cache first - with proper error handling
        cache_key = f"{package_name}_part{part_num}_of{num_parts}_{doc_type}"
        cache_key = re.sub(r'[^\w\-]', '_', cache_key)  # Sanitize for filename
        cache_file = self.cache_dir / f"{cache_key}.md"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_content = f.read()
                    if cached_content and len(cached_content) > 10:  # Ensure it's not an empty or corrupt file
                        console.print(f"[green]Using cached result for chunk {part_num}[/green]")
                        return cached_content
                    else:
                        console.print("[yellow]Cache file is empty or too small, regenerating...[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error reading cache: {str(e)}[/yellow]")

        # Make sure cache directory exists
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine which prompt template to use
        if hasattr(self, 'get_prompt_for_chunk'):
            # Use the imported prompt template function
            prompt = self.get_prompt_for_chunk(
                package_name=package_name,
                chunk_content=chunk_content,
                part_num=part_num,
                num_parts=num_parts,
                package_description=package_description,
                doc_type=doc_type
            )
        else:
            # Use the built-in templates
            if part_num == 1:
                prompt = self.main_prompt_template.format(
                    package_name=package_name,
                    package_description=package_description or "Python package",
                    num_parts=num_parts,
                    chunk_content=chunk_content
                )
            elif part_num <= num_parts:
                prompt = self.follow_up_prompt_template.format(
                    part_num=part_num,
                    num_parts=num_parts,
                    chunk_content=chunk_content
                )
            else:
                # Final part for anything missed
                prompt = self.final_prompt_template.format(
                    part_num=part_num,
                    prev_parts_count=num_parts
                )
        
        # Estimate prompt tokens
        estimated_prompt_tokens = self.estimate_tokens(prompt)
        console.print(f"[cyan]Estimated prompt tokens: {estimated_prompt_tokens:,}[/cyan]")
                
        # Send to LLM with robust retry logic
        retry_delay = self.retry_delay
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Prepare API parameters with error handling
                system_prompt_content = "You are an expert AI assistant. Your task is to distill Python library documentation into a concise, LLM-friendly format. Focus on accuracy and completeness, removing redundancy but preserving all critical technical information."
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt_content},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }

                # Track API call time
                start_time = time.time()

                # Use aisuite client with proper error handling
                response = self.client.chat.completions.create(**api_params) # Removed await

                # Calculate elapsed time
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Extract the response content with comprehensive fallbacks
                response_content = self._extract_response_content(response)

                if not response_content:
                    raise ValueError("Received empty response from LLM API")

                # Get completion token count
                completion_tokens = self.estimate_tokens(response_content)

                # Track token usage using the adapter
                update_aisuite_response_tracking(
                    response=response,  # Pass the actual response object
                    task=f"distill_{doc_type}_part{part_num}",
                    prompt_tokens=estimated_prompt_tokens,  # Fallback if not available in response
                    model=self.model
                )

                # Get completion token count for logging
                completion_tokens = self.estimate_tokens(response_content)
                console.print(f"[cyan]Response received in {elapsed_ms}ms, tokens: {completion_tokens:,}[/cyan]")
                
                # Rate limit to avoid hitting API limits
                await asyncio.sleep(1)
                
                # Save to cache with error handling
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(response_content)
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to write to cache: {str(e)}[/yellow]")
                    # Continue despite cache write failure
                
                return response_content
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]Error distilling chunk {part_num} (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {retry_delay}s...[/yellow]")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)  # Exponential backoff with a cap
                else:
                    error_msg = f"Failed to distill chunk {part_num} after {self.max_retries} attempts: {str(last_error)}"
                    console.print(f"[red]{error_msg}[/red]")                    
                    raise LLMAPIError(error_msg) from last_error
                
        # If we get here, all attempts failed - provide a useful error message in markdown
        error_message = f"""## Error Processing Part {part_num}

    Failed to process this section due to API errors after {self.max_retries} attempts.

    **Error:** {str(last_error)}

    Please check your API credentials, network connection, or try again later.
    """
        return error_message

    def _extract_response_content(self, response):
        """
        Extract content from an LLM API response with comprehensive fallbacks.
        
        Args:
            response: Response object from LLM API
            
        Returns:
            Text content from the response
        """
        # Try all possible response formats
        try:
            # Most common format for newer aisuite/anthropic
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    return response.choices[0].text
            
            # Alternative formats
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'completion'):
                return response.completion
            elif hasattr(response, 'answer'):
                return response.answer
            elif hasattr(response, 'generations') and len(response.generations) > 0:
                if hasattr(response.generations[0], 'text'):
                    return response.generations[0].text
                elif hasattr(response.generations[0], 'content'):
                    return response.generations[0].content
            
            # Last resort - try to convert the entire response to a string
            if response is not None:
                return str(response)
            
            # If nothing worked
            raise ValueError(f"Unknown response format: {type(response)}")
            
        except Exception as e:
            console.print(f"[red]Error extracting content from response: {e}[/red]")
            raise
        
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Text to estimate token count for
            
        Returns:
            Estimated number of tokens
        """
        try:
            
            # Determine which encoding to use based on model
            if "claude" in self.provider_model.lower():
                # Claude models roughly use 4 chars per token
                return len(text) // 4
            elif "gpt" in self.provider_model.lower():
                # For OpenAI models, use the appropriate encoding
                try:
                    if "gpt-4" in self.provider_model.lower():
                        encoding = tiktoken.encoding_for_model("gpt-4")
                    elif "gpt-3.5-turbo" in self.provider_model.lower():
                        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    else:
                        # Default to cl100k_base for newer models
                        encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception:
                    # If specific model encoding fails, use cl100k_base
                    try:
                        encoding = tiktoken.get_encoding("cl100k_base")
                        return len(encoding.encode(text))
                    except Exception:
                        # If that fails too, fall back to character count
                        return len(text) // 4
            elif "gemini" in self.provider_model.lower():
                # Google's Gemini models, approximate to 4 chars per token
                return len(text) // 4
            elif "mistral" in self.provider_model.lower() or "mixtral" in self.provider_model.lower():
                # Mistral models, approximate or use tiktoken
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception:
                    return len(text) // 4
            else:
                # Default case, try cl100k_base as a reasonable approximation
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception:
                    return len(text) // 4
        except ImportError:
            # If tiktoken is not available, fall back to character-based estimation
            return len(text) // 4

    async def distill_documentation(self, package: Package, doc_path: str) -> Optional[str]:
        """
        Distill the full documentation for a package.
        
        Args:
            package: Package object
            doc_path: Path to the combined markdown file
            
        Returns:
            Path to the distilled documentation file
        """
        console.print(f"[green]Distilling documentation for {package.name}...[/green]")
        
        # Check file size first to avoid memory issues
        file_size = Path(doc_path).stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50MB threshold
            console.print(f"[yellow]Documentation file is very large ({file_size/1024/1024:.2f} MB). This may impact performance.[/yellow]")
        
        # Use context manager for proper cleanup of temp directory
        with tempfile.TemporaryDirectory(prefix="llm_docs_") as temp_dir_path:
            temp_dir = Path(temp_dir_path)
            temp_file = temp_dir / f"{package.name}_distillation_in_progress.md"
            
            try:
                # Read the combined markdown file in chunks if it's large
                content = ""
                if file_size > 20 * 1024 * 1024:  # 20MB
                    # Process large file in chunks to avoid memory issues
                    console.print(f"[cyan]Reading large file in chunks ({file_size/1024/1024:.2f} MB)[/cyan]")
                    chunk_size = 10 * 1024 * 1024  # 10MB chunks
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        while chunk := f.read(chunk_size):
                            content += chunk
                else:
                    # For smaller files, read all at once
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Try to detect the type of documentation
                doc_type = "general"
                content_lower = content.lower()
                if "api reference" in content_lower or "class reference" in content_lower:
                    doc_type = "api"
                elif "tutorial" in content_lower or "guide" in content_lower:
                    doc_type = "tutorial"
                
                # Determine number of chunks based on content size
                num_chunks = 5
                if len(content) > 800000:  # For very large docs, use more chunks
                    num_chunks = 7
                elif len(content) < 200000:  # For smaller docs, use fewer chunks
                    num_chunks = 3
                    
                # Split into chunks
                chunks_with_tokens = self.split_into_chunks(content, num_chunks)
                chunks = [chunk for chunk, _ in chunks_with_tokens]
                
                # Start header for the distilled content
                combined_distilled = f"# {package.name} - Distilled LLM-Optimized Documentation\n\n"
                combined_distilled += f"*This is a condensed, LLM-optimized version of the documentation for {package.name}.*\n\n"
                combined_distilled += f"*Original documentation processed on: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
                combined_distilled += f"*Distilled using {self.model}*\n\n"
                
                # Write this initial header to the temp file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(combined_distilled)
                
                # Process each chunk with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TextColumn("[bold]{task.completed}/{task.total}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Distilling documentation...", total=num_chunks + 1)  # +1 for final part
                    
                    for i, chunk in enumerate(chunks, 1):
                        progress.update(task, description=f"Processing part {i}/{num_chunks}...")
                        
                        # Get package description if available
                        package_description = f"which {package.description}" if package.description else "Python package"
                        
                        # Distill the chunk
                        distilled_content = await self.distill_chunk(
                            package_name=package.name,
                            chunk_content=chunk,
                            part_num=i,
                            num_parts=num_chunks,
                            package_description=package_description,
                            doc_type=doc_type
                        )
                        
                        # Append to the temp file
                        with open(temp_file, 'a', encoding='utf-8') as f:
                            f.write(f"\n\n{distilled_content}")
                        
                        progress.update(task, completed=i)
                    
                    # Add a final part to capture anything missed
                    progress.update(task, description="Processing final review part...")
                    
                    final_part = await self.distill_chunk(
                        package_name=package.name,
                        chunk_content="",
                        part_num=num_chunks + 1,
                        num_parts=num_chunks,
                        package_description=package_description,
                        doc_type=doc_type
                    )
                    
                    # Append to the temp file
                    with open(temp_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n{final_part}")
                    
                    progress.update(task, completed=num_chunks + 1)
                
                # Read the complete file
                with open(temp_file, 'r', encoding='utf-8') as f:
                    combined_distilled = f.read()
                
                # Save the distilled documentation
                output_filename = f"{package.name}__distilled_lllm_documentation__as_of_{datetime.now().strftime('%m_%d_%Y')}.md"
                output_path = self.output_dir / output_filename
                
                # Ensure the output directory exists
                self.output_dir.mkdir(exist_ok=True, parents=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(combined_distilled)
                
                console.print(f"[green]Distillation complete! Output saved to {output_path}[/green]")
                return str(output_path)
                
            except Exception as e:
                console.print(f"[red]Error distilling documentation: {str(e)}[/red]")
                return None
            # No need for explicit cleanup in finally block - the tempfile.TemporaryDirectory context manager handles it
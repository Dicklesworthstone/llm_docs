"""
Module for distilling documentation using LLMs via aisuite.
"""

import asyncio
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aisuite as ai
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from llm_docs.config import config
from llm_docs.storage.models import Package

# Initialize console
console = Console()

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
        # Estimate token count (rough approximation: 4 chars = 1 token)
        estimated_tokens = len(markdown_content) / 4
        
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
        
        # Check cache first
        cache_key = f"{package_name}_part{part_num}_of{num_parts}_{doc_type}"
        cache_key = re.sub(r'[^\w\-]', '_', cache_key)  # Sanitize for filename
        cache_file = self.cache_dir / f"{cache_key}.md"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_content = f.read()
                    console.print(f"[green]Using cached result for chunk {part_num}[/green]")
                    return cached_content
            except Exception as e:
                console.print(f"[yellow]Error reading cache: {str(e)}[/yellow]")
        
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
        
        # Send to LLM with retry logic
        retry_delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                # Use aisuite client instead of Anthropic directly
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract the response content
                response_content = response.choices[0].message.content
                
                # Rate limit to avoid hitting API limits
                await asyncio.sleep(1)
                
                # Save to cache
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(response_content)
                except Exception as e:
                    console.print(f"[yellow]Error writing to cache: {str(e)}[/yellow]")
                
                return response_content
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    console.print(f"[yellow]Error distilling chunk {part_num}: {e}. Retrying in {retry_delay}s...[/yellow]")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    console.print(f"[red]Failed to distill chunk {part_num} after {self.max_retries} attempts: {e}[/red]")
                    return f"## Error Processing Part {part_num}\n\nFailed to process this section due to API error: {str(e)}"
    
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
        
        # Create a temporary file for intermediate results
        temp_dir = Path(tempfile.mkdtemp(prefix="llm_docs_"))
        temp_file = temp_dir / f"{package.name}_distillation_in_progress.md"
        
        try:
            # Read the combined markdown file
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
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_distilled)
            
            console.print(f"[green]Distillation complete! Output saved to {output_path}[/green]")
            return str(output_path)
            
        except Exception as e:
            console.print(f"[red]Error distilling documentation: {str(e)}[/red]")
            return None
        finally:
            # Clean up temporary files
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as e:
                    console.print(f"[yellow]Failed to delete temporary file: {e}[/yellow]")
                
            try:
                temp_dir.rmdir()
            except OSError as e:
                console.print(f"[yellow]Failed to delete temporary directory: {e}[/yellow]")
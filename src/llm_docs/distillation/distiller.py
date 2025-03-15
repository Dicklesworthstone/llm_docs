"""
Module for distilling documentation using Claude 3.7 Sonnet.
"""

import os
import re
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from termcolor import colored
from tqdm import tqdm

from anthropic import AsyncAnthropic

from llm_docs.storage.models import Package, DistillationJob

class DocumentationDistiller:
    """Distills documentation using Claude 3.7 Sonnet."""
    
    def __init__(self, output_dir: str = "distilled_docs", 
                 api_key: Optional[str] = None, 
                 max_chunk_tokens: int = 80000):
        """
        Initialize the documentation distiller.
        
        Args:
            output_dir: Directory to store distilled documentation
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_chunk_tokens: Maximum number of tokens per chunk
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")
            
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.max_chunk_tokens = max_chunk_tokens
        
        # Prompt templates from the README
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
    
    def split_into_chunks(self, markdown_content: str, num_chunks: int = 5) -> List[str]:
        """
        Split the markdown content into chunks for processing.
        
        Args:
            markdown_content: Full markdown content
            num_chunks: Number of chunks to split into
            
        Returns:
            List of markdown content chunks
        """
        # Estimate token count (rough approximation: 4 chars = 1 token)
        estimated_tokens = len(markdown_content) / 4
        
        if estimated_tokens <= self.max_chunk_tokens:
            return [markdown_content]
        
        # Split by markdown sections
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
        
        # Distribute sections to chunks based on length
        for i, section in enumerate(combined_sections):
            chunks[i % num_chunks].append(section)
            
        # Combine sections in each chunk
        return ["".join(chunk) for chunk in chunks]
    
    async def distill_chunk(self, package_name: str, chunk_content: str, 
                          part_num: int, num_parts: int, 
                          package_description: str = "") -> str:
        """
        Distill a chunk of documentation using Claude.
        
        Args:
            package_name: Name of the package
            chunk_content: Content of the chunk to distill
            part_num: Part number (1-based)
            num_parts: Total number of parts
            package_description: Optional description of the package
            
        Returns:
            Distilled documentation for the chunk
        """
        print(colored(f"Distilling chunk {part_num}/{num_parts} for {package_name}...", "cyan"))
        
        # Determine which prompt template to use
        if part_num == 1:
            prompt = self.main_prompt_template.format(
                package_name=package_name,
                package_description=package_description,
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
        
        # Send to Claude 3.7 Sonnet
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                message = await self.client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract the response content
                response_content = message.content[0].text
                
                # Rate limit to avoid hitting API limits
                await asyncio.sleep(1)
                
                return response_content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(colored(f"Error distilling chunk {part_num}: {e}. Retrying in {retry_delay}s...", "yellow"))
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(colored(f"Failed to distill chunk {part_num} after {max_retries} attempts: {e}", "red"))
                    return f"## Error Processing Part {part_num}\n\nFailed to process this section."
    
    async def distill_documentation(self, package: Package, doc_path: str) -> Optional[str]:
        """
        Distill the full documentation for a package.
        
        Args:
            package: Package object
            doc_path: Path to the combined markdown file
            
        Returns:
            Path to the distilled documentation file
        """
        print(colored(f"Distilling documentation for {package.name}...", "green"))
        
        # Read the combined markdown file
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        num_chunks = 5
        if len(content) > 800000:  # For very large docs, use more chunks
            num_chunks = 7
        elif len(content) < 200000:  # For smaller docs, use fewer chunks
            num_chunks = 3
            
        chunks = self.split_into_chunks(content, num_chunks)
        
        # Process each chunk
        distilled_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            distilled_content = await self.distill_chunk(
                package_name=package.name,
                chunk_content=chunk,
                part_num=i,
                num_parts=num_chunks,
                package_description=f"which {package.description if package.description else 'is a Python package'}"
            )
            
            distilled_parts.append(distilled_content)
        
        # Add a final part to capture anything missed
        final_part = await self.distill_chunk(
            package_name=package.name,
            chunk_content="",
            part_num=num_chunks + 1,
            num_parts=num_chunks
        )
        
        distilled_parts.append(final_part)
        
        # Combine all parts
        combined_distilled = f"# {package.name} - Distilled LLM-Optimized Documentation\n\n"
        combined_distilled += f"*This is a condensed, LLM-optimized version of the documentation for {package.name}.*\n\n"
        combined_distilled += f"*Original documentation processed on: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        combined_distilled += f"*Distilled using Claude 3.7 Sonnet*\n\n"
        combined_distilled += "\n\n".join(distilled_parts)
        
        # Save the distilled documentation
        output_filename = f"{package.name}__distilled_lllm_documentation__as_of_{datetime.now().strftime('%m_%d_%Y')}.md"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_distilled)
        
        return str(output_path)

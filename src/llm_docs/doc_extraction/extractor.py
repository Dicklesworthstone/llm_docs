"""
Module for extracting documentation from package websites and converting to markdown.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from browser_use import Browser
from bs4 import BeautifulSoup
from markitdown import markitdown
from termcolor import colored
from tqdm import tqdm

from llm_docs.storage.models import Package


class DocumentationExtractor:
    """Extracts documentation from package websites and converts to markdown."""
    
    def __init__(self, output_dir: str = "docs"):
        """Initialize the documentation extractor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        
    async def find_documentation_url(self, package_name: str) -> Optional[str]:
        """
        Find the documentation URL for a package using Google search.
        
        Args:
            package_name: Name of the package
            
        Returns:
            URL of the documentation site if found, None otherwise
        """
        print(colored(f"Finding documentation for {package_name}...", "cyan"))
        
        search_query = f"{package_name} python documentation"
        
        async with Browser() as browser:
            page = await browser.new_page()
            
            # Navigate to Google search
            await page.goto(f"https://www.google.com/search?q={search_query}")
            
            # Accept cookies if necessary
            try:
                accept_button = await page.wait_for_selector('button[aria-label="Accept all"]', timeout=3000)
                if accept_button:
                    await accept_button.click()
            except TimeoutError:
                pass  # No cookie banner found within timeout
                
            # Extract search results
            results = await page.evaluate("""
                () => {
                    const results = [];
                    const elements = document.querySelectorAll('a[href^="/url"]');
                    for (const el of elements) {
                        const url = new URL(el.href, window.location.origin);
                        const actualUrl = url.searchParams.get('q');
                        if (actualUrl && !actualUrl.includes('google.com')) {
                            results.push(actualUrl);
                        }
                    }
                    return results;
                }
            """)
            
            if not results:
                return None
                
            # Try to find the most likely documentation URL
            # Prioritize common documentation hosts
            doc_keywords = ['docs', 'documentation', 'readthedocs', 'github.io', 'rtfd.io']
            
            for keyword in doc_keywords:
                for url in results:
                    if keyword in url.lower():
                        return url
            
            # If no keyword match, return the first result
            return results[0]
    
    async def map_documentation_site(self, doc_url: str) -> List[str]:
        """
        Map out a documentation site to find all relevant pages.
        
        Args:
            doc_url: URL of the documentation site
            
        Returns:
            List of URLs to documentation pages in a logical order
        """
        print(colored(f"Mapping documentation site: {doc_url}", "cyan"))
        
        visited_urls = set()
        to_visit = [doc_url]
        doc_pages = []
        
        base_domain = urlparse(doc_url).netloc
        
        async with Browser() as browser:
            page = await browser.new_page()
            
            # Process the queue of URLs to visit
            with tqdm(desc="Mapping pages") as pbar:
                while to_visit and len(visited_urls) < 100:  # Limit to 100 pages
                    current_url = to_visit.pop(0)
                    
                    if current_url in visited_urls:
                        continue
                        
                    try:
                        await page.goto(current_url, timeout=20000)
                        visited_urls.add(current_url)
                        doc_pages.append(current_url)
                        pbar.update(1)
                        
                        # Extract links from the current page
                        links = await page.evaluate("""
                            () => {
                                const links = [];
                                document.querySelectorAll('a[href]').forEach(a => {
                                    links.push(a.href);
                                });
                                return links;
                            }
                        """)
                        
                        # Filter and normalize links
                        for link in links:
                            parsed = urlparse(link)
                            
                            # Skip external links, anchors, etc.
                            if parsed.netloc and parsed.netloc != base_domain:
                                continue
                            if not parsed.path or parsed.path == '/':
                                continue
                            if link.endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif', '.pdf')):
                                continue
                                
                            full_url = urljoin(current_url, link)
                            
                            if full_url not in visited_urls and full_url not in to_visit:
                                # Prioritize API documentation pages
                                if 'api' in full_url.lower() or 'reference' in full_url.lower():
                                    to_visit.insert(0, full_url)
                                else:
                                    to_visit.append(full_url)
                    except Exception as e:
                        print(colored(f"Error visiting {current_url}: {e}", "red"))
        
        # Sort pages to put API/reference documentation first
        doc_pages.sort(key=lambda url: 0 if ('api' in url.lower() or 'reference' in url.lower()) else 1)
        
        return doc_pages
    
    async def extract_content_from_page(self, url: str) -> str:
        """
        Extract the main content from a documentation page.
        
        Args:
            url: URL of the documentation page
            
        Returns:
            HTML content of the main documentation area
        """
        print(colored(f"Extracting content from: {url}", "cyan"))
        
        async with Browser() as browser:
            page = await browser.new_page()
            
            try:
                await page.goto(url, timeout=20000)
                
                # Get the page HTML
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove navigation, headers, footers, etc.
                for selector in ['nav', 'header', 'footer', '.sidebar', '.navigation', '.menu', 
                               '#sidebar', '#menu', '.toc', '#toc', '.search', '#search']:
                    for element in soup.select(selector):
                        element.extract()
                
                # Try to find the main content
                main_content = None
                for selector in ['main', 'article', '.content', '#content', '.documentation', '#documentation']:
                    content = soup.select_one(selector)
                    if content:
                        main_content = content
                        break
                
                # If no main content identified, use the body
                if not main_content:
                    main_content = soup.body
                
                # Convert to string
                content_html = str(main_content)
                
                return content_html
            except Exception as e:
                print(colored(f"Error extracting content from {url}: {e}", "red"))
                return ""
    
    async def convert_html_to_markdown(self, html: str) -> str:
        """
        Convert HTML content to Markdown.
        
        Args:
            html: HTML content
            
        Returns:
            Markdown representation of the HTML content
        """
        # Use markitdown to convert HTML to Markdown
        markdown = markitdown(html)
        
        # Clean up the markdown
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excess newlines
        
        return markdown
    
    async def process_package_documentation(self, package: Package) -> Optional[str]:
        """
        Process the documentation for a package.
        
        Args:
            package: Package object
            
        Returns:
            Path to the generated combined markdown file
        """
        print(colored(f"Processing documentation for {package.name}...", "green"))
        
        # Find documentation URL
        doc_url = await self.find_documentation_url(package.name)
        if not doc_url:
            print(colored(f"Could not find documentation for {package.name}", "yellow"))
            return None
        
        # Map the documentation site
        doc_pages = await self.map_documentation_site(doc_url)
        
        # Extract content from each page and convert to markdown
        combined_markdown = f"# {package.name} Documentation\n\n"
        combined_markdown += f"*This is the original documentation for {package.name}, combined from multiple pages.*\n\n"
        combined_markdown += f"*Source: {doc_url}*\n\n"
        combined_markdown += f"*Extraction date: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        for page_url in tqdm(doc_pages, desc="Extracting content"):
            html_content = await self.extract_content_from_page(page_url)
            if html_content:
                markdown_content = await self.convert_html_to_markdown(html_content)
                
                # Add page title and URL as a header
                page_title = page_url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
                if not page_title:
                    page_title = "Main Page"
                
                combined_markdown += f"## {page_title}\n\n"
                combined_markdown += f"*Source: {page_url}*\n\n"
                combined_markdown += markdown_content
                combined_markdown += "\n\n---\n\n"
        
        # Save the combined markdown
        output_filename = f"{package.name}__combined_original_documentation__as_of_{datetime.now().strftime('%m_%d_%Y')}.md"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_markdown)
        
        return str(output_path)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

"""
Module for extracting documentation from package websites and converting to markdown.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from browser_use import Browser
from bs4 import BeautifulSoup
from markitdown import markitdown
from rich.console import Console
from tqdm import tqdm

from llm_docs.storage.models import Package

# Initialize console
console = Console()

class DocumentationExtractor:
    """Extracts documentation from package websites and converts to markdown."""
    
    def __init__(self, output_dir: str = "docs", cache_dir: Optional[str] = None):
        """
        Initialize the documentation extractor.
        
        Args:
            output_dir: Directory to store extracted documentation
            cache_dir: Directory to cache intermediate results (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/doc_extraction")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        
        # Common documentation domains
        self.doc_domains = [
            'readthedocs.io', 'readthedocs.org', 
            'rtfd.io', 'rtfd.org',
            'docs.python.org',
            'github.io',
            'wiki.python.org'
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()        
        
    async def find_documentation_url(self, package_name: str) -> Optional[str]:
        """
        Find the documentation URL for a package using multiple strategies.
        
        Args:
            package_name: Name of the package
            
        Returns:
            URL of the documentation site if found, None otherwise
        """
        console.print(f"[cyan]Finding documentation for {package_name}...[/cyan]")
        
        # Check cache first
        cache_file = self.cache_dir / f"{package_name}_doc_url.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if "url" in cached_data and cached_data["url"]:
                        console.print(f"[green]Using cached documentation URL: {cached_data['url']}[/green]")
                        return cached_data["url"]
            except (json.JSONDecodeError, KeyError, OSError) as e:
                console.print(f"[yellow]Cache read error: {str(e)}. Will try other methods.[/yellow]")
        
        # List of strategies to try, in order of preference
        strategies = [
            self._try_pypi_api,
            self._try_readthedocs_url,
            self._try_google_search,
            self._try_github_repo
        ]
        
        # Try each strategy in order
        for strategy_func in strategies:
            try:
                url = await strategy_func(package_name)
                if url:
                    # Found a URL, cache it and return
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump({"url": url}, f)
                    except OSError as e:
                        console.print(f"[yellow]Failed to cache URL: {str(e)}[/yellow]")
                    return url
            except Exception as e:
                console.print(f"[yellow]Strategy failed: {str(e)}. Trying next...[/yellow]")
        
        # If all strategies failed, return None
        console.print(f"[red]Could not find documentation URL for {package_name}[/red]")
        return None

    async def _try_pypi_api(self, package_name: str) -> Optional[str]:
        """Try to find documentation URL from PyPI API."""
        try:
            response = await self.client.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                project_urls = data.get("info", {}).get("project_urls", {})
                
                # Check for documentation URLs in project_urls
                for key, url in project_urls.items():
                    if any(doc_term in key.lower() for doc_term in ["doc", "docs", "documentation"]):
                        console.print(f"[green]Found documentation URL from PyPI: {url}[/green]")
                        return url
                
                # Check for documentation URL in info
                doc_url = data.get("info", {}).get("documentation_url")
                if doc_url:
                    console.print(f"[green]Found documentation URL from PyPI info: {doc_url}[/green]")
                    return doc_url
                    
                # Try homepage as fallback if it looks like a documentation site
                homepage = data.get("info", {}).get("home_page")
                if homepage and any(doc_domain in homepage for doc_domain in self.doc_domains):
                    console.print(f"[green]Using homepage as documentation URL: {homepage}[/green]")
                    return homepage
            return None
        except Exception as e:
            console.print(f"[yellow]PyPI API error: {str(e)}[/yellow]")
            raise

    async def _try_readthedocs_url(self, package_name: str) -> Optional[str]:
        """Try common ReadTheDocs URL patterns."""
        # Common readthedocs URL patterns
        rtd_patterns = [
            f"https://{package_name}.readthedocs.io/en/latest/",
            f"https://{package_name}.readthedocs.io/en/stable/",
            f"https://{package_name.replace('_', '-')}.readthedocs.io/en/latest/",
            f"https://{package_name.replace('-', '_')}.readthedocs.io/en/latest/"
        ]
        
        for url in rtd_patterns:
            try:
                response = await self.client.head(url, timeout=5.0, follow_redirects=True)
                if response.status_code < 400:  # Any successful response
                    console.print(f"[green]Found ReadTheDocs URL: {url}[/green]")
                    return url
            except Exception:
                continue
        
        return None

    async def _try_google_search(self, package_name: str) -> Optional[str]:
        """Try to find documentation URL via Google search."""
        search_query = f"{package_name} python documentation"
        
        try:
            async with Browser() as browser:
                page = await browser.new_page()
                
                # Navigate to Google search
                await page.goto(f"https://www.google.com/search?q={search_query}")
                
                # Accept cookies if necessary
                try:
                    accept_button = await page.wait_for_selector('button[aria-label="Accept all"]', timeout=3000)
                    if accept_button:
                        await accept_button.click()
                except asyncio.TimeoutError:
                    pass  # No cookie banner or other issue
                    
                # Extract search results
                results = await page.evaluate("""
                    () => {
                        const results = [];
                        document.querySelectorAll('a[href^="/url"]').forEach(a => {
                            const url = new URL(a.href, window.location.origin);
                            const actualUrl = url.searchParams.get('q');
                            if (actualUrl && !actualUrl.includes('google.com')) {
                                results.push(actualUrl);
                            }
                        });
                        return results;
                    }
                """)
                
                if not results:
                    return None
                    
                # Check each result using multiple heuristics
                for url in results:
                    url_lower = url.lower()
                    package_lower = package_name.lower()
                    
                    # First priority: URL contains package name and documentation terms
                    if package_lower in url_lower and any(term in url_lower for term in ["doc", "documentation", "manual", "guide", "tutorial"]):
                        console.print(f"[green]Found documentation URL from Google search: {url}[/green]")
                        return url
                
                # Second priority: Domain-based match with common documentation sites
                for url in results:
                    if any(doc_domain in url for doc_domain in self.doc_domains):
                        console.print(f"[green]Found documentation URL from domain match: {url}[/green]")
                        return url
                
                # Fall back to first result if nothing else matches
                if results:
                    first_url = results[0]
                    console.print(f"[yellow]Using first search result as documentation URL: {first_url}[/yellow]")
                    return first_url
                
                return None
        
        except Exception as e:
            console.print(f"[yellow]Google search error: {str(e)}[/yellow]")
            raise

    async def _try_github_repo(self, package_name: str) -> Optional[str]:
        """Try to find documentation in the GitHub repository if it exists."""
        # Try GitHub API to find repository
        try:
            response = await self.client.get(f"https://api.github.com/search/repositories?q={package_name}+language:python", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if data.get("items") and len(data["items"]) > 0:
                    repo = data["items"][0]
                    repo_name = repo.get("full_name")
                    
                    # Check typical documentation locations in GitHub repos
                    doc_locations = [
                        f"https://github.com/{repo_name}/blob/main/docs/README.md",
                        f"https://github.com/{repo_name}/blob/master/docs/README.md",
                        f"https://github.com/{repo_name}/blob/main/docs/index.md",
                        f"https://github.com/{repo_name}/blob/master/docs/index.md",
                        f"https://github.com/{repo_name}/tree/main/docs",
                        f"https://github.com/{repo_name}/tree/master/docs",
                        f"https://github.com/{repo_name}/wiki"
                    ]
                    
                    # Try each location
                    for url in doc_locations:
                        try:
                            resp = await self.client.head(url, timeout=5.0)
                            if resp.status_code < 400:
                                console.print(f"[green]Found documentation in GitHub repository: {url}[/green]")
                                return url
                        except Exception:
                            continue
        
        except Exception as e:
            console.print(f"[yellow]GitHub API error: {str(e)}[/yellow]")
            
        return None
    
    async def map_documentation_site(self, doc_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Map out a documentation site to find all relevant pages.
        
        Args:
            doc_url: URL of the documentation site
            max_pages: Maximum number of pages to map
            
        Returns:
            List of dictionaries with information about documentation pages
        """
        console.print(f"[cyan]Mapping documentation site: {doc_url}[/cyan]")
        
        # Parse the URL to get the base domain
        parsed_url = urlparse(doc_url)
        base_domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Set to track visited and queued URLs
        visited_urls = set()
        to_visit = [doc_url]
        doc_pages = []
        
        # Check for robots.txt
        try:
            robots_url = f"{base_url}/robots.txt"
            response = await self.client.get(robots_url)
            if response.status_code == 200:
                content = response.text
                # Simple check if Googlebot is disallowed (we'll respect the same rules)
                if "User-agent: Googlebot" in content and "Disallow: /" in content:
                    console.print("[yellow]This site disallows crawling in robots.txt. Limiting to main page only.[/yellow]")
                    # In this case, we'll just process the main page
                    doc_pages.append({
                        "url": doc_url,
                        "title": "Main Documentation",
                        "priority": 0  # Highest priority
                    })
                    return doc_pages
        except Exception:
            pass  # If we can't check robots.txt, proceed with caution
        
        async with Browser() as browser:
            page = await browser.new_page()
            
            # Set a reasonable request interval to avoid overloading the server
            request_interval = 2.0  # seconds
            
            # Configure page for better performance
            await page.set_viewport_size({"width": 1280, "height": 800})
            
            # Process the queue of URLs to visit
            with tqdm(desc="Mapping pages", total=max_pages) as pbar:
                while to_visit and len(visited_urls) < max_pages:
                    current_url = to_visit.pop(0)
                    
                    if current_url in visited_urls:
                        continue
                    
                    # Add the URL to visited set before attempting to visit it
                    visited_urls.add(current_url)
                    
                    try:
                        await page.goto(current_url, timeout=20000)
                        await asyncio.sleep(request_interval)  # Be polite to the server
                        
                        # Get page title
                        title = await page.title()
                        
                        # Determine page priority:
                        # - API/reference documentation gets highest priority
                        # - Guides/tutorials get medium priority
                        # - Others get default priority
                        priority = 2  # Default priority
                        url_lower = current_url.lower()
                        
                        if "api" in url_lower or "reference" in url_lower:
                            priority = 0  # Highest priority
                        elif any(term in url_lower for term in ["guide", "tutorial", "quickstart", "getting-started"]):
                            priority = 1  # Medium priority
                        
                        # Add to doc_pages
                        doc_pages.append({
                            "url": current_url,
                            "title": title,
                            "priority": priority
                        })
                        
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
                            
                            # Skip external links
                            if parsed.netloc and parsed.netloc != base_domain:
                                continue
                            
                            # Skip anchors, query parameters, etc.
                            if not parsed.path or parsed.path == '/':
                                continue
                                
                            # Skip media files, PDFs, etc.
                            if link.endswith(('.png', '.jpg', '.jpeg', '.svg', '.gif', '.pdf', '.zip', '.tar.gz')):
                                continue
                            
                            # Skip some common non-documentation paths
                            if any(path in parsed.path for path in ["/search", "/download", "/community", "/blog"]):
                                continue
                                
                            full_url = urljoin(current_url, link)
                            
                            # Remove anchor and query parameters for comparison
                            clean_url = full_url.split('#')[0].split('?')[0]
                            
                            if clean_url not in visited_urls and clean_url not in to_visit:
                                # Prioritize API documentation pages
                                if "api" in clean_url.lower() or "reference" in clean_url.lower():
                                    to_visit.insert(0, clean_url)
                                else:
                                    to_visit.append(clean_url)
                    except Exception as e:
                        console.print(f"[yellow]Error visiting {current_url}: {str(e)}[/yellow]")
        
        # Sort pages by priority
        doc_pages.sort(key=lambda x: x["priority"])
        
        console.print(f"[green]Found {len(doc_pages)} documentation pages[/green]")
        return doc_pages
    
    async def extract_content_from_page(self, url: str) -> str:
        """
        Extract the main content from a documentation page.
        
        Args:
            url: URL of the documentation page
            
        Returns:
            HTML content of the main documentation area
        """
        try:
            async with Browser() as browser:
                page = await browser.new_page()
                
                # Configure page
                await page.set_viewport_size({"width": 1280, "height": 800})
                
                try:
                    await page.goto(url, timeout=30000)
                    
                    # Wait for content to load
                    await asyncio.sleep(1)
                    
                    # Get the page HTML
                    html = await page.content()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Remove navigation, headers, footers, etc.
                    selectors_to_remove = [
                        'nav', 'header', 'footer', '.sidebar', '.navigation', '.menu', 
                        '#sidebar', '#menu', '.toc', '#toc', '.search', '#search',
                        '.breadcrumbs', '.breadcrumb', '.admonition', '.warning', 
                        '.note', '.header-bar', '.installation',
                        '.headerlink', '.editlink', '.social-links',
                        'script', 'style'
                    ]
                    
                    for selector in selectors_to_remove:
                        for element in soup.select(selector):
                            element.extract()
                    
                    # Try to find the main content using common selectors
                    main_content_selectors = [
                        'main', 'article', '.content', '#content', '.documentation', '#documentation',
                        '.document', '#document', '.section', '.body', '.page-content', 
                        '.main-content', '#main-content', '.markdown-body', '.rst-content'
                    ]
                    
                    main_content = None
                    for selector in main_content_selectors:
                        content = soup.select_one(selector)
                        if content and len(str(content)) > 500:  # Must have meaningful content
                            main_content = content
                            break
                    
                    # If no main content identified, use the body minus the removed elements
                    if not main_content or len(str(main_content)) < 500:
                        main_content = soup.body
                    
                    # Convert to string
                    content_html = str(main_content)
                    
                    # Use a regex to remove HTML comments
                    content_html = re.sub(r'<!--.*?-->', '', content_html, flags=re.DOTALL)
                    
                    return content_html
                    
                except Exception as e:
                    console.print(f"[yellow]Error accessing {url}: {str(e)}[/yellow]")
                    return f"<h1>Error accessing page</h1><p>Could not access {url}: {str(e)}</p>"
        except Exception as e:
            console.print(f"[red]Browser error for {url}: {str(e)}[/red]")
            return f"<h1>Browser error</h1><p>Could not process {url}: {str(e)}</p>"
    
    async def convert_html_to_markdown(self, html: str, url: str = "") -> str:
        """
        Convert HTML content to Markdown.
        
        Args:
            html: HTML content
            url: Source URL (for reference)
            
        Returns:
            Markdown representation of the HTML content
        """
        try:
            # Use markitdown to convert HTML to Markdown
            markdown = markitdown(html)
            
            # Clean up the markdown
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excess newlines
            
            # Fix common markitdown conversion issues
            markdown = re.sub(r'\\\[', '[', markdown)  # Fix escaped brackets
            markdown = re.sub(r'\\\]', ']', markdown)  # Fix escaped brackets
            markdown = re.sub(r'\\_', '_', markdown)    # Fix escaped underscores
            
            # Add source URL reference
            if url:
                markdown = f"Source: {url}\n\n{markdown}"
            
            return markdown
            
        except Exception as e:
            console.print(f"[yellow]Error converting HTML to Markdown: {str(e)}[/yellow]")
            # Return a simplified version if conversion fails
            return f"Source: {url}\n\n{html}"
    
    async def process_package_documentation(self, package: Package) -> Optional[str]:
        """
        Process the documentation for a package.
        
        Args:
            package: Package object
            
        Returns:
            Path to the generated combined markdown file
        """
        console.print(f"[green]Processing documentation for {package.name}...[/green]")
        
        # Find documentation URL
        doc_url = await self.find_documentation_url(package.name)
        if not doc_url:
            console.print(f"[yellow]Could not find documentation for {package.name}[/yellow]")
            return None
        
        # Store the docs URL in the package
        package.docs_url = doc_url
        
        # Map the documentation site
        doc_pages = await self.map_documentation_site(doc_url)
        
        if not doc_pages:
            console.print(f"[yellow]No documentation pages found for {package.name}[/yellow]")
            return None
        
        # Extract content from each page and convert to markdown
        combined_markdown = f"# {package.name} Documentation\n\n"
        combined_markdown += f"*This is the original documentation for {package.name}, combined from multiple pages.*\n\n"
        combined_markdown += f"*Main documentation source: {doc_url}*\n\n"
        combined_markdown += f"*Extraction date: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        # Use semaphore to limit concurrent browser instances
        semaphore = asyncio.Semaphore(5)
        
        async def process_page(page_info):
            async with semaphore:
                page_url = page_info["url"]
                title = page_info.get("title", page_url.split('/')[-1].replace('-', ' ').replace('_', ' ').title())
                if not title:
                    title = "Documentation Page"
                
                html_content = await self.extract_content_from_page(page_url)
                if html_content:
                    markdown_content = await self.convert_html_to_markdown(html_content, page_url)
                    
                    return f"## {title}\n\n{markdown_content}\n\n---\n\n"
                return ""
        
        # Process pages concurrently
        tasks = [process_page(page) for page in doc_pages]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        for markdown_section in results:
            combined_markdown += markdown_section
        
        # Save the combined markdown
        output_filename = f"{package.name}__combined_original_documentation__as_of_{datetime.now().strftime('%m_%d_%Y')}.md"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_markdown)
        
        console.print(f"[green]Documentation saved to {output_path}[/green]")
        return str(output_path)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

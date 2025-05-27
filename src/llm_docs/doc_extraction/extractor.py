"""
Module for extracting documentation from package websites and converting to markdown.
Improved with better error handling, performance optimizations, and integration
with logging and token tracking systems.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx
from browser_use import Agent, Browser
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from markitdown import _markitdown as markitdown
from rich.console import Console

from llm_docs.storage.models import Package
from llm_docs.utils.llm_docs_logger import logger
from llm_docs.utils.token_tracking import tracker

# Initialize console
console = Console()

class DocumentationExtractionError(Exception):
    """Base exception for documentation extraction errors."""
    pass

class URLDiscoveryError(DocumentationExtractionError):
    """Exception raised when documentation URL cannot be found."""
    pass

class ContentExtractionError(DocumentationExtractionError):
    """Exception raised when content extraction fails."""
    pass

class MarkdownConversionError(DocumentationExtractionError):
    """Exception raised when HTML to Markdown conversion fails."""
    pass

class DocumentationExtractor:
    """Extracts documentation from package websites and converts to markdown."""
    
    def __init__(self, output_dir: str = "docs", cache_dir: Optional[str] = None, 
                 concurrency_limit: int = 3, use_vision: bool = True):
        """
        Initialize the documentation extractor.
        
        Args:
            output_dir: Directory to store extracted documentation
            cache_dir: Directory to cache intermediate results (optional)
            concurrency_limit: Maximum number of concurrent extraction operations
            use_vision: Whether to use vision capabilities for browser automation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/doc_extraction")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.client = httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36"
            }
        )
        
        # Common documentation domains
        self.doc_domains = [
            'readthedocs.io', 'readthedocs.org', 
            'rtfd.io', 'rtfd.org',
            'docs.python.org',
            'github.io',
            'wiki.python.org',
            'docs.rs',
            'mkdocs.org',
            'gitbook.io'
        ]
        
        # Initialize browser configuration
        self.browser_config = BrowserConfig(
            headless=True,
            disable_security=True,
            new_context_config=BrowserContextConfig(
                wait_for_network_idle_page_load_time=3.0,
                browser_window_size={'width': 1280, 'height': 1100},
                highlight_elements=False
            )
        )
        
        # Initialize the LLM model for the agent with proper defaults
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1
            )
        except Exception as e:
            logger.warning(f"Failed to initialize default LLM: {e}. Will fall back to default.")
            self.llm = None
        
        # Concurrency control
        self.concurrency_limit = concurrency_limit
        self.extraction_semaphore = asyncio.Semaphore(concurrency_limit)
        
        # Use vision capabilities for browser automation
        self.use_vision = use_vision
        
        # Track processed URLs to avoid duplicates
        self.processed_urls: Set[str] = set()
        
        # Rate limiters
        self.google_rate_limiter = self._create_rate_limiter(rate=0.5)  # Max 1 request per 2 seconds
        self.github_rate_limiter = self._create_rate_limiter(rate=0.2)  # Max 1 request per 5 seconds
        self.pypi_rate_limiter = self._create_rate_limiter(rate=1.0)    # Max 1 request per second
        self.general_rate_limiter = self._create_rate_limiter(rate=0.5) # Default rate limit
        
        # Cache settings
        self.cache_ttl = timedelta(days=7)  # Cache TTL for URL discovery
        self.extraction_cache_ttl = timedelta(days=30)  # Cache TTL for content extraction

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _create_rate_limiter(self, rate: float = 1.0, burst: int = 1):
        """Create a token bucket rate limiter."""
        class RateLimiter:
            def __init__(self, rate, burst):
                self.rate = rate  # tokens per second
                self.burst = burst  # max tokens
                self.tokens = burst  # current tokens
                self.last_update = time.monotonic()
                self.lock = asyncio.Lock()
                
            async def acquire(self):
                async with self.lock:
                    now = time.monotonic()
                    elapsed = now - self.last_update
                    self.last_update = now
                    
                    # Add tokens based on elapsed time
                    self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                    
                    if self.tokens >= 1:
                        self.tokens -= 1
                        return
                        
                    # Not enough tokens, sleep until we have one
                    wait_time = (1 - self.tokens) / self.rate
                    logger.rate_limit(wait_time)
                    await asyncio.sleep(wait_time)
                    self.tokens = 0  # We've used our token
        
        return RateLimiter(rate, burst)

    async def find_documentation_url(self, package_name: str) -> Optional[str]:
        """
        Find the documentation URL for a package using multiple strategies.
        
        Args:
            package_name: Name of the package
            
        Returns:
            URL of the documentation site if found, None otherwise
        """
        logger.doc_extraction_start(package_name)
        
        # Check cache first with TTL validation
        cache_file = self.cache_dir / f"{package_name}_doc_url.json"
        if cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if "url" in cached_data and cached_data["url"]:
                            url = cached_data["url"]
                            logger.info(f"Using cached documentation URL for {package_name}: {url}")
                            return url
                else:
                    logger.info(f"Cache expired for {package_name}. Refreshing...")
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Cache read error for {package_name}: {str(e)}. Will try other methods.")
        
        # List of strategies to try, in order of preference
        strategies = [
            self._try_pypi_api,
            self._try_readthedocs_url,
            self._try_google_search,
            self._try_github_repo
        ]
        
        # Initialize a list to store errors for comprehensive error reporting
        errors = []
        
        # Try each strategy in order with rate limiting
        for strategy_func in strategies:
            try:
                strategy_name = strategy_func.__name__.replace("_try_", "")
                logger.info(f"Trying to find docs using {strategy_name} for {package_name}...")
                
                # Apply strategy-specific rate limiting
                if strategy_name == 'pypi_api':
                    await self.pypi_rate_limiter.acquire()
                elif strategy_name == 'github_repo':
                    await self.github_rate_limiter.acquire()
                elif strategy_name == 'google_search':
                    await self.google_rate_limiter.acquire()
                else:
                    await self.general_rate_limiter.acquire()
                
                url = await strategy_func(package_name)
                if url:
                    # Found a URL, cache it and return
                    try:
                        os.makedirs(self.cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump({"url": url, "timestamp": datetime.now().isoformat()}, f)
                    except OSError as e:
                        logger.warning(f"Failed to cache URL for {package_name}: {str(e)}")
                    
                    logger.doc_url_found(package_name, url)
                    return url
            except Exception as e:
                error_msg = f"Strategy {strategy_func.__name__} failed for {package_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                
                # Add a small delay between strategies
                await asyncio.sleep(0.5)
        
        # If all strategies failed, log a comprehensive error and return None
        logger.error(f"Could not find documentation URL for {package_name}. Errors: {', '.join(errors)}")
        return None

    async def _try_pypi_api(self, package_name: str) -> Optional[str]:
        """Try to find documentation URL from PyPI API."""
        try:
            logger.api_call("PyPI", f"pypi/{package_name}/json")
            response = await self.client.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                
                # Check multiple sources in order of preference
                
                # 1. Direct documentation_url field
                doc_url = data.get("info", {}).get("documentation_url")
                if doc_url and self._is_valid_url(doc_url):
                    logger.info(f"Found documentation URL from PyPI documentation_url for {package_name}: {doc_url}")
                    return doc_url
                
                # 2. Project URLs dictionary with documentation-related keys
                project_urls = data.get("info", {}).get("project_urls", {})
                doc_keys = ["documentation", "docs", "doc", "document", "rtd", "readthedocs"]
                
                for key, url in project_urls.items():
                    if any(doc_term.lower() in key.lower() for doc_term in doc_keys) and self._is_valid_url(url):
                        logger.info(f"Found documentation URL from PyPI project_urls for {package_name}: {url}")
                        return url
                
                # 3. Homepage if it looks like a documentation site
                homepage = data.get("info", {}).get("home_page")
                if homepage and self._is_valid_url(homepage) and any(doc_domain in homepage.lower() for doc_domain in self.doc_domains):
                    logger.info(f"Using homepage as documentation URL for {package_name}: {homepage}")
                    return homepage
                    
                # 4. See if there's a GitHub repo in project URLs and try documentation in GitHub
                for _, url in project_urls.items():
                    if "github" in url.lower() and self._is_valid_url(url):
                        # Check if there's a docs directory or wiki
                        github_components = url.split('github.com/')
                        if len(github_components) == 2:
                            repo_path = github_components[1].strip('/')
                            docs_candidates = [
                                f"https://github.com/{repo_path}/tree/main/docs",
                                f"https://github.com/{repo_path}/tree/master/docs",
                                f"https://github.com/{repo_path}/wiki",
                                f"https://{repo_path.split('/')[0]}.github.io/{repo_path.split('/')[-1]}"
                            ]
                            
                            for candidate in docs_candidates:
                                try:
                                    await self.github_rate_limiter.acquire()
                                    resp = await self.client.head(candidate, timeout=5.0)
                                    if resp.status_code < 400:
                                        logger.info(f"Found documentation in GitHub for {package_name}: {candidate}")
                                        return candidate
                                except Exception:
                                    continue
            return None
        except Exception as e:
            logger.warning(f"PyPI API error for {package_name}: {str(e)}")
            raise

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is syntactically valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def _try_readthedocs_url(self, package_name: str) -> Optional[str]:
        """Try common ReadTheDocs URL patterns."""
        # Common readthedocs URL patterns
        package_name_variants = [
            package_name,
            package_name.replace('_', '-'),
            package_name.replace('-', '_'),
            package_name.lower(),
            # Handle special cases for packages with dots
            package_name.replace('.', '-'),
            package_name.replace('.', '_'),
        ]
        
        subdomains = [name for name in package_name_variants]
        
        # Try some common project name transformations
        if package_name.startswith('python-'):
            subdomains.append(package_name[7:])
        
        # Try with different versions
        versions = ['latest', 'stable', 'main', 'master']
        
        # Generate combinations of subdomain and version
        rtd_patterns = []
        for subdomain in subdomains:
            for version in versions:
                rtd_patterns.append(f"https://{subdomain}.readthedocs.io/en/{version}/")
        
        # Try each pattern
        for url in rtd_patterns:
            try:
                logger.browser_action(f"Checking ReadTheDocs URL for {package_name}", url=url)
                response = await self.client.head(url, timeout=5.0, follow_redirects=True)
                if response.status_code < 400:  # Any successful response
                    logger.info(f"Found ReadTheDocs URL for {package_name}: {url}")
                    return url
            except Exception:
                continue
        
        return None

    async def _try_google_search(self, package_name: str) -> Optional[str]:
        """Try to find documentation URL via Google search."""
        search_query = f"{package_name} python documentation"
        
        logger.info(f"Searching for documentation for {package_name} using Google...")
        
        try:
            # Track token usage for the agent
            agent_start_tokens = tracker.get_total_tokens()
            
            # Track token usage for the agent
            agent_start_tokens = tracker.get_total_tokens()
            
            # Set up initial actions to perform Google search
            initial_actions = [
                {'open_tab': {'url': f"https://www.google.com/search?q={search_query}"}}
            ]

            # Create an agent, letting browser-use handle browser lifecycle
            agent = Agent(
                task=f"Search for the official documentation site for the Python package '{package_name}'. Look for links to readthedocs.io, official websites with documentation sections, or GitHub documentation. Return only the URL to the main documentation page (not a specific section or article).",
                llm=self.llm,
                use_vision=self.use_vision,
                initial_actions=initial_actions # Pass initial_actions to constructor
            )
            
            # Run the agent
            logger.browser_action(f"Running search agent for {package_name}", url=f"https://www.google.com/search?q={search_query}")
            await agent.browser_session.start() 
            result = await agent.run() # Call run() with no arguments
            
            # Calculate token usage
            agent_end_tokens = tracker.get_total_tokens()
            tokens_used = agent_end_tokens - agent_start_tokens
            logger.token_usage(
                prompt_tokens=tokens_used * 0.7,  # Approximate split between prompt and completion
                completion_tokens=tokens_used * 0.3,
                model="gpt-4o"
            )
            
            # Extract the URLs from the agent's output
            extracted_content = result.final_result()
            
            if not extracted_content:
                logger.warning(f"Agent couldn't find documentation URL for {package_name}")
                return None
                
            # Extract URLs using comprehensive regex pattern
            urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!.~\'()*+,;=:@/\\&?#]*)?', extracted_content)
            
            if not urls:
                urls = re.findall(r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!.~\'()*+,;=:@/\\&?#]*)?', extracted_content)
                urls = [f"https://{url}" for url in urls]
            
            if not urls:
                logger.warning(f"No URLs found in agent's response for {package_name}")
                return None
                
            # Filter URLs to prioritize documentation sites
            for url in urls:
                url_lower = url.lower()
                package_lower = package_name.lower()
                
                # First priority: URL contains package name and is on a documentation domain
                if package_lower in url_lower and any(doc_domain in url_lower for doc_domain in self.doc_domains):
                    logger.info(f"Found documentation URL from agent search for {package_name}: {url}")
                    return url
            
            # Second priority: URL contains package name and documentation terms
            for url in urls:
                url_lower = url.lower()
                package_lower = package_name.lower()
                if package_lower in url_lower and any(term in url_lower for term in ["doc", "documentation", "manual", "guide", "tutorial"]):
                    logger.info(f"Found documentation URL from agent search for {package_name}: {url}")
                    return url
            
            # Third priority: Domain-based match with common documentation sites
            for url in urls:
                if any(doc_domain in url for doc_domain in self.doc_domains):
                    logger.info(f"Found documentation URL from domain match for {package_name}: {url}")
                    return url
            
            # Fall back to first URL if nothing else matches
            if urls:
                first_url = urls[0]
                logger.info(f"Using first URL from agent's response as documentation URL for {package_name}: {first_url}")
                return first_url
            
            return None
                
        except Exception as e:
            logger.error(f"Agent search error for {package_name}: {str(e)}")
            raise
        finally:
            if 'agent' in locals() and hasattr(agent, 'browser_session') and agent.browser_session:
                await agent.browser_session.close()

    async def _try_github_repo(self, package_name: str) -> Optional[str]:
        """Try to find documentation in the GitHub repository if it exists."""
        # Try GitHub API to find repository
        try:
            # Use rate limiting for GitHub API requests to avoid hitting limits
            await self.github_rate_limiter.acquire()
            
            logger.api_call("GitHub", f"search/repositories?q={package_name}+language:python")
            response = await self.client.get(
                f"https://api.github.com/search/repositories?q={package_name}+language:python", 
                timeout=10.0,
                headers={"Accept": "application/vnd.github.v3+json"}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("items") and len(data["items"]) > 0:
                    # Find the repository that most closely matches the package name
                    best_repo = None
                    for repo in data["items"]:
                        repo_name = repo.get("name", "").lower()
                        if repo_name == package_name.lower():
                            best_repo = repo
                            break
                        elif package_name.lower() in repo_name or repo_name in package_name.lower():
                            if not best_repo:
                                best_repo = repo
                    
                    if not best_repo and data["items"]:
                        best_repo = data["items"][0]  # Fall back to first result
                    
                    if best_repo:
                        repo_name = best_repo.get("full_name")
                        repo_url = best_repo.get("html_url")
                        logger.info(f"Found GitHub repository for {package_name}: {repo_name}")
                        
                        # Check if there's a GitHub Pages site
                        pages_url = f"https://{repo_name.split('/')[0]}.github.io/{repo_name.split('/')[1]}"
                        try:
                            await self.github_rate_limiter.acquire()
                            pages_response = await self.client.head(pages_url, timeout=5.0)
                            if pages_response.status_code < 400:
                                logger.info(f"Found GitHub Pages site for {package_name}: {pages_url}")
                                return pages_url
                        except Exception:
                            pass  # Continue to other methods if GitHub Pages not found
                        
                        # Check typical documentation locations in GitHub repos
                        doc_locations = [
                            f"{repo_url}/blob/main/docs/README.md",
                            f"{repo_url}/blob/master/docs/README.md",
                            f"{repo_url}/blob/main/docs/index.md",
                            f"{repo_url}/blob/master/docs/index.md",
                            f"{repo_url}/tree/main/docs",
                            f"{repo_url}/tree/master/docs",
                            f"{repo_url}/wiki"
                        ]
                        
                        # Try each location
                        for url in doc_locations:
                            try:
                                await self.github_rate_limiter.acquire()
                                resp = await self.client.head(url, timeout=5.0)
                                if resp.status_code < 400:
                                    logger.info(f"Found documentation in GitHub repository for {package_name}: {url}")
                                    return url
                            except Exception:
                                continue
        
        except Exception as e:
            logger.warning(f"GitHub API error for {package_name}: {str(e)}")
            
        return None
    
    async def map_documentation_site(self, doc_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """
        Map out a documentation site to find all relevant pages using an AI agent.
        
        Args:
            doc_url: URL of the documentation site
            max_pages: Maximum number of pages to map
            
        Returns:
            List of dictionaries with information about documentation pages
        """
        logger.info(f"Mapping documentation site structure: {doc_url}")
        
        # Parse the URL to get the base domain
        parsed_url = urlparse(doc_url)
        base_domain = parsed_url.netloc
        
        # Check cache with TTL validation
        cache_key = f"site_map_{base_domain}_{self._hash_url(doc_url)}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        if isinstance(cached_data, list) and len(cached_data) > 0:
                            logger.info(f"Using cached site map for {doc_url} with {len(cached_data)} pages")
                            return cached_data
                else:
                    logger.info(f"Site map cache expired for {doc_url}. Refreshing...")
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Cache read error for site map {doc_url}: {str(e)}. Will regenerate.")
        
        # Track token usage for the agent
        # Track token usage for the agent
        agent_start_tokens = tracker.get_total_tokens()
        
        try:
            # Set up initial actions to visit the documentation site
            initial_actions = [
                {'open_tab': {'url': doc_url}}
            ]

            # Create an agent, letting browser-use handle browser lifecycle
            agent = Agent(
                task=f"""
                Map this documentation site for the Python package and find all important pages. Focus on:
                
                1. Main API reference documentation
                2. Core function and class definitions
                3. Getting started guides and tutorials
                4. Installation instructions
                5. Common usage examples
                
                For each page, save its URL, title, and assign a priority:
                - 0 (highest): API reference, class/function documentation
                - 1 (medium): User guides, tutorials, examples
                - 2 (low): Other related pages
                
                Stay only on {base_domain} and focus on content directly related to the package.
                Ignore external links, blog posts, and community pages unless they contain core documentation.
                
                Return a structured list of pages with title, URL, and priority.
                """,
                llm=self.llm,
                use_vision=self.use_vision,
                initial_actions=initial_actions # Pass initial_actions to constructor
            )
            
            # Run the agent
            logger.browser_action(f"Running site mapping agent for {doc_url}")
            await agent.browser_session.start() 
            result = await agent.run() # Call run() with no arguments
            
            # Calculate token usage
            agent_end_tokens = tracker.get_total_tokens()
            tokens_used = agent_end_tokens - agent_start_tokens
            logger.token_usage(
                prompt_tokens=tokens_used * 0.7,  # Approximate split between prompt and completion
                completion_tokens=tokens_used * 0.3,
                model="gpt-4o"
            )
            
            # Try to extract site structure from the agent's result
            extracted_content = result.final_result()
            visited_urls = result.urls()
            screenshots = result.screenshots()
            
            logger.info(f"Agent visited {len(visited_urls)} URLs and took {len(screenshots)} screenshots")
            
            # Process agent's output to extract structured data
            doc_pages = self._parse_agent_site_mapping(extracted_content, visited_urls, base_domain, doc_url)
            
            # If we couldn't extract structured data from the agent output, 
            # fall back to using the visited URLs with default priorities
            if not doc_pages:
                logger.warning("Couldn't parse agent's site mapping, using visited URLs instead")
                doc_pages = []
                
                # Get page titles from browser history
                page_titles = {}
                # Ensure result.history and result.history.history are accessible
                if hasattr(result, 'history') and result.history and hasattr(result.history, 'history') and result.history.history:
                    for history_step in result.history.history: # history_step is AgentHistory
                        if not (history_step.model_output and hasattr(history_step.model_output, 'action') and \
                                history_step.model_output.action and hasattr(history_step, 'result') and \
                                history_step.result and hasattr(history_step, 'state')):
                            continue

                        # Iterate through actions taken in this step and their corresponding results
                        for i, action_model_instance in enumerate(history_step.model_output.action):
                            if i >= len(history_step.result): # Ensure result exists for this action
                                continue

                            action_data = action_model_instance.model_dump(exclude_unset=True)
                            if not action_data:
                                continue
                            
                            action_name = next(iter(action_data.keys()))
                            
                            if action_name == 'get_page_content':
                                action_outcome = history_step.result[i] # This is an ActionResult
                                html_content_for_title = getattr(action_outcome, 'extracted_content', None) # Use a distinct variable name
                                current_url_at_step = getattr(history_step.state, 'url', None)

                                if html_content_for_title and current_url_at_step:
                                    try:
                                        soup = BeautifulSoup(html_content_for_title, 'lxml')
                                        title = soup.title.string if soup.title else None
                                        if title:
                                            page_titles[current_url_at_step] = title.strip()
                                    except Exception as e:
                                        logger.warning(f"Error parsing HTML for title from {current_url_at_step}: {e}")
                
                # Process visited URLs
                for url in visited_urls:
                    # Skip URLs outside the base domain
                    if base_domain not in url:
                        continue
                        
                    # Determine priority based on URL pattern
                    priority = 2  # Default priority
                    url_lower = url.lower()
                    
                    if "api" in url_lower or "reference" in url_lower:
                        priority = 0  # Highest priority
                    elif any(term in url_lower for term in ["guide", "tutorial", "quickstart", "getting-started"]):
                        priority = 1  # Medium priority
                    
                    # Try to get title from page_titles, or generate from URL
                    title = page_titles.get(url)
                    if not title:
                        # Extract a title from the URL path
                        path = urlparse(url).path
                        if path:
                            parts = [p for p in path.split('/') if p]
                            if parts:
                                last_part = parts[-1]
                                # Remove file extensions and replace delimiters with spaces
                                title = re.sub(r'\.(html|md|rst|txt)$', '', last_part)
                                title = title.replace('-', ' ').replace('_', ' ').title()
                    
                    if not title:
                        title = "Documentation Page"
                    
                    # Add to doc_pages with title
                    doc_pages.append({
                        "url": url,
                        "title": title,
                        "priority": priority
                    })
            
            # Sort pages by priority
            doc_pages.sort(key=lambda x: x["priority"])
            
            # Limit to max_pages if needed, but ensure we keep high priority pages
            if len(doc_pages) > max_pages:
                logger.info(f"Limiting to {max_pages} pages from {len(doc_pages)} total found")
                # Keep all high priority (0) pages, then medium (1), then low (2) until we hit max_pages
                high_priority = [p for p in doc_pages if p["priority"] == 0]
                medium_priority = [p for p in doc_pages if p["priority"] == 1]
                low_priority = [p for p in doc_pages if p["priority"] == 2]
                
                # Take all high priority, then as many medium and low as we can fit
                result_pages = high_priority
                remaining = max_pages - len(result_pages)
                
                if remaining > 0:
                    result_pages.extend(medium_priority[:remaining])
                    remaining = max_pages - len(result_pages)
                    
                if remaining > 0:
                    result_pages.extend(low_priority[:remaining])
                    
                doc_pages = result_pages
            
            # Cache the results with TTL
            try:
                with open(cache_file, 'w') as f:
                    json.dump(doc_pages, f)
            except OSError as e:
                logger.warning(f"Failed to cache site map for {doc_url}: {str(e)}")
            
            logger.info(f"Found {len(doc_pages)} documentation pages to process")
            
            # Log the breakdown by priority
            priority_counts = {}
            for page in doc_pages:
                priority = page["priority"]
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            logger.info(f"Priority breakdown: {priority_counts}")
            
            return doc_pages
            
        except Exception as e:
            logger.error(f"Error mapping site {doc_url}: {str(e)}")
            raise ContentExtractionError(f"Failed to map documentation site: {str(e)}") from e
        finally:
            if 'agent' in locals() and hasattr(agent, 'browser_session') and agent.browser_session:
                await agent.browser_session.close()
    
    def _hash_url(self, url: str) -> str:
        """Create a simplified hash of a URL for caching."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:10]
    
    def _parse_agent_site_mapping(self, content: str, visited_urls: List[str], base_domain: str, doc_url: str) -> List[Dict[str, Any]]:
        """
        Parse the agent's output to extract structured site mapping.
        
        Args:
            content: Agent's output content
            visited_urls: List of URLs visited by the agent
            base_domain: Base domain of the documentation site
            doc_url: Original documentation URL
            
        Returns:
            List of page dictionaries with url, title, and priority
        """
        doc_pages = []
        
        try:
            # Try to extract JSON directly if the agent returned it
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed_data = json.loads(json_str)
                    if isinstance(parsed_data, list):
                        for item in parsed_data:
                            if isinstance(item, dict) and "url" in item:
                                priority = item.get("priority", 2)
                                if isinstance(priority, str):
                                    try:
                                        priority = int(priority)
                                    except ValueError:
                                        priority = 2
                                doc_pages.append({
                                    "url": item["url"],
                                    "title": item.get("title", "Documentation Page"),
                                    "priority": priority
                                })
                        return doc_pages
                except json.JSONDecodeError:
                    pass
            
            # Try to extract URLs and priorities using regex patterns
            
            # Pattern 1: Markdown links with priority indicators
            # Example: "- Priority 0: [API Reference](https://example.com/api)"
            pattern1 = r'(?:priority|importance|level)[^\[]*\s*(\d)\s*:?\s*\[([^\]]+)\]\(([^)]+)\)'
            matches1 = re.finditer(pattern1, content.lower())
            
            for match in matches1:
                priority = int(match.group(1))
                title = match.group(2).strip()
                url = match.group(3).strip()
                
                # Verify URL is from the same domain
                if base_domain in url:
                    doc_pages.append({
                        "url": url,
                        "title": title,
                        "priority": priority
                    })
            
            # Pattern 2: URL with priority indicators
            # Example: "Priority 1: https://example.com/guide - User Guide"
            pattern2 = r'(?:priority|importance|level)[^\n\r:]*\s*(\d)\s*:?\s*(https?://[^\s]+)[^\n\r]*(?:-|\s+)([^\n\r]+)'
            matches2 = re.finditer(pattern2, content.lower())
            
            for match in matches2:
                priority = int(match.group(1))
                url = match.group(2).strip()
                title = match.group(3).strip()
                
                # Verify URL is from the same domain
                if base_domain in url:
                    doc_pages.append({
                        "url": url,
                        "title": title,
                        "priority": priority
                    })
            
            # Pattern 3: Simple list with URLs
            # Example: "- https://example.com/api"
            if not doc_pages:
                pattern3 = r'(?:^|\n)\s*(?:-|\*|\d+\.)\s*(https?://[^\s]+)'
                matches3 = re.finditer(pattern3, content)
                
                for match in matches3:
                    url = match.group(1).strip()
                    
                    # Verify URL is from the same domain
                    if base_domain in url:
                        # Determine priority based on URL pattern
                        priority = 2  # Default priority
                        url_lower = url.lower()
                        
                        if "api" in url_lower or "reference" in url_lower:
                            priority = 0  # Highest priority
                        elif any(term in url_lower for term in ["guide", "tutorial", "quickstart", "getting-started"]):
                            priority = 1  # Medium priority
                        
                        # Get a title from the URL if possible
                        title = url.split('/')[-1].replace('-', ' ').replace('_', ' ').title() or "Documentation Page"
                        
                        doc_pages.append({
                            "url": url,
                            "title": title,
                            "priority": priority
                        })
            
            # If we found pages, deduplicate by URL
            if doc_pages:
                # Remove duplicates while preserving order
                seen_urls = set()
                unique_pages = []
                
                for page in doc_pages:
                    if page["url"] not in seen_urls:
                        seen_urls.add(page["url"])
                        unique_pages.append(page)
                
                return unique_pages
        
        except Exception as e:
            logger.warning(f"Error parsing agent site mapping: {str(e)}")
        
        # If parsing failed or no pages found, return empty list
        return []
    
    async def extract_content_from_page(self, url: str, package_name: str) -> str:
        """
        Extract the main content from a documentation page using an AI agent.
        
        Args:
            url: URL of the documentation page
            package_name: Name of the package
            
        Returns:
            HTML content of the main documentation area
        """
        async with self.extraction_semaphore:
            logger.browser_action(f"Extracting content from page for {package_name}", url=url)
            
            # Check cache with TTL validation
            cache_key = f"content_{self._hash_url(url)}"
            cache_file = self.cache_dir / f"{cache_key}.html"
            
            if cache_file.exists():
                try:
                    cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if cache_age < self.extraction_cache_ttl:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cached_content = f.read()
                            if cached_content and len(cached_content) > 100:  # Ensure it's not empty or corrupt
                                logger.info(f"Using cached content for {url}")
                                return cached_content
                    else:
                        logger.info(f"Content cache expired for {url}. Refreshing...")
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning(f"Cache read error for {url}: {str(e)}. Will regenerate.")
            
            # Track token usage for the agent
            agent_start_tokens = tracker.get_total_tokens()
            
            # Track token usage for the agent
            agent_start_tokens = tracker.get_total_tokens()
            
            try:
                # Set up initial actions to visit the page
                initial_actions = [
                    {'open_tab': {'url': url}}
                ]

                # Create an agent, letting browser-use handle browser lifecycle
                agent = Agent(
                    task=f"""
                    Extract the main documentation content from this page about the Python package '{package_name}'.
                    
                    Focus on:
                    1. Main documentation text
                    2. Code examples and syntax
                    3. API details and signatures
                    4. Important explanations
                    
                    IGNORE:
                    - Navigation bars
                    - Headers and footers
                    - Sidebars
                    - Advertisement
                    - Comment sections
                    
                    Return the content in a clean, well-formatted way that preserves the structure and all technical details.
                    Make sure to include code blocks, tables, lists, and other elements exactly as they appear.
                    """,
                    llm=self.llm,
                    use_vision=self.use_vision,
                    initial_actions=initial_actions # Pass initial_actions to constructor
                )
                
                # Add this URL to processed set to avoid duplicate processing
                self.processed_urls.add(url)
                
                # Run the agent
                await agent.browser_session.start() 
                result = await agent.run() # Call run() with no arguments
                
                # Calculate token usage
                agent_end_tokens = tracker.get_total_tokens()
                tokens_used = agent_end_tokens - agent_start_tokens
                logger.token_usage(
                    prompt_tokens=tokens_used * 0.7,  # Approximate split between prompt and completion
                    completion_tokens=tokens_used * 0.3,
                    model="gpt-4o"
                )
                
                # Get the HTML content from the agent's screenshots or response
                # First, check if we can get the HTML content directly from the page
                extracted_content = result.final_result()
                screenshots = result.screenshots()
                
                logger.info(f"Agent extracted content from {url} ({len(extracted_content)} chars, {len(screenshots)} screenshots)")
                
                # If the agent didn't return substantial content, try to get it from the page HTML
                if len(extracted_content) < 100:
                    # Get the page HTML from the browser actions
                    html_content = None
                    # Ensure result.history and result.history.history are accessible
                    if hasattr(result, 'history') and result.history and hasattr(result.history, 'history') and result.history.history:
                        for history_step in result.history.history: # history_step is AgentHistory
                            if not (history_step.model_output and hasattr(history_step.model_output, 'action') and \
                                    history_step.model_output.action and hasattr(history_step, 'result') and \
                                    history_step.result): # No need to check for state here
                                continue

                            # Iterate through actions taken in this step and their corresponding results
                            for i, action_model_instance in enumerate(history_step.model_output.action):
                                if i >= len(history_step.result): # Ensure result exists for this action
                                    continue

                                action_data = action_model_instance.model_dump(exclude_unset=True)
                                if not action_data:
                                    continue
                                
                                action_name = next(iter(action_data.keys()))
                                
                                if action_name == 'get_page_content':
                                    action_outcome = history_step.result[i] # This is an ActionResult
                                    current_html_content = getattr(action_outcome, 'extracted_content', None)
                                    if current_html_content: # Check if content was extracted
                                        html_content = current_html_content # Assign to the outer scope html_content
                                        break # Found the first get_page_content with actual content
                            if html_content: # If we found content, break outer loop too
                                break
                    
                    if html_content:
                        # Parse HTML with BeautifulSoup
                        soup = BeautifulSoup(html_content, 'lxml')
                        
                        # Remove navigation, headers, footers, etc.
                        selectors_to_remove = [
                            'nav', 'header', 'footer', '.sidebar', '.navigation', '.menu', 
                            '#sidebar', '#menu', '.toc', '#toc', '.search', '#search',
                            '.breadcrumbs', '.breadcrumb', '.admonition', '.warning', 
                            '.note', '.header-bar', '.installation',
                            '.headerlink', '.editlink', '.social-links',
                            'script', 'style', 'iframe', 'noscript'
                        ]
                        
                        for selector in selectors_to_remove:
                            for element in soup.select(selector):
                                element.extract()
                        
                        # Try to find the main content using common selectors
                        main_content_selectors = [
                            'main', 'article', '.content', '#content', '.documentation', '#documentation',
                            '.document', '#document', '.section', '.body', '.page-content', 
                            '.main-content', '#main-content', '.markdown-body', '.rst-content',
                            '.sphinxsidebar', '.bodywrapper', '.documentwrapper'
                        ]
                        
                        main_content = None
                        for selector in main_content_selectors:
                            content = soup.select_one(selector)
                            if content and len(str(content)) > 500:  # Must have meaningful content
                                main_content = content
                                break
                        
                        # If no main content identified with selectors, try heuristic approaches
                        if not main_content or len(str(main_content)) < 500:
                            # Try to find the largest content block
                            paragraphs = soup.find_all('p')
                            if paragraphs:
                                # Count paragraphs per parent to find the main content area
                                parent_counts = {}
                                for p in paragraphs:
                                    if p.parent:
                                        parent_id = id(p.parent)
                                        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
                                
                                # Find the parent with the most paragraphs
                                if parent_counts:
                                    main_parent_id = max(parent_counts, key=parent_counts.get)
                                    for p in paragraphs:
                                        if p.parent and id(p.parent) == main_parent_id:
                                            main_content = p.parent
                                            break
                        
                        # If still no main content, use the body minus the removed elements
                        if not main_content or len(str(main_content)) < 500:
                            main_content = soup.body if soup.body is not None else soup
                            
                            # If even the body doesn't exist (extremely rare), use the whole soup
                            if not main_content or len(str(main_content)) < 100:
                                main_content = soup
                        
                        # Convert to string
                        content_html = str(main_content)
                        
                        # Use a regex to remove HTML comments
                        content_html = re.sub(r'<!--.*?-->', '', content_html, flags=re.DOTALL)
                        
                        # Improve code blocks formatting
                        content_html = self._improve_code_blocks(content_html)
                        
                        # Cache the content
                        try:
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                f.write(content_html)
                        except OSError as e:
                            logger.warning(f"Failed to cache content for {url}: {str(e)}")
                        
                        return content_html
                
                # Convert the agent's extracted content to HTML
                if extracted_content:
                    # Check if the content is already HTML
                    if re.search(r'<\w+>', extracted_content) and re.search(r'</\w+>', extracted_content):
                        html_content = extracted_content
                    else:
                        # Basic markdown to HTML conversion for the agent's output
                        html_content = f"<div class='agent-extracted-content'>{extracted_content}</div>"
                    
                    # Cache the content
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                    except OSError as e:
                        logger.warning(f"Failed to cache content for {url}: {str(e)}")
                    
                    return html_content
                
                # If all else fails, return a placeholder
                return f"<div class='extraction-failed'>Content extraction failed for {url}</div>"
                
            except Exception as e:
                logger.error(f"Error extracting content from {url} for {package_name}: {str(e)}")
                return f"<div class='extraction-error'>Error extracting content: {str(e)}</div>"
            finally:
                if 'agent' in locals() and hasattr(agent, 'browser_session') and agent.browser_session:
                    await agent.browser_session.close()
    
    def _improve_code_blocks(self, html_content: str) -> str:
        """Improve the formatting of code blocks in HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Process <pre> and <code> elements
        for pre in soup.find_all('pre'):
            # Add class for styling if missing
            if 'class' not in pre.attrs:
                pre['class'] = pre.get('class', []) + ['code-block']
                
            # If pre contains a code element, make sure it's properly formatted
            code = pre.find('code')
            if code:
                # Preserve language class if present
                lang_class = None
                for cls in code.get('class', []):
                    if cls.startswith('language-') or cls.startswith('lang-'):
                        lang_class = cls
                        break
                
                if lang_class and 'class' in pre.attrs:
                    pre['class'] = pre.get('class', []) + [lang_class]
        
        return str(soup)
    
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
            # If the content appears to be already in markdown format
            if html.startswith("<div class='agent-extracted-content'>") and "```" in html:
                # It's likely the agent already returned markdown, extract it
                markdown = re.sub(r'<div class=\'agent-extracted-content\'>(.*?)</div>', r'\1', html, flags=re.DOTALL)
            else:
                # Use markitdown to convert HTML to Markdown
                markdown = markitdown.markitdown(html) # Corrected call
                
                # Clean up the markdown
                markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excess newlines
                
                # Fix common markitdown conversion issues
                markdown = re.sub(r'\\\[', '[', markdown)  # Fix escaped brackets
                markdown = re.sub(r'\\\]', ']', markdown)  # Fix escaped brackets
                markdown = re.sub(r'\\_', '_', markdown)    # Fix escaped underscores
                
                # Improve code block formatting
                markdown = self._improve_markdown_code_blocks(markdown)
            
            # Add source URL reference
            if url:
                markdown = f"Source: {url}\n\n{markdown}"
            
            return markdown
            
        except Exception as e:
            logger.warning(f"Error converting HTML to Markdown: {str(e)}")
            # Return a simplified version if conversion fails
            return f"Source: {url}\n\n{html}"
    
    def _improve_markdown_code_blocks(self, markdown: str) -> str:
        """Improve the formatting of code blocks in Markdown content."""
        # Fix triple backtick code blocks missing language identifier
        markdown = re.sub(r'```\s*\n', '```python\n', markdown)
        
        # Fix code blocks where indentation was used instead of backticks
        lines = markdown.split('\n')
        in_indented_block = False
        new_lines = []
        indented_block = []
        
        for i, line in enumerate(lines):
            # Check if this is an indented line that should be part of a code block
            is_indented = line.startswith('    ') and not line.startswith('     #')
            prev_line_empty = i > 0 and not lines[i-1].strip()
            next_line_empty = i < len(lines) - 1 and not lines[i+1].strip()
            
            if is_indented and not in_indented_block and prev_line_empty:
                # Start a new code block
                in_indented_block = True
                new_lines.append('```python')
                indented_block = [line[4:]]  # Remove indentation
            elif is_indented and in_indented_block:
                # Continue the block
                indented_block.append(line[4:])
            elif not is_indented and in_indented_block:
                # End the block
                if not line.strip() or next_line_empty:
                    new_lines.extend(indented_block)
                    new_lines.append('```')
                    in_indented_block = False
                    indented_block = []
                    if line.strip():  # Add the current line only if it's not empty
                        new_lines.append(line)
                else:
                    # This wasn't a code block after all, just indented text
                    in_indented_block = False
                    for indented_line in indented_block:
                        new_lines.append('    ' + indented_line)
                    indented_block = []
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Handle case where document ends with indented block
        if in_indented_block:
            new_lines.extend(indented_block)
            new_lines.append('```')
        
        return '\n'.join(new_lines)
    
    async def process_package_documentation(self, package: Package, force_extract: bool = False) -> Optional[str]:
        """
        Process the documentation for a package.
        
        Args:
            package: Package object
            force_extract: Force extraction even if documentation already exists
            
        Returns:
            Path to the generated combined markdown file
        """
        logger.doc_extraction_start(package.name)
        
        # Check if documentation already exists and force_extract is False
        if not force_extract and package.original_doc_path and Path(package.original_doc_path).exists():
            logger.info(f"Documentation already exists for {package.name} at {package.original_doc_path}")
            return package.original_doc_path
        
        # Find documentation URL if not already set
        doc_url = package.docs_url
        if not doc_url:
            doc_url = await self.find_documentation_url(package.name)
            if not doc_url:
                logger.warning(f"Could not find documentation for {package.name}")
                return None
            
            # Store the docs URL in the package
            package.docs_url = doc_url
        
        # Map the documentation site
        try:
            doc_pages = await self.map_documentation_site(doc_url)
            
            if not doc_pages:
                logger.warning(f"No documentation pages found for {package.name}")
                return None
        except Exception as e:
            logger.error(f"Error mapping documentation site for {package.name}: {str(e)}")
            return None
        
        # Reset processed URLs for this package
        self.processed_urls = set()
        
        # Extract content from each page and convert to markdown
        combined_markdown = f"# {package.name} Documentation\n\n"
        combined_markdown += f"*This is the original documentation for {package.name}, combined from multiple pages.*\n\n"
        combined_markdown += f"*Main documentation source: {doc_url}*\n\n"
        combined_markdown += f"*Extraction date: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        # Process pages concurrently with controlled concurrency and error handling
        tasks = []
        errors = []
        
        for page_info in doc_pages:
            task = self._process_page_safe(page_info, package.name, errors)
            tasks.append(task)
        
        # Wait for all tasks with a timeout
        try:
            results = await asyncio.gather(*tasks)
            
            # Filter out None results (errors) and add to combined markdown
            valid_results = [r for r in results if r]
            for markdown_section in valid_results:
                combined_markdown += markdown_section
            
            # Report errors if any
            if errors:
                error_count = len(errors)
                logger.warning(f"Encountered {error_count} errors while processing pages for {package.name}")
                
                if error_count <= 5:
                    for error in errors:
                        logger.warning(f"  - {error}")
                
                # Add error information to the markdown
                if valid_results:
                    combined_markdown += "\n\n## Processing Errors\n\n"
                    combined_markdown += f"*Note: {error_count} pages failed to process properly.*\n\n"
        except asyncio.TimeoutError:
            logger.error(f"Timed out while processing pages for {package.name}")
            if not combined_markdown:
                return None
            
            combined_markdown += "\n\n## Processing Timeout\n\n"
            combined_markdown += "*Note: Processing timed out. Only partial documentation is available.*\n\n"
        
        # Save the combined markdown
        output_filename = f"{package.name}__combined_original_documentation__as_of_{datetime.now().strftime('%m_%d_%Y')}.md"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_markdown)
        
        logger.doc_extraction_complete(package.name, str(output_path), len(doc_pages))
        return str(output_path)
    
    async def _process_page_safe(self, page_info: Dict[str, Any], package_name: str, errors: List[str]) -> Optional[str]:
        """Process a single page with error handling."""
        try:
            return await self._process_page(page_info, package_name)
        except Exception as e:
            error_msg = f"Error processing page {page_info.get('url', 'unknown URL')}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            return None
    
    async def _process_page(self, page_info: Dict[str, Any], package_name: str) -> Optional[str]:
        """Process a single documentation page."""
        page_url = page_info["url"]
        title = page_info.get("title", page_url.split('/')[-1].replace('-', ' ').replace('_', ' ').title())
        if not title:
            title = "Documentation Page"
        
        # Skip if we've already processed this URL
        if page_url in self.processed_urls:
            logger.info(f"Skipping already processed URL: {page_url}")
            return ""
        
        # Add to processed URLs
        self.processed_urls.add(page_url)
        
        logger.info(f"Processing page: {title} ({page_url})")
        
        try:
            # Extract content with retry logic
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    html_content = await self.extract_content_from_page(page_url, package_name)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error extracting content from {page_url} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise ContentExtractionError(f"Failed to extract content after {max_retries} attempts: {str(e)}") from e
            
            if html_content:
                markdown_content = await self.convert_html_to_markdown(html_content, page_url)
                logger.info(f"Converted page {title} to markdown ({len(markdown_content)} chars)")
                return f"## {title}\n\n{markdown_content}\n\n---\n\n"
            else:
                raise ContentExtractionError("Extracted content was empty")
        except Exception as e:
            logger.warning(f"Failed to extract content from {page_url}: {str(e)}")
            # Return minimal information about the failed page
            return f"## {title}\n\n*Note: Content extraction failed for this page ({page_url})*\n\n---\n\n"
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
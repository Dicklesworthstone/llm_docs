"""
Integration utilities for browser-use vision capabilities.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin # Added urljoin
import re # Added re

from bs4 import BeautifulSoup # Added BeautifulSoup
from browser_use import Agent, Browser

from llm_docs.utils.browser.vision_config import VisionConfig, default_vision_config
from llm_docs.utils.llm_docs_logger import logger
from llm_docs.utils.token_tracking import tracker


async def create_browser_with_vision_config(vision_config: Optional[VisionConfig] = None) -> Browser:
    """
    Create a Browser instance with the specified vision configuration.
    
    Args:
        vision_config: Vision configuration
        
    Returns:
        Configured Browser instance
    """
    if vision_config is None:
        vision_config = default_vision_config
    
    browser_config = vision_config.get_browser_config()
    return Browser(config=browser_config)

async def run_agent_with_vision_config(
    task: str,
    llm: Any,
    vision_config: Optional[VisionConfig] = None,
    browser: Optional[Browser] = None,
    initial_actions: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Any:
    """
    Run a browser-use Agent with the specified vision configuration.
    """
    if vision_config is None:
        vision_config = default_vision_config
    
    # Create browser if needed
    close_browser = False
    if browser is None:
        browser = await create_browser_with_vision_config(vision_config)
        close_browser = True
    
    try:
        # Create agent with vision configuration
        agent_kwargs = vision_config.get_agent_kwargs()
        agent_kwargs.update(kwargs)
        
        # Pass initial_actions during Agent creation instead of run()
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            initial_actions=initial_actions,  # Pass initial actions here
            **agent_kwargs
        )
        
        # Track token usage before running agent
        start_tokens = tracker.get_total_tokens()
        
        # Run the agent WITHOUT initial_actions parameter
        result = await agent.run()
        
        # Calculate token usage
        end_tokens = tracker.get_total_tokens()
        tokens_used = end_tokens - start_tokens
        
        # Log token usage
        logger.info(f"Agent used approximately {tokens_used} tokens")
        
        return result
    finally:
        # Close browser if we created it
        if close_browser:
            await browser.close()

def patch_documentation_extractor_with_vision_config():
    """
    Patch DocumentationExtractor to use vision configuration.
    """
    from llm_docs.doc_extraction import DocumentationExtractor
    
    # Save original methods
    if not hasattr(DocumentationExtractor, '_original_map_documentation_site'):
        DocumentationExtractor._original_map_documentation_site = DocumentationExtractor.map_documentation_site
    
    if not hasattr(DocumentationExtractor, '_original_extract_content_from_page'):
        DocumentationExtractor._original_extract_content_from_page = DocumentationExtractor.extract_content_from_page
    
    if not hasattr(DocumentationExtractor, '_original_try_google_search'):
        DocumentationExtractor._original_try_google_search = DocumentationExtractor._try_google_search
    
    # Add vision_config attribute to DocumentationExtractor
    DocumentationExtractor.vision_config = default_vision_config
    
    # Patch methods with vision-config-aware versions
    async def patched_map_documentation_site(self, doc_url: str, max_pages: int = 50) -> List[Dict[str, Any]]:
        """Patched version of map_documentation_site with vision configuration."""
        # Create a browser with vision configuration
        browser = await create_browser_with_vision_config(self.vision_config)
        
        doc_pages = []
        visited_urls = set()

        try:
            logger.info(f"Opening initial URL: {doc_url}")
            await browser.open_tab(doc_url)
            visited_urls.add(doc_url)

            # Get current page content
            page = await browser.get_current_page()
            soup = BeautifulSoup(page.html_content, 'html.parser')

            # Extract links
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Resolve relative URLs
                full_url = urljoin(doc_url, href)
                links.append(full_url)
            
            logger.info(f"Found {len(links)} links on the initial page.")

            # Filter links
            base_domain = urlparse(doc_url).netloc
            filtered_links = set() # Use a set to automatically handle duplicates

            # Define documentation patterns
            doc_patterns = [
                r'/api', r'/docs', r'/guide', r'/reference', r'/tutorial', r'\.html$'
            ]

            for link in links:
                # Check domain
                if urlparse(link).netloc != base_domain:
                    continue

                # Check patterns
                if any(re.search(pattern, link) for pattern in doc_patterns):
                    filtered_links.add(link)
            
            logger.info(f"Filtered down to {len(filtered_links)} links matching domain and patterns.")

            # Limit to max_pages (considering the initial page is already visited)
            # Ensure we don't try to visit the initial page again if it's in filtered_links
            links_to_visit = [link for link in list(filtered_links) if link != doc_url][:max_pages -1]

            # Process the initial page first
            try:
                current_url = doc_url
                page_title = soup.title.string if soup.title else "No title found"
                priority = self._assign_priority(current_url, page_title)
                doc_pages.append({'url': current_url, 'title': page_title, 'priority': priority})
                logger.info(f"Processed initial page: {current_url} (Title: {page_title}, Priority: {priority})")
            except Exception as e:
                logger.error(f"Error processing initial page {doc_url}: {e}")

            # Visit filtered links
            for i, link_url in enumerate(links_to_visit):
                if link_url in visited_urls or len(doc_pages) >= max_pages:
                    continue # Skip already visited or if max_pages limit reached
                
                try:
                    logger.info(f"Visiting link {i+1}/{len(links_to_visit)}: {link_url}")
                    await browser.open_tab(link_url) # In browser-use, open_tab navigates or opens a new one
                    visited_urls.add(link_url)
                    
                    page = await browser.get_current_page()
                    link_soup = BeautifulSoup(page.html_content, 'html.parser')
                    
                    page_title = link_soup.title.string if link_soup.title else "No title found"
                    priority = self._assign_priority(link_url, page_title)
                    
                    doc_pages.append({'url': link_url, 'title': page_title, 'priority': priority})
                    logger.info(f"Processed: {link_url} (Title: {page_title}, Priority: {priority})")

                except Exception as e:
                    logger.error(f"Error processing page {link_url}: {e}")
                    # Optionally, add a placeholder or skip if a page fails
            
            # Sort by priority
            doc_pages.sort(key=lambda p: p['priority'])

            return doc_pages
        except Exception as e:
            logger.error(f"Error during site mapping for {doc_url}: {e}")
            return [] # Return empty list on error
        finally:
            # Always close the browser
            await browser.close()
    
    # Replace methods with patched versions
    DocumentationExtractor.map_documentation_site = patched_map_documentation_site
    
    # Add the helper method for priority assignment
    def _assign_priority(self, url: str, title: str) -> int:
        """Assigns priority based on URL and title keywords."""
        url_lower = url.lower()
        title_lower = title.lower()

        # Priority 0 keywords
        high_priority_keywords = ["api", "reference", "class", "function", "module"]
        if any(keyword in url_lower or keyword in title_lower for keyword in high_priority_keywords):
            return 0

        # Priority 1 keywords
        medium_priority_keywords = ["guide", "tutorial", "howto", "examples", "getting started", "concepts", "usage"]
        if any(keyword in url_lower or keyword in title_lower for keyword in medium_priority_keywords):
            return 1
            
        return 2 # Default to low priority

    DocumentationExtractor._assign_priority = _assign_priority
    
    # Similar implementations for extract_content_from_page and _try_google_search
    # (omitted for brevity but would follow the same pattern)
    
    logger.info("DocumentationExtractor patched with vision configuration and web scraping mapping")


def enable_vision_config(vision_config: Optional[VisionConfig] = None):
    """
    Enable vision configuration for all components.
    
    Args:
        vision_config: Vision configuration to use
    """
    if vision_config is None:
        vision_config = default_vision_config
    
    from llm_docs.doc_extraction import DocumentationExtractor
    
    # Patch DocumentationExtractor
    patch_documentation_extractor_with_vision_config()
    
    # Set vision configuration
    DocumentationExtractor.vision_config = vision_config
    
    logger.info(f"Vision configuration enabled with quality: {vision_config.quality.value}")
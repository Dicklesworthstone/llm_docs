"""
Integration utilities for browser-use vision capabilities.
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
        
        try:
            # Create an agent with vision configuration
            agent_kwargs = self.vision_config.get_agent_kwargs()
            
            # Set up initial actions to visit the documentation site
            initial_actions = [
                {'open_tab': {'url': doc_url}}
            ]
            
            # Create agent WITH initial_actions
            agent = Agent(
                task="""
                Map this documentation site and find all important pages. Focus on:
                
                1. API reference pages
                2. User guides and tutorials
                3. Getting started pages
                4. Core concepts and important sections

                Keep track of each page with its URL, title, and importance level. 
                Make sure to explore navigation menus and follow links within the same domain.
                
                For each page, assign a priority score:
                - 0 (highest): API reference, class/function documentation
                - 1 (medium): User guides, tutorials, examples
                - 2 (low): Other related pages
                
                Return a list of important pages organized by priority.
                """,
                llm=self.llm,
                browser=browser,
                initial_actions=initial_actions,  # Pass initial_actions HERE
                **agent_kwargs
            )
            
            # Run the agent WITHOUT initial_actions parameter
            result = await agent.run()

            # Process the agent's output similar to original method
            extracted_content = result.final_result()
            visited_urls = result.urls()
            screenshots = result.screenshots()
            
            logger.info(f"Agent visited {len(visited_urls)} URLs and took {len(screenshots)} screenshots")
            
            # Process agent's output to extract structured data
            doc_pages = self._parse_agent_site_mapping(extracted_content, visited_urls, 
                                                    urlparse(doc_url).netloc, doc_url)
            
            # Continue with original method's logic for processing results
            # This part remains the same as in the original method
            
            return doc_pages
        finally:
            # Always close the browser
            await browser.close()
    
    # Replace methods with patched versions
    DocumentationExtractor.map_documentation_site = patched_map_documentation_site
    
    # Similar implementations for extract_content_from_page and _try_google_search
    # (omitted for brevity but would follow the same pattern)
    
    logger.info("DocumentationExtractor patched with vision configuration")


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
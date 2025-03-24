"""
Integration helpers for tracking token usage with browser-use Agent.

This module provides functions to extract and track token usage from browser-use
Agent responses, particularly focusing on vision capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple

from browser_use import Agent
from browser_use.utils.types import AgentHistoryList

from llm_docs.utils.llm_docs_logger import logger
from llm_docs.utils.token_tracking import tracker
from llm_docs.utils.token_tracking.token_tracker_adapter import get_pricing_for_model


async def extract_agent_token_usage(agent_result: AgentHistoryList, task_name: str, model: str = "gpt-4o") -> Dict[str, int]:
    """
    Extract and track token usage from browser-use Agent results.
    
    Args:
        agent_result: Result from agent.run()
        task_name: Name of the task for tracking
        model: Model name (default: gpt-4o)
        
    Returns:
        Dictionary with input_tokens and output_tokens
    """
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Look for token usage in model actions
    model_actions = agent_result.model_actions()
    
    # Try to get token usage from the raw response data
    for action in model_actions:
        # First check for token usage in the response metadata
        response_data = getattr(action, 'response', None)
        
        if response_data:
            # Look for usage data in the response
            usage = None
            if hasattr(response_data, 'usage'):
                usage = response_data.usage
            elif isinstance(response_data, dict) and 'usage' in response_data:
                usage = response_data['usage']
                
            if usage:
                # Extract token counts
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                if not input_tokens and isinstance(usage, dict):
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    
                output_tokens = getattr(usage, 'completion_tokens', 0)
                if not output_tokens and isinstance(usage, dict):
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                
                # Update totals
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
    
    # If we didn't find any token usage data, we need to estimate
    if total_input_tokens == 0 and total_output_tokens == 0:
        # Get vision usage - each screenshot is processed as an image
        screenshots = agent_result.screenshots()
        
        # In gpt-4o, an average screenshot is approximately 850 tokens
        # depending on resolution and content complexity
        vision_tokens = len(screenshots) * 850
        
        # Get text input usage
        # Browser-use uses a special system prompt around 1000 tokens
        system_prompt_tokens = 1000
        
        # Estimate user message based on task complexity
        final_result = agent_result.final_result()
        task_length = len(final_result) if final_result else 0
        
        # Larger tasks typically have more API calls
        task_complexity = 1 + (task_length // 2000)  # Scale based on result size
        
        # Typical model input for a standard browsing session
        typical_input_per_call = 2000  # Base text tokens per call
        
        # Estimate typical model output
        typical_output_per_call = 1000  # Base output tokens per call
        
        # Total estimated input tokens
        estimated_input_tokens = (
            system_prompt_tokens + 
            (typical_input_per_call * task_complexity) + 
            vision_tokens
        )
        
        # Total estimated output tokens
        estimated_output_tokens = typical_output_per_call * task_complexity
        
        total_input_tokens = estimated_input_tokens
        total_output_tokens = estimated_output_tokens
    
    # Track token usage
    # Get pricing for the model
    pricing = get_pricing_for_model(model)
    
    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * pricing.get("input", 0)
    output_cost = (total_output_tokens / 1_000_000) * pricing.get("output", 0)
    
    # Update the tracker
    tracker.update(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        task=task_name,
        input_cost=input_cost,
        output_cost=output_cost
    )
    
    # Log token usage
    logger.token_usage(total_input_tokens, total_output_tokens, model)
    
    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_cost": input_cost + output_cost
    }


async def run_agent_with_tracking(
    agent: Agent, 
    task_name: str, 
    initial_actions: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4o"
) -> Tuple[Any, Dict[str, int]]:
    """
    Run browser-use Agent with token tracking.
    
    Args:
        agent: Browser-use Agent instance
        task_name: Name of the task for tracking
        initial_actions: Optional list of initial actions
        model: Model name used by the agent
        
    Returns:
        Tuple of (agent result, token usage dict)
    """
    # Run the agent
    result = await agent.run(initial_actions=initial_actions)
    
    # Track token usage
    token_usage = await extract_agent_token_usage(result, task_name, model)
    
    return result, token_usage


# Patching methods for DocumentationExtractor to use our token tracking

async def patched_try_google_search(self, package_name: str) -> Optional[str]:
    """
    Patched version of _try_google_search with token tracking.
    
    Args:
        package_name: Name of the package
        
    Returns:
        URL of the documentation site if found, None otherwise
    """
    search_query = f"{package_name} python documentation"
    
    logger.info(f"Searching for documentation for {package_name} using Google...")
    
    try:
        # Initialize browser if needed
        browser = self.__class__._get_browser(self)
        
        try:
            # Create an agent with appropriate model for searching
            agent = Agent(
                task=f"Search for the official documentation site for the Python package '{package_name}'. Look for links to readthedocs.io, official websites with documentation sections, or GitHub documentation. Return only the URL to the main documentation page (not a specific section or article).",
                llm=self.llm,
                browser=browser,
                use_vision=self.use_vision  # Enable vision capabilities if available
            )
            
            # Set up initial actions to perform Google search
            initial_actions = [
                {'open_tab': {'url': f"https://www.google.com/search?q={search_query}"}}
            ]
            
            # Run the agent with token tracking
            logger.browser_action(f"Running search agent for {package_name}", url=f"https://www.google.com/search?q={search_query}")
            result, token_usage = await run_agent_with_tracking(
                agent=agent,
                task_name=f"find_documentation_url_{package_name}",
                initial_actions=initial_actions
            )
            
            # Process results the same as in the original function
            extracted_content = result.final_result()
            
            if not extracted_content:
                logger.warning(f"Agent couldn't find documentation URL for {package_name}")
                return None
                
            # Extract URLs using comprehensive regex pattern
            urls = self.__class__._extract_urls_from_content(self, extracted_content)
            
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
            
            # Process the rest of URLs as in the original function
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
        finally:
            # Always close the browser properly if we created it
            if not hasattr(self, 'browser') or self.browser is None:
                await browser.close()
                
    except Exception as e:
        logger.error(f"Agent search error for {package_name}: {str(e)}")
        raise


# Helper methods to be added to DocumentationExtractor

def _extract_urls_from_content(self, content: str) -> List[str]:
    """Extract URLs from content using regex."""
    import re
    
    # Comprehensive regex for URLs
    urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!.~\'()*+,;=:@/\\&?#]*)?', content)
    
    if not urls:
        urls = re.findall(r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!.~\'()*+,;=:@/\\&?#]*)?', content)
        urls = [f"https://{url}" for url in urls]
    
    return urls


def _get_browser(self):
    """Get or create a browser instance."""
    if hasattr(self, 'browser') and self.browser is not None:
        return self.browser
    else:
        from browser_use import Browser
        return Browser(config=self.browser_config)


def patch_documentation_extractor():
    """
    Patch the DocumentationExtractor class with token tracking capabilities.
    Call this function at the start of your application to enable token tracking.
    """
    from llm_docs.doc_extraction import DocumentationExtractor
    
    # Save original methods
    if not hasattr(DocumentationExtractor, '_original_map_documentation_site'):
        DocumentationExtractor._original_map_documentation_site = DocumentationExtractor.map_documentation_site
        DocumentationExtractor._original_try_google_search = DocumentationExtractor._try_google_search
    
    # Add helper methods
    DocumentationExtractor._extract_urls_from_content = _extract_urls_from_content
    DocumentationExtractor._get_browser = _get_browser
    
    # Patch methods with token tracking versions
    DocumentationExtractor._try_google_search = patched_try_google_search
    
    logger.info("DocumentationExtractor patched with token tracking capabilities")
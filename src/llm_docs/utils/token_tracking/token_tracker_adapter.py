#!/usr/bin/env python
"""
Adapter for token usage tracking in llm-docs.
Repurposes the existing TokenUsageTracker to work with aisuite.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

# Import the existing TokenUsageTracker
from llm_docs.utils.token_tracking.token_usage_tracker import Config, TokenUsageTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("llm_docs.token_tracker")

# Create a global tracker instance
tracker = TokenUsageTracker()

class LLMProvider(Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    OTHER = "other"

def get_provider_from_model(model_string: str) -> LLMProvider:
    """Determine the provider from the model string."""
    model_lower = model_string.lower()
    
    if "gpt" in model_lower or "openai" in model_lower or "davinci" in model_lower:
        return LLMProvider.OPENAI
    elif "claude" in model_lower or "anthropic" in model_lower:
        return LLMProvider.ANTHROPIC
    elif "gemini" in model_lower or "google" in model_lower:
        return LLMProvider.GOOGLE
    elif "mistral" in model_lower:
        return LLMProvider.MISTRAL
    elif "ollama" in model_lower:
        return LLMProvider.OLLAMA
    else:
        return LLMProvider.OTHER

def update_aisuite_response_tracking(
    response: Any, 
    task: str, 
    prompt_tokens: Optional[int] = None, 
    model: Optional[str] = None
) -> None:
    """
    Track token usage from an aisuite response.
    
    Args:
        response: The response from aisuite
        task: Description of the task being performed
        prompt_tokens: Optional manual count of prompt tokens
        model: Optional model string if not available in response
    """
    # Set default API provider based on model if available
    if model:
        provider = get_provider_from_model(model)
        if provider == LLMProvider.ANTHROPIC:
            Config.API_PROVIDER = "CLAUDE"
        elif provider == LLMProvider.OPENAI:
            Config.API_PROVIDER = "OPENAI"
        elif provider == LLMProvider.GOOGLE:
            Config.API_PROVIDER = "GEMINI"
        else:
            Config.API_PROVIDER = "OPENAI"  # Default fallback
    
    # Try to extract token usage from response
    input_tokens = 0
    output_tokens = 0
    
    try:
        # Check if response has a usage attribute (OpenAI style)
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                input_tokens = usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                output_tokens = usage.completion_tokens
        
        # Check for Anthropic style usage
        elif hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens'):
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
        
        # Check for usage as dictionary
        elif isinstance(getattr(response, 'usage', None), dict):
            usage_dict = response.usage
            input_tokens = usage_dict.get('prompt_tokens', 0) or usage_dict.get('input_tokens', 0)
            output_tokens = usage_dict.get('completion_tokens', 0) or usage_dict.get('output_tokens', 0)
        
        # If we can't find token counts in the response, use provided prompt_tokens and estimate output
        elif prompt_tokens is not None:
            input_tokens = prompt_tokens
            # Estimate output tokens based on response content
            content = ""
            if hasattr(response, 'content'):
                content = response.content
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    content = response.choices[0].message.content
                elif hasattr(response.choices[0], 'text'):
                    content = response.choices[0].text
            
            # Estimate at 4 chars per token for English text
            output_tokens = len(content) // 4 if content else 0
        
        # If all else fails, log a warning
        if input_tokens == 0 and output_tokens == 0:
            logger.warning(f"Could not extract token usage from response for task '{task}'")
            return
            
        # Update the tracker
        tracker.update(input_tokens, output_tokens, task)
        logger.debug(f"Tracked usage for {task}: {input_tokens} input, {output_tokens} output tokens")
        
    except Exception as e:
        logger.warning(f"Error tracking token usage: {str(e)}")

def print_usage_report():
    """Print the usage report."""
    tracker.print_usage_report()

def get_usage_summary() -> Dict[str, Any]:
    """Get the usage summary."""
    return tracker.get_enhanced_summary()
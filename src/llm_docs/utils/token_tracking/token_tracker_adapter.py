"""
Adapter for token usage tracking in llm-docs.
Updated with accurate pricing models and direct extraction from API responses.
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

# Updated pricing based on the latest pricing information (per 1M tokens)
PRICING = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00, "cached_input": 2.50},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "output": 2.40, "cached_input": 0.30},
    "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-search-preview": {"input": 2.50, "output": 10.00},
    "gpt-4.5-preview": {"input": 75.00, "output": 150.00, "cached_input": 37.50},
    "computer-use-preview": {"input": 3.00, "output": 12.00},
    "o1": {"input": 15.00, "output": 60.00, "cached_input": 7.50},
    "o1-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    
    # Anthropic models
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    
    # Google models
    "gemini-1.5-pro": {"input": 7.0, "output": 21.0},
    "gemini-1.0-pro": {"input": 0.125, "output": 0.375},
    
    # Mistral models
    "mistral-large": {"input": 2.0, "output": 6.0},
    "mistral-medium": {"input": 0.27, "output": 0.81},
    "mistral-small": {"input": 0.2, "output": 0.6},
}

def get_provider_from_model(model_string: str) -> LLMProvider:
    """Determine the provider from the model string."""
    model_lower = model_string.lower()
    
    if "gpt" in model_lower or "openai" in model_lower or "davinci" in model_lower or "o1" in model_lower or "o3" in model_lower:
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

def get_base_model_name(model_string: str) -> str:
    """Extract the base model name from a full model string."""
    # Handle aisuite format (provider:model)
    if ":" in model_string:
        model_string = model_string.split(":")[-1]
    
    # Handle versioned models
    model_parts = model_string.split('-')
    if len(model_parts) > 3 and model_parts[0] in ['gpt', 'claude']:
        # For models like gpt-4o-mini-search-preview-2025-03-11
        # Extract the base pattern without the date
        date_pattern = [part for part in model_parts if part.isdigit() or (len(part) == 2 and part.isdigit())]
        if date_pattern:
            date_index = model_parts.index(date_pattern[0])
            return "-".join(model_parts[:date_index])
    
    # For simpler model names like gpt-4o, claude-3-opus, etc.
    return model_string

def get_pricing_for_model(model_string: str) -> Dict[str, float]:
    """Get pricing information for a model."""
    base_model = get_base_model_name(model_string)
    
    # Try to find exact pricing
    if base_model in PRICING:
        return PRICING[base_model]
    
    # Try to find pricing for a similar model
    for model_name, pricing in PRICING.items():
        if base_model.startswith(model_name):
            return pricing
    
    # Default pricing based on provider
    provider = get_provider_from_model(model_string)
    if provider == LLMProvider.OPENAI:
        return {"input": 2.50, "output": 10.00}  # Default to gpt-4o pricing
    elif provider == LLMProvider.ANTHROPIC:
        return {"input": 3.0, "output": 15.0}  # Default to Claude 3 Sonnet pricing
    elif provider == LLMProvider.GOOGLE:
        return {"input": 0.125, "output": 0.375}  # Default to Gemini 1.0 Pro pricing
    elif provider == LLMProvider.MISTRAL:
        return {"input": 0.2, "output": 0.6}  # Default to Mistral Small pricing
    else:
        return {"input": 0.2, "output": 0.6}  # Generic default

def update_aisuite_response_tracking(
    response: Any, 
    task: str, 
    prompt_tokens: Optional[int] = None, 
    model: Optional[str] = None
) -> None:
    """
    Track token usage from an aisuite response with updated pricing.
    
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
            
        # Get pricing for the model
        model_name = model or "gpt-4o"  # Default to gpt-4o if model not provided
        pricing = get_pricing_for_model(model_name)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
        
        # Update the tracker with accurate costs
        tracker.update(
            input_tokens=input_tokens, 
            output_tokens=output_tokens, 
            task=task
        )
        
        logger.debug(f"Tracked usage for {task}: {input_tokens} input, {output_tokens} output tokens, "
                    f"cost: ${input_cost + output_cost:.6f}")
        
    except Exception as e:
        logger.warning(f"Error tracking token usage: {str(e)}")

def print_usage_report():
    """Print the usage report with accurate pricing."""
    tracker.print_usage_report()

def get_usage_summary() -> Dict[str, Any]:
    """Get the usage summary with enhanced cost information."""
    return tracker.get_enhanced_summary()

def get_browser_use_vision_cost(num_images: int, model: str = "gpt-4o") -> float:
    """
    Calculate the cost of vision API calls for browser-use.
    
    Args:
        num_images: Number of images processed
        model: Model used (default: gpt-4o)
        
    Returns:
        Estimated cost in USD
    """
    # According to OpenAI, each image is approximately 85 tokens for the first 
    # 65KB and 170 tokens for the next 65KB in high-detail mode
    # Browser-use typically uses medium resolution screenshots
    tokens_per_image = 850  # Approximate for medium detail, typical screenshot
    
    total_tokens = num_images * tokens_per_image
    
    # Get pricing for the model
    pricing = get_pricing_for_model(model)
    
    # Calculate input cost (images count as input tokens)
    image_cost = (total_tokens / 1_000_000) * pricing.get("input", 0)
    
    return image_cost
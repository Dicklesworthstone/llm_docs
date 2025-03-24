"""
Configuration utilities for controlling vision resolution in browser-use.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

from llm_docs.utils.llm_docs_logger import logger


class ScreenshotQuality(Enum):
    """Screenshot quality levels for browser-use."""
    LOW = "low"        # Low quality, ~300-400 tokens per image
    MEDIUM = "medium"  # Medium quality, ~600-700 tokens per image
    HIGH = "high"      # High quality, ~850-1000 tokens per image


@dataclass
class VisionConfig:
    """Configuration for browser-use vision capabilities."""
    
    # Whether to use vision at all
    use_vision: bool = True
    
    # Quality level for screenshots
    quality: ScreenshotQuality = ScreenshotQuality.MEDIUM
    
    # Maximum screenshots per request
    max_screenshots: int = 5
    
    # Viewport size (width, height)
    viewport_size: Tuple[int, int] = (1280, 800)
    
    # Custom browser-use Agent options
    agent_options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.agent_options is None:
            self.agent_options = {}
    
    def get_browser_config(self):
        """Get browser-use configuration based on this vision config."""
        from browser_use.browser.browser import BrowserConfig
        from browser_use.browser.context import BrowserContextConfig
        
        # Set resolution based on quality
        if self.quality == ScreenshotQuality.LOW:
            width, height = 800, 600
            logger.info(f"Using low quality vision configuration: {width}x{height}")
        elif self.quality == ScreenshotQuality.MEDIUM:
            width, height = 1280, 800
            logger.info(f"Using medium quality vision configuration: {width}x{height}")
        else:  # HIGH
            width, height = 1600, 1000
            logger.info(f"Using high quality vision configuration: {width}x{height}")
        
        # Create context config without the screenshot_quality parameter
        context_config = BrowserContextConfig(
            browser_window_size={'width': width, 'height': height},
            highlight_elements=False,
            wait_for_network_idle_page_load_time=2.0
        )
        
        # Create browser config
        browser_config = BrowserConfig(
            headless=True,
            disable_security=True,
            new_context_config=context_config
        )
        
        return browser_config
    
    def get_agent_kwargs(self):
        """
        Get Agent constructor keyword arguments based on this vision config.
        
        Returns:
            Dictionary with Agent constructor keyword arguments
        """
        kwargs = {
            "use_vision": self.use_vision,
            **self.agent_options
        }
        
        return kwargs
    
    def get_estimated_token_cost(self, num_screenshots: int = 1) -> float:
        """
        Estimate the token cost for vision usage.
        
        Args:
            num_screenshots: Number of screenshots
            
        Returns:
            Estimated cost in USD
        """
        # Tokens per screenshot based on quality
        if self.quality == ScreenshotQuality.LOW:
            tokens_per_screenshot = 350
        elif self.quality == ScreenshotQuality.MEDIUM:
            tokens_per_screenshot = 650
        else:  # HIGH
            tokens_per_screenshot = 950
        
        # Limit to max screenshots
        actual_screenshots = min(num_screenshots, self.max_screenshots)
        
        # Total tokens
        total_tokens = tokens_per_screenshot * actual_screenshots
        
        # Cost per million tokens (using gpt-4o input price)
        cost_per_million = 2.50
        
        # Calculate cost
        cost = (total_tokens / 1_000_000) * cost_per_million
        
        return cost


# Default configuration
default_vision_config = VisionConfig(
    use_vision=True,
    quality=ScreenshotQuality.MEDIUM,
    max_screenshots=5,
    viewport_size=(1280, 800)
)

# Low-cost configuration
low_cost_vision_config = VisionConfig(
    use_vision=True,
    quality=ScreenshotQuality.LOW,
    max_screenshots=3,
    viewport_size=(800, 600)
)

# High-detail configuration
high_detail_vision_config = VisionConfig(
    use_vision=True,
    quality=ScreenshotQuality.HIGH,
    max_screenshots=10,
    viewport_size=(1600, 1000)
)
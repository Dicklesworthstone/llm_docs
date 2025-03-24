"""
Browser utilities for controlling vision resolution.
"""

from llm_docs.utils.browser.vision_config import (
    ScreenshotQuality,
    VisionConfig,
    default_vision_config,
    high_detail_vision_config,
    low_cost_vision_config,
)
from llm_docs.utils.browser.vision_integration import (
    create_browser_with_vision_config,
    enable_vision_config,
    patch_documentation_extractor_with_vision_config,
    run_agent_with_vision_config,
)

__all__ = [
    'ScreenshotQuality',
    'VisionConfig',
    'default_vision_config',
    'low_cost_vision_config',
    'high_detail_vision_config',
    'create_browser_with_vision_config',
    'run_agent_with_vision_config',
    'patch_documentation_extractor_with_vision_config',
    'enable_vision_config'
]
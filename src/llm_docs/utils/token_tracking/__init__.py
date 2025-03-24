from .token_tracker_adapter import (
    get_usage_summary,
    print_usage_report,
    update_aisuite_response_tracking,
)
from .token_usage_tracker import TokenUsageTracker

# Define which symbols are exported from this module
__all__ = [
    "get_usage_summary",
    "print_usage_report", 
    "update_aisuite_response_tracking",
    "TokenUsageTracker",
    "tracker"
]

# Create a global tracker instance
tracker = TokenUsageTracker()
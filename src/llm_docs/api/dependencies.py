"""
Dependencies for the FastAPI application.
"""

import time
from typing import Dict

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader


# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, rate_limit: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum number of requests per window
            window_seconds: Window size in seconds
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, Dict[float, int]] = {}
        
    def is_rate_limited(self, key: str) -> bool:
        """
        Check if a key is rate limited.
        
        Args:
            key: Identifier for the client (e.g. IP address)
            
        Returns:
            True if rate limited, False otherwise
        """
        now = time.time()
        
        # Initialize if key not seen before
        if key not in self.requests:
            self.requests[key] = {now: 1}
            return False
            
        # Clean up old entries
        self.requests[key] = {
            ts: count for ts, count in self.requests[key].items()
            if now - ts < self.window_seconds
        }
        
        # Count recent requests
        recent_count = sum(self.requests[key].values())
        
        # Add current request
        if now in self.requests[key]:
            self.requests[key][now] += 1
        else:
            self.requests[key][now] = 1
            
        return recent_count >= self.rate_limit


# Create rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(request: Request):
    """
    Rate limiting dependency.
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    key = request.client.host if request.client else "unknown"
    
    if rate_limiter.is_rate_limited(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


# API key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# In a real application, you'd store these in a secure database
API_KEYS = {
    "test-key": {"name": "Test User", "admin": False},
    "admin-key": {"name": "Admin User", "admin": True},
}

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)) -> Dict:
    """
    Verify API key and return user info.
    
    Args:
        api_key: API key from request header
        
    Returns:
        User info dictionary
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key in API_KEYS:
        return API_KEYS[api_key]
        
    raise HTTPException(
        status_code=401,
        detail="Invalid API key"
    )

def require_admin(user_info: Dict = None):
    """
    Require admin privileges.
    
    Args:
        user_info: User info from API key verification
        
    Raises:
        HTTPException: If user is not an admin
    """
    if user_info is None:
        user_info = verify_api_key()
        
    if not user_info.get("admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
        
    return user_info

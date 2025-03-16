"""
Dependencies for the FastAPI application.
"""

import time
from typing import Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from rich.console import Console
from starlette.middleware.base import BaseHTTPMiddleware

# Initialize console
console = Console()

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
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In a real application, you'd store these in a secure database
API_KEYS = {
    "test-key": {"name": "Test User", "admin": False},
    "admin-key": {"name": "Admin User", "admin": True},
}

def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Dict:
    """
    Verify API key and return user info.
    
    Args:
        api_key: API key from request header
        
    Returns:
        User info dictionary
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    if api_key in API_KEYS:
        return API_KEYS[api_key]
        
    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )

def optional_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Optional[Dict]:
    """
    Optionally verify API key without requiring it.
    
    Args:
        api_key: API key from request header
        
    Returns:
        User info dictionary if valid, None otherwise
    """
    if not api_key:
        return None
        
    if api_key in API_KEYS:
        return API_KEYS[api_key]
        
    return None

def require_admin(user_info: Dict = Depends(verify_api_key)):
    """
    Require admin privileges.
    
    Args:
        user_info: User info from API key verification
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not user_info.get("admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
        
    return user_info

# Error handling middleware
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to catch and properly format unhandled exceptions."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            console.print(f"[bold red]Unhandled error: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            
            # Return a JSON response
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

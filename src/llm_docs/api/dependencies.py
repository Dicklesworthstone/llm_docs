"""
Dependencies for the FastAPI application.
"""

import os
import time
from typing import Callable, Dict, Optional

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from rich.console import Console
from starlette.middleware.base import BaseHTTPMiddleware

# Initialize console
console = Console()

# Initialize Redis client for distributed rate limiting
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

# Rate limiting
class RedisRateLimiter:
    """Redis-based rate limiter for distributed environments."""
    
    def __init__(self, rate_limit: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum number of requests per window
            window_seconds: Window size in seconds
        """
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        
    async def is_rate_limited(self, key: str) -> bool:
        """
        Check if a key is rate limited.
        
        Args:
            key: Identifier for rate limiting (e.g., IP address)
            
        Returns:
            True if rate limit exceeded, False otherwise
        """
        now = int(time.time())
        window_start = now - self.window_seconds
        
        # Create Redis key for this rate limit check
        redis_key = f"rate_limit:{key}"
        
        async with redis_client.pipeline() as pipe:
            # Add current timestamp to sorted set
            await pipe.zadd(redis_key, {str(now): now})
            
            # Remove timestamps outside the current window
            await pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count requests in the current window
            await pipe.zcard(redis_key)
            
            # Set TTL on the key to auto-cleanup
            await pipe.expire(redis_key, self.window_seconds * 2)
            
            # Execute pipeline
            _, _, request_count, _ = await pipe.execute()
            
            return request_count > self.rate_limit


# Create rate limiter instance
rate_limiter = RedisRateLimiter()

async def rate_limit(request: Request):
    """
    Rate limiting dependency.
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    key = request.client.host if request.client else "unknown"
    
    if await rate_limiter.is_rate_limited(key):
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
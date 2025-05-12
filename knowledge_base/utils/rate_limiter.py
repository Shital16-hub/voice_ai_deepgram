"""
Rate limiting utilities for OpenAI API.
"""
import asyncio
import time
from typing import Dict, Optional
import redis.asyncio as redis
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting for OpenAI API calls."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize rate limiter."""
        self.redis = redis_client or redis.from_url("redis://localhost:6379/0")
        self.max_requests_per_minute = 60
        self.max_tokens_per_day = 1000000
    
    async def check_rate_limit(self, user_id: str, tokens: int) -> bool:
        """Check if request is within rate limits."""
        try:
            current_time = datetime.now()
            
            # Check requests per minute
            minute_key = f"rate_limit:requests:{user_id}:{current_time.strftime('%Y%m%d%H%M')}"
            requests_count = await self.redis.incr(minute_key)
            await self.redis.expire(minute_key, 60)
            
            if requests_count > self.max_requests_per_minute:
                logger.warning(f"Rate limit exceeded for user {user_id}: {requests_count} requests/minute")
                return False
            
            # Check tokens per day
            day_key = f"rate_limit:tokens:{user_id}:{current_time.strftime('%Y%m%d')}"
            token_count = await self.redis.incrby(day_key, tokens)
            await self.redis.expire(day_key, 86400)  # 24 hours
            
            if token_count > self.max_tokens_per_day:
                logger.warning(f"Token limit exceeded for user {user_id}: {token_count} tokens/day")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Allow the request if rate limiting fails
            return True
    
    async def get_remaining_quota(self, user_id: str) -> Dict[str, int]:
        """Get remaining quota for user."""
        try:
            current_time = datetime.now()
            
            # Get current usage
            minute_key = f"rate_limit:requests:{user_id}:{current_time.strftime('%Y%m%d%H%M')}"
            day_key = f"rate_limit:tokens:{user_id}:{current_time.strftime('%Y%m%d')}"
            
            requests_used = int(await self.redis.get(minute_key) or 0)
            tokens_used = int(await self.redis.get(day_key) or 0)
            
            return {
                "requests_remaining": max(0, self.max_requests_per_minute - requests_used),
                "tokens_remaining": max(0, self.max_tokens_per_day - tokens_used)
            }
        except Exception as e:
            logger.error(f"Error getting remaining quota: {e}")
            return {
                "requests_remaining": self.max_requests_per_minute,
                "tokens_remaining": self.max_tokens_per_day
            }
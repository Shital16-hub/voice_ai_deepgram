"""
Rate limiting utilities for OpenAI API with fallback when Redis is unavailable.
"""
import asyncio
import time
from typing import Dict, Optional,Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting for OpenAI API calls with Redis fallback."""
    
    def __init__(self, redis_client: Optional[Any] = None):
        """Initialize rate limiter."""
        self.max_requests_per_minute = 60
        self.max_tokens_per_day = 1000000
        self.redis_available = False
        self.memory_store = {}  # Fallback in-memory storage
        
        # Try to initialize Redis
        try:
            import redis.asyncio as redis
            self.redis = redis_client or redis.from_url("redis://localhost:6379/0", socket_timeout=1)
            # Test Redis connection
            asyncio.create_task(self._test_redis_connection())
        except ImportError:
            logger.warning("Redis not available - using memory storage for rate limiting")
            self.redis = None
        except Exception as e:
            logger.warning(f"Redis connection failed - using memory storage: {e}")
            self.redis = None
    
    async def _test_redis_connection(self):
        """Test Redis connection and set availability flag."""
        try:
            if self.redis:
                await asyncio.wait_for(self.redis.ping(), timeout=1.0)
                self.redis_available = True
                logger.info("Redis rate limiter available")
        except Exception as e:
            logger.warning(f"Redis test failed - using memory storage: {e}")
            self.redis_available = False
    
    async def check_rate_limit(self, user_id: str, tokens: int) -> bool:
        """Check if request is within rate limits."""
        current_time = datetime.now()
        
        # Check requests per minute
        minute_key = f"rate_limit:requests:{user_id}:{current_time.strftime('%Y%m%d%H%M')}"
        day_key = f"rate_limit:tokens:{user_id}:{current_time.strftime('%Y%m%d')}"
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                requests_count = await self.redis.incr(minute_key)
                await self.redis.expire(minute_key, 60)
                
                token_count = await self.redis.incrby(day_key, tokens)
                await self.redis.expire(day_key, 86400)  # 24 hours
                
                if requests_count > self.max_requests_per_minute:
                    logger.warning(f"Rate limit exceeded for user {user_id}: {requests_count} requests/minute")
                    return False
                
                if token_count > self.max_tokens_per_day:
                    logger.warning(f"Token limit exceeded for user {user_id}: {token_count} tokens/day")
                    return False
                
                return True
            except Exception as e:
                logger.warning(f"Redis rate limit error: {e}")
                self.redis_available = False
        
        # Fallback to memory storage
        # Clean old entries
        self._cleanup_memory_store()
        
        # Check minute limit
        requests_count = self.memory_store.get(minute_key, 0) + 1
        self.memory_store[minute_key] = requests_count
        
        # Check day limit
        token_count = self.memory_store.get(day_key, 0) + tokens
        self.memory_store[day_key] = token_count
        
        if requests_count > self.max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for user {user_id}: {requests_count} requests/minute")
            return False
        
        if token_count > self.max_tokens_per_day:
            logger.warning(f"Token limit exceeded for user {user_id}: {token_count} tokens/day")
            return False
        
        return True
    
    def _cleanup_memory_store(self):
        """Clean up old entries from memory store."""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key in list(self.memory_store.keys()):
            if 'requests:' in key:
                # Extract timestamp from key
                try:
                    time_str = key.split(':')[-1]
                    key_time = datetime.strptime(time_str, '%Y%m%d%H%M')
                    if (current_time - key_time).total_seconds() > 60:
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)
            elif 'tokens:' in key:
                # Extract date from key
                try:
                    date_str = key.split(':')[-1]
                    key_date = datetime.strptime(date_str, '%Y%m%d')
                    if (current_time - key_date).days > 1:
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_store[key]
    
    async def get_remaining_quota(self, user_id: str) -> Dict[str, int]:
        """Get remaining quota for user."""
        current_time = datetime.now()
        minute_key = f"rate_limit:requests:{user_id}:{current_time.strftime('%Y%m%d%H%M')}"
        day_key = f"rate_limit:tokens:{user_id}:{current_time.strftime('%Y%m%d')}"
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                requests_used = int(await self.redis.get(minute_key) or 0)
                tokens_used = int(await self.redis.get(day_key) or 0)
                
                return {
                    "requests_remaining": max(0, self.max_requests_per_minute - requests_used),
                    "tokens_remaining": max(0, self.max_tokens_per_day - tokens_used)
                }
            except Exception as e:
                logger.warning(f"Redis quota error: {e}")
                self.redis_available = False
        
        # Fallback to memory storage
        requests_used = self.memory_store.get(minute_key, 0)
        tokens_used = self.memory_store.get(day_key, 0)
        
        return {
            "requests_remaining": max(0, self.max_requests_per_minute - requests_used),
            "tokens_remaining": max(0, self.max_tokens_per_day - tokens_used)
        }
"""
Caching utilities for knowledge base queries.
"""
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manage caching for knowledge base queries."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, ttl: int = 3600):
        """Initialize cache manager."""
        self.redis = redis_client or redis.from_url("redis://localhost:6379/0")
        self.ttl = ttl
    
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        # Create hash from query and context
        content = query
        if context:
            content += json.dumps(context, sort_keys=True)
        
        return f"cache:query:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        """Get cached response."""
        try:
            cache_key = self._generate_cache_key(query, context)
            cached_value = await self.redis.get(cache_key)
            
            if cached_value:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached_value)
            
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, query: str, response: str, context: Optional[Dict] = None):
        """Cache response."""
        try:
            cache_key = self._generate_cache_key(query, context)
            await self.redis.setex(
                cache_key, 
                self.ttl, 
                json.dumps(response)
            )
            logger.debug(f"Cached response for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache entries matching pattern."""
        try:
            keys = await self.redis.keys(f"cache:query:{pattern}*")
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
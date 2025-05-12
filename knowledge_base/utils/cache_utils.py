"""
Caching utilities for knowledge base queries with fallback when Redis is unavailable.
"""
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manage caching for knowledge base queries with Redis fallback."""
    
    def __init__(self, redis_client: Optional[Any] = None, ttl: int = 3600):
        """Initialize cache manager."""
        self.ttl = ttl
        self.redis_available = False
        self.memory_cache = {}  # Fallback in-memory cache
        self.max_memory_items = 100  # Limit memory cache size
        
        # Try to initialize Redis
        try:
            import redis.asyncio as redis
            self.redis = redis_client or redis.from_url("redis://localhost:6379/0", socket_timeout=1)
            # Test Redis connection
            asyncio.create_task(self._test_redis_connection())
        except ImportError:
            logger.warning("Redis not available - using memory cache only")
            self.redis = None
        except Exception as e:
            logger.warning(f"Redis connection failed - using memory cache only: {e}")
            self.redis = None
    
    async def _test_redis_connection(self):
        """Test Redis connection and set availability flag."""
        try:
            if self.redis:
                await asyncio.wait_for(self.redis.ping(), timeout=1.0)
                self.redis_available = True
                logger.info("Redis cache available")
        except Exception as e:
            logger.warning(f"Redis test failed - using memory cache: {e}")
            self.redis_available = False
    
    def _generate_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        # Create hash from query and context
        content = query
        if context:
            content += json.dumps(context, sort_keys=True)
        
        return f"cache:query:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        """Get cached response."""
        cache_key = self._generate_cache_key(query, context)
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                cached_value = await self.redis.get(cache_key)
                if cached_value:
                    logger.debug(f"Redis cache hit for query: {query[:50]}...")
                    return json.loads(cached_value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self.redis_available = False
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            logger.debug(f"Memory cache hit for query: {query[:50]}...")
            return self.memory_cache[cache_key]
        
        return None
    
    async def set(self, query: str, response: str, context: Optional[Dict] = None):
        """Cache response."""
        cache_key = self._generate_cache_key(query, context)
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                await self.redis.setex(
                    cache_key, 
                    self.ttl, 
                    json.dumps(response)
                )
                logger.debug(f"Cached response in Redis for query: {query[:50]}...")
                return
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self.redis_available = False
        
        # Fallback to memory cache
        self.memory_cache[cache_key] = response
        
        # Limit memory cache size
        if len(self.memory_cache) > self.max_memory_items:
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        logger.debug(f"Cached response in memory for query: {query[:50]}...")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all cache entries matching pattern."""
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                keys = await self.redis.keys(f"cache:query:{pattern}*")
                if keys:
                    await self.redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} Redis cache entries")
                return
            except Exception as e:
                logger.warning(f"Redis invalidate error: {e}")
                self.redis_available = False
        
        # Fallback to memory cache
        keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.memory_cache[key]
        logger.info(f"Invalidated {len(keys_to_remove)} memory cache entries")
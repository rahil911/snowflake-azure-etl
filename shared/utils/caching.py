"""
Caching utilities for the multi-agent data intelligence platform.

This module provides comprehensive caching functionality including decorators,
cache managers, and multi-level caching strategies for improved performance.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, ConfigDict


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


class CacheEntry(BaseModel):
    """Cache entry with metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    tags: List[str] = Field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = CacheStats()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> None:
        """Set value by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class MemoryCache(BaseCache):
    """In-memory cache implementation with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        super().__init__(max_size, default_ttl)
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access info
            entry.touch()
            self.stats.hits += 1
            
            return entry.value
    
    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get for non-async contexts."""
        if key not in self._cache:
            self.stats.misses += 1
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired:
            del self._cache[key]
            self.stats.misses += 1
            self.stats.evictions += 1
            return None
        
        entry.touch()
        self.stats.hits += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> None:
        """Set value by key."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            ttl = ttl or self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl,
                tags=tags or []
            )
            
            self._cache[key] = entry
            self.stats.sets += 1
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> None:
        """Synchronous set for non-async contexts."""
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru_sync()
        
        ttl = ttl or self.default_ttl
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl,
            tags=tags or []
        )
        
        self._cache[key] = entry
        self.stats.sets += 1
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats.deletes += 1
                return True
            return False
    
    def delete_sync(self, key: str) -> bool:
        """Synchronous delete for non-async contexts."""
        if key in self._cache:
            del self._cache[key]
            self.stats.deletes += 1
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries by tags."""
        deleted = 0
        async with self._lock:
            keys_to_delete = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._cache[key]
                deleted += 1
                self.stats.deletes += 1
        
        return deleted
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        expired_keys = []
        async with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self.stats.evictions += 1
        
        return len(expired_keys)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self.stats.evictions += 1
    
    def _evict_lru_sync(self) -> None:
        """Synchronous LRU eviction."""
        if not self._cache:
            return
        
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
        self.stats.evictions += 1
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'default_ttl': self.default_ttl,
            'stats': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'hit_rate': self.stats.hit_rate,
                'sets': self.stats.sets,
                'deletes': self.stats.deletes,
                'evictions': self.stats.evictions
            },
            'entries': [
                {
                    'key': entry.key,
                    'age_seconds': entry.age_seconds,
                    'access_count': entry.access_count,
                    'is_expired': entry.is_expired,
                    'tags': entry.tags
                }
                for entry in self._cache.values()
            ]
        }


class CacheManager:
    """Multi-level cache manager with different cache strategies."""
    
    def __init__(self):
        self.caches: Dict[str, BaseCache] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Create default caches
        self.register_cache('memory', MemoryCache(max_size=1000, default_ttl=3600))
        self.register_cache('short_term', MemoryCache(max_size=500, default_ttl=300))
        self.register_cache('long_term', MemoryCache(max_size=100, default_ttl=86400))
    
    def register_cache(self, name: str, cache: BaseCache) -> None:
        """Register a cache instance."""
        self.caches[name] = cache
        self.logger.info(f"Registered cache: {name}")
    
    def get_cache(self, name: str = 'memory') -> Optional[BaseCache]:
        """Get cache by name."""
        return self.caches.get(name)
    
    async def start_cleanup_task(self, interval_seconds: int = 300) -> None:
        """Start periodic cleanup of expired entries."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.cleanup_all_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def cleanup_all_expired(self) -> int:
        """Cleanup expired entries from all caches."""
        total_cleaned = 0
        for name, cache in self.caches.items():
            if hasattr(cache, 'cleanup_expired'):
                cleaned = await cache.cleanup_expired()
                total_cleaned += cleaned
                if cleaned > 0:
                    self.logger.debug(f"Cleaned {cleaned} expired entries from {name} cache")
        return total_cleaned
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            name: {
                'stats': cache.get_stats().__dict__,
                'info': cache.get_cache_info() if hasattr(cache, 'get_cache_info') else {}
            }
            for name, cache in self.caches.items()
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def _generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    # Create a stable representation of arguments
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {}
    }
    
    # Serialize and hash
    key_str = json.dumps(key_data, default=str, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_result(
    ttl: int = 3600,
    cache_name: str = 'memory',
    key_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        cache_name: Name of cache to use
        key_prefix: Optional prefix for cache key
        tags: Optional tags for cache entry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_manager().get_cache(cache_name)
            if not cache:
                return await func(*args, **kwargs)
            
            # Generate cache key
            base_key = _generate_cache_key(func.__name__, *args, **kwargs)
            cache_key = f"{key_prefix}:{base_key}" if key_prefix else base_key
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache_manager().get_cache(cache_name)
            if not cache or not hasattr(cache, 'get_sync'):
                return func(*args, **kwargs)
            
            # Generate cache key
            base_key = _generate_cache_key(func.__name__, *args, **kwargs)
            cache_key = f"{key_prefix}:{base_key}" if key_prefix else base_key
            
            # Try to get from cache
            result = cache.get_sync(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set_sync(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


async def invalidate_cache( # Changed to async def
    cache_name: str = 'memory',
    key: Optional[str] = None,
    key_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> int:
    """
    Invalidate cache entries.
    
    Args:
        cache_name: Name of cache
        key: Specific key to invalidate
        key_prefix: Prefix to match for invalidation
        tags: Tags to match for invalidation
    
    Returns:
        Number of entries invalidated
    """
    cache = get_cache_manager().get_cache(cache_name)
    if not cache:
        return 0
    
    invalidated = 0
    
    if key:
        # Invalidate specific key
        # Prefer async 'delete' if available and it's an async method, else use 'delete_sync'
        if hasattr(cache, 'delete') and asyncio.iscoroutinefunction(getattr(cache, 'delete')):
            if await cache.delete(key): # type: ignore
                invalidated = 1
        elif hasattr(cache, 'delete_sync'):
            if cache.delete_sync(key): # type: ignore
                invalidated = 1
    elif tags:
        # Invalidate by tags (if supported and async)
        if hasattr(cache, 'delete_by_tags') and asyncio.iscoroutinefunction(getattr(cache, 'delete_by_tags')):
            invalidated = await cache.delete_by_tags(tags) # type: ignore
        # Add a sync fallback if one existed or was intended for other cache types
        # For now, this relies on delete_by_tags being async as per MemoryCache.
    elif key_prefix:
        # Invalidate by prefix
        # MemoryCache.keys() is synchronous. This part of the logic remains internally synchronous.
        # If a cache implementation offers async key iteration, this could be further optimized.
        keys_to_delete = [k for k in cache.keys() if k.startswith(key_prefix)]
        for k_to_delete in keys_to_delete:
            # Prefer async 'delete' if available, else use 'delete_sync'
            if hasattr(cache, 'delete') and asyncio.iscoroutinefunction(getattr(cache, 'delete')):
                if await cache.delete(k_to_delete): # type: ignore
                    invalidated += 1
            elif hasattr(cache, 'delete_sync'):
                if cache.delete_sync(k_to_delete): # type: ignore
                    invalidated += 1
    else:
        # Clear entire cache
        # Prefer async 'clear' if available
        if hasattr(cache, 'clear') and asyncio.iscoroutinefunction(getattr(cache, 'clear')):
            await cache.clear() # type: ignore
        elif hasattr(cache, 'clear_sync'): # Assuming a clear_sync might exist for some
            cache.clear_sync() # type: ignore
        elif hasattr(cache, 'clear'): # If 'clear' is not async but exists (e.g. from BaseCache not overridden as async)
            cache.clear() # type: ignore
        invalidated = -1  # Indicate full clear
    
    return invalidated


# Specialized cache decorators

def cache_query_result(ttl: int = 1800, cache_name: str = 'memory'):
    """Cache database query results."""
    return cache_result(ttl=ttl, cache_name=cache_name, key_prefix='query', tags=['database'])


def cache_api_result(ttl: int = 300, cache_name: str = 'short_term'):
    """Cache API call results."""
    return cache_result(ttl=ttl, cache_name=cache_name, key_prefix='api', tags=['external'])


def cache_computation(ttl: int = 3600, cache_name: str = 'memory'):
    """Cache expensive computation results."""
    return cache_result(ttl=ttl, cache_name=cache_name, key_prefix='compute', tags=['computation'])


def cache_analytics(ttl: int = 7200, cache_name: str = 'long_term'):
    """Cache analytics and report results."""
    return cache_result(ttl=ttl, cache_name=cache_name, key_prefix='analytics', tags=['analytics'])


# Add alias for MemoryCache as InMemoryCache
InMemoryCache = MemoryCache


class RedisCache(BaseCache):
    """Redis-based cache implementation (placeholder for future Redis integration)."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, redis_url: str = "redis://localhost:6379"):
        super().__init__(max_size, default_ttl)
        self.redis_url = redis_url
        # For now, fallback to memory cache until Redis integration
        self._fallback = MemoryCache(max_size, default_ttl)
        self.logger.warning("RedisCache not yet implemented, falling back to MemoryCache")
    
    def get(self, key: str) -> Optional[Any]:
        return self._fallback.get_sync(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> None:
        self._fallback.set_sync(key, value, ttl, tags)
    
    def delete(self, key: str) -> bool:
        return self._fallback.delete_sync(key)
    
    def clear(self) -> None:
        self._fallback._cache.clear()
    
    def keys(self) -> List[str]:
        return list(self._fallback._cache.keys())


class MultiLevelCache(BaseCache):
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        super().__init__(max_size, default_ttl)
        self.l1_cache = MemoryCache(max_size // 2, default_ttl // 2)  # Faster, smaller
        self.l2_cache = RedisCache(max_size, default_ttl)  # Larger, persistent
    
    def get(self, key: str) -> Optional[Any]:
        # Try L1 first
        value = self.l1_cache.get_sync(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set_sync(key, value)
            self.stats.hits += 1
            return value
        
        self.stats.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: Optional[List[str]] = None) -> None:
        # Set in both levels
        self.l1_cache.set_sync(key, value, ttl, tags)
        self.l2_cache.set(key, value, ttl, tags)
        self.stats.sets += 1
    
    def delete(self, key: str) -> bool:
        deleted_l1 = self.l1_cache.delete_sync(key)
        deleted_l2 = self.l2_cache.delete(key)
        if deleted_l1 or deleted_l2:
            self.stats.deletes += 1
            return True
        return False
    
    def clear(self) -> None:
        self.l1_cache.clear()
        self.l2_cache.clear()
    
    def keys(self) -> List[str]:
        l1_keys = set(self.l1_cache.keys())
        l2_keys = set(self.l2_cache.keys())
        return list(l1_keys.union(l2_keys))


# Add missing functions and classes
def cached(ttl: int = 3600, cache_name: str = 'memory'):
    """Decorator for caching function results."""
    return cache_result(ttl=ttl, cache_name=cache_name)


def timed_cache(ttl: int = 3600):
    """Decorator for time-based caching."""
    return cache_result(ttl=ttl)


def cache_key_builder(*args, **kwargs) -> str:
    """Build cache key from arguments."""
    return _generate_cache_key(*args, **kwargs)


class CacheConfig:
    """Cache configuration settings."""
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600, cleanup_interval: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval


class CacheError(Exception):
    """Base cache error."""
    pass


class CacheKeyError(CacheError):
    """Cache key related error."""
    pass


class CacheMissError(CacheError):
    """Cache miss error."""
    pass 
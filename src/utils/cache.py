"""
Intelligent caching system for financial data APIs.
Provides in-memory + disk persistence with LRU eviction.
"""

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import wraps

import diskcache as dc
from pydantic import BaseModel


class CacheConfig(BaseModel):
    """Configuration for the cache system."""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB in-memory
    max_disk_size: int = 1024 * 1024 * 1024   # 1GB on disk
    cache_dir: str = "cache"
    default_ttl: int = 3600  # 1 hour default TTL
    compression_level: int = 1  # Fast compression
    enable_async: bool = True


class CacheStats(BaseModel):
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    memory_size: int = 0
    disk_size: int = 0


class FinancialDataCache:
    """
    High-performance cache for financial data with memory + disk persistence.
    
    Features:
    - In-memory LRU cache for hot data
    - Disk persistence for cold data
    - Configurable size limits
    - TTL support
    - Async operations
    - Compression
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # Create cache directory
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        # Initialize diskcache with our settings
        self.cache = dc.Cache(
            directory=str(cache_path),
            size_limit=self.config.max_disk_size,
            eviction_policy='least-recently-used',
            compress_level=self.config.compression_level
        )
        
        # In-memory cache for hot data (faster access)
        self._memory_cache: Dict[str, Any] = {}
        self._memory_access_times: Dict[str, float] = {}
        self._memory_size = 0
        
        print(f"✅ Cache initialized: {self.config.cache_dir}")
        print(f"   Memory limit: {self.config.max_memory_size / 1024 / 1024:.1f}MB")
        print(f"   Disk limit: {self.config.max_disk_size / 1024 / 1024:.1f}MB")
    
    def _generate_key(self, data_type: str, **kwargs) -> str:
        """Generate a consistent cache key from parameters."""
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_data = f"{data_type}:{json.dumps(sorted_kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_memory_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        try:
            return len(json.dumps(data).encode())
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            return len(str(data).encode())
    
    def _evict_memory_if_needed(self):
        """Evict least recently used items from memory cache."""
        while (self._memory_size > self.config.max_memory_size and 
               self._memory_cache):
            
            # Find least recently used item
            lru_key = min(self._memory_access_times.keys(), 
                         key=lambda k: self._memory_access_times[k])
            
            # Remove from memory cache
            if lru_key in self._memory_cache:
                self._memory_size -= self._get_memory_size(self._memory_cache[lru_key])
                del self._memory_cache[lru_key]
                del self._memory_access_times[lru_key]
                self.stats.evictions += 1
    
    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """Check if cached data is expired."""
        return time.time() - timestamp > ttl
    
    def get(self, data_type: str, ttl: Optional[int] = None, **kwargs) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            data_type: Type of data (e.g., 'prices', 'financials')
            ttl: Time-to-live override (uses default if None)
            **kwargs: Parameters to generate cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        key = self._generate_key(data_type, **kwargs)
        ttl = ttl or self.config.default_ttl
        
        self.stats.total_requests += 1
        
        # Check memory cache first
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if not self._is_expired(timestamp, ttl):
                self._memory_access_times[key] = time.time()
                self.stats.hits += 1
                return data
            else:
                # Expired, remove from memory
                self._memory_size -= self._get_memory_size(data)
                del self._memory_cache[key]
                del self._memory_access_times[key]
        
        # Check disk cache
        try:
            cached_data = self.cache.get(key)
            if cached_data:
                data, timestamp = cached_data
                if not self._is_expired(timestamp, ttl):
                    # Move to memory cache for faster future access
                    self._memory_cache[key] = cached_data
                    self._memory_access_times[key] = time.time()
                    self._memory_size += self._get_memory_size(data)
                    self._evict_memory_if_needed()
                    
                    self.stats.hits += 1
                    return data
                else:
                    # Expired, remove from disk
                    self.cache.delete(key)
        except Exception as e:
            print(f"⚠️  Cache read error: {e}")
        
        self.stats.misses += 1
        return None
    
    def set(self, data_type: str, data: Any, ttl: Optional[int] = None, **kwargs):
        """
        Store data in cache.
        
        Args:
            data_type: Type of data (e.g., 'prices', 'financials')
            data: Data to cache
            ttl: Time-to-live override (uses default if None)
            **kwargs: Parameters to generate cache key
        """
        key = self._generate_key(data_type, **kwargs)
        ttl = ttl or self.config.default_ttl
        timestamp = time.time()
        
        cache_entry = (data, timestamp)
        
        # Store in memory cache
        self._memory_cache[key] = cache_entry
        self._memory_access_times[key] = timestamp
        self._memory_size += self._get_memory_size(data)
        self._evict_memory_if_needed()
        
        # Store in disk cache asynchronously
        if self.config.enable_async:
            try:
                # Check if there's a running event loop
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._store_to_disk_async(key, cache_entry))
            except RuntimeError:
                # No event loop running, fall back to synchronous
                try:
                    self.cache.set(key, cache_entry, expire=ttl)
                except Exception as e:
                    print(f"⚠️  Cache write error: {e}")
        else:
            try:
                self.cache.set(key, cache_entry, expire=ttl)
            except Exception as e:
                print(f"⚠️  Cache write error: {e}")
    
    async def _store_to_disk_async(self, key: str, data: Any):
        """Asynchronously store data to disk cache."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.cache.set, key, data)
        except Exception as e:
            print(f"⚠️  Async cache write error: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        self.stats.hit_rate = (self.stats.hits / max(self.stats.total_requests, 1)) * 100
        self.stats.memory_size = self._memory_size
        self.stats.disk_size = self.cache.volume()
        return self.stats
    
    def clear(self):
        """Clear all cached data."""
        self._memory_cache.clear()
        self._memory_access_times.clear()
        self._memory_size = 0
        self.cache.clear()
        self.stats = CacheStats()
        print("🗑️  Cache cleared")
    
    def close(self):
        """Close cache and cleanup resources."""
        self.cache.close()
        print("🔒 Cache closed")


# Global cache instance
_cache_instance: Optional[FinancialDataCache] = None


def get_cache() -> FinancialDataCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = FinancialDataCache()
    return _cache_instance


def cached(data_type: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Usage:
        @cached('prices', ttl=1800)  # 30 minutes
        def get_prices(ticker, start_date, end_date):
            # API call here
            return data
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Try to get from cache
            cached_result = cache.get(data_type, ttl=ttl, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # Not in cache, call function
            result = func(*args, **kwargs)
            
            # Store in cache
            if result is not None:
                cache.set(data_type, result, ttl=ttl, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def print_cache_stats():
    """Print cache performance statistics."""
    cache = get_cache()
    stats = cache.get_stats()
    
    print(f"\n📊 Cache Statistics:")
    print(f"   Hit Rate: {stats.hit_rate:.1f}% ({stats.hits}/{stats.total_requests})")
    print(f"   Memory: {stats.memory_size / 1024 / 1024:.1f}MB")
    print(f"   Disk: {stats.disk_size / 1024 / 1024:.1f}MB")
    print(f"   Evictions: {stats.evictions}")


# Convenience functions for common data types
def cache_prices(ticker: str, start_date: str, end_date: str, data: Any, ttl: int = 1800):
    """Cache price data (30 min TTL)."""
    get_cache().set('prices', data, ttl=ttl, ticker=ticker, start_date=start_date, end_date=end_date)


def get_cached_prices(ticker: str, start_date: str, end_date: str, ttl: int = 1800) -> Optional[Any]:
    """Get cached price data."""
    return get_cache().get('prices', ttl=ttl, ticker=ticker, start_date=start_date, end_date=end_date)


def cache_financials(ticker: str, end_date: str, limit: int, data: Any, ttl: int = 3600):
    """Cache financial metrics (1 hour TTL)."""
    get_cache().set('financials', data, ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_financials(ticker: str, end_date: str, limit: int, ttl: int = 3600) -> Optional[Any]:
    """Get cached financial metrics."""
    return get_cache().get('financials', ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)


def cache_news(ticker: str, end_date: str, limit: int, data: Any, ttl: int = 1800):
    """Cache news data (30 min TTL)."""
    get_cache().set('news', data, ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_news(ticker: str, end_date: str, limit: int, ttl: int = 1800) -> Optional[Any]:
    """Get cached news data."""
    return get_cache().get('news', ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)


def cache_insider_trades(ticker: str, end_date: str, limit: int, data: Any, ttl: int = 3600):
    """Cache insider trades (1 hour TTL)."""
    get_cache().set('insider_trades', data, ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_insider_trades(ticker: str, end_date: str, limit: int, ttl: int = 3600) -> Optional[Any]:
    """Get cached insider trades."""
    return get_cache().get('insider_trades', ttl=ttl, ticker=ticker, end_date=end_date, limit=limit)

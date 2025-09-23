"""
Intelligent caching system for financial data APIs.
Provides in-memory + disk persistence with LRU eviction.
"""

import asyncio
import hashlib
import json
import os
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import diskcache as dc
from pydantic import BaseModel


class CacheConfig(BaseModel):
    """Configuration for the cache system."""
    max_memory_size: int = 100 * 1024 * 1024  # 100MB in-memory
    max_disk_size: int = 1024 * 1024 * 1024   # 1GB on disk
    cache_dir: str = "cache"
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
        
        # Async disk write infrastructure
        self._disk_queue = queue.Queue(maxsize=1000)  # Prevent memory bloat
        self._disk_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache-disk")
        self._disk_writer_thread = threading.Thread(target=self._disk_writer_loop, daemon=True)
        self._disk_writer_thread.start()
        
        # Thread safety
        self._memory_lock = threading.RLock()
        
        print(f"✅ Cache initialized: {self.config.cache_dir}")
        print(f"   Memory limit: {self.config.max_memory_size / 1024 / 1024:.1f}MB")
        print(f"   Disk limit: {self.config.max_disk_size / 1024 / 1024:.1f}MB")
        print(f"   Async disk writes: Enabled")
    
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
    
    def _disk_writer_loop(self):
        """Background thread that processes disk write queue."""
        while True:
            try:
                # Get next disk write task (blocks until available)
                key, data = self._disk_queue.get(timeout=1.0)
                
                # Perform disk write
                self.cache.set(key, data)
                self._disk_queue.task_done()
                
            except queue.Empty:
                # Timeout - continue loop (allows thread to be daemon)
                continue
            except Exception as e:
                print(f"⚠️  Background disk write error: {e}")
                self._disk_queue.task_done()
    
    def _queue_disk_write(self, key: str, data: Any):
        """Queue a disk write operation for background processing."""
        try:
            # Non-blocking put - if queue is full, skip disk write
            self._disk_queue.put_nowait((key, data))
        except queue.Full:
            # Queue is full - skip this disk write to prevent blocking
            print(f"⚠️  Disk write queue full, skipping write for key: {key[:20]}...")
    
    def _evict_memory_if_needed(self):
        """Evict least recently used items from memory cache and queue disk write."""
        with self._memory_lock:
            while (self._memory_size > self.config.max_memory_size and 
                   self._memory_cache):
                
                # Find least recently used item
                lru_key = min(self._memory_access_times.keys(), 
                             key=lambda k: self._memory_access_times[k])
                
                # Remove from memory cache
                if lru_key in self._memory_cache:
                    # Get the data before removing
                    evicted_data = self._memory_cache[lru_key]
                    
                    # Queue disk write (non-blocking)
                    self._queue_disk_write(lru_key, evicted_data)
                    
                    # Now remove from memory
                    self._memory_size -= self._get_memory_size(evicted_data)
                    del self._memory_cache[lru_key]
                    del self._memory_access_times[lru_key]
                    self.stats.evictions += 1
    
    
    def get(self, data_type: str, **kwargs) -> Optional[Any]:
        """
        Get data from cache (LRU eviction only, no TTL).
        
        Flow: Memory Cache → Disk Cache → API (if both miss)
        
        Args:
            data_type: Type of data (e.g., 'prices', 'financials')
            **kwargs: Parameters to generate cache key
            
        Returns:
            Cached data or None if not found
        """
        key = self._generate_key(data_type, **kwargs)
        
        self.stats.total_requests += 1
        
        # Step 1: Check memory cache first (fastest)
        with self._memory_lock:
            if key in self._memory_cache:
                data, timestamp = self._memory_cache[key]
                self._memory_access_times[key] = time.time()
                self.stats.hits += 1
                return data
        
        # Step 2: Check disk cache (slower but persistent)
        try:
            cached_data = self.cache.get(key)
            if cached_data:
                data, timestamp = cached_data
                # Hydrate from disk to memory for faster future access
                with self._memory_lock:
                    self._memory_cache[key] = cached_data
                    self._memory_access_times[key] = time.time()
                    self._memory_size += self._get_memory_size(data)
                    self._evict_memory_if_needed()
                
                self.stats.hits += 1
                return data
        except Exception as e:
            print(f"⚠️  Disk cache read error: {e}")
        
        # Step 3: True cache miss - data not in memory or disk
        self.stats.misses += 1
        return None
    
    def set(self, data_type: str, data: Any, **kwargs):
        """
        Store data in cache (LRU eviction only, no TTL).
        
        Flow: Store in Memory → Queue Disk Write → Evict if needed
        
        Args:
            data_type: Type of data (e.g., 'prices', 'financials')
            data: Data to cache
            **kwargs: Parameters to generate cache key
        """
        key = self._generate_key(data_type, **kwargs)
        timestamp = time.time()
        
        cache_entry = (data, timestamp)
        
        # Step 1: Store in memory cache (fast access) - thread-safe
        with self._memory_lock:
            self._memory_cache[key] = cache_entry
            self._memory_access_times[key] = timestamp
            self._memory_size += self._get_memory_size(data)
            
            # Step 2: Queue disk write (non-blocking)
            self._queue_disk_write(key, cache_entry)
            
            # Step 3: Evict from memory if needed (after queuing disk write)
            self._evict_memory_if_needed()
    
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
        # Wait for pending disk writes to complete
        try:
            self._disk_queue.join()  # Wait for all queued writes to complete
        except Exception as e:
            print(f"⚠️  Error waiting for disk writes: {e}")
        
        # Shutdown thread pool
        try:
            self._disk_executor.shutdown(wait=True)
        except Exception as e:
            print(f"⚠️  Error shutting down disk executor: {e}")
        
        # Close disk cache
        self.cache.close()
        print("🔒 Cache closed (async disk writes completed)")


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
def cache_prices(ticker: str, start_date: str, end_date: str, data: Any):
    """Cache price data (LRU eviction only)."""
    get_cache().set('prices', data, ticker=ticker, start_date=start_date, end_date=end_date)


def get_cached_prices(ticker: str, start_date: str, end_date: str) -> Optional[Any]:
    """Get cached price data."""
    return get_cache().get('prices', ticker=ticker, start_date=start_date, end_date=end_date)


def cache_financials(ticker: str, end_date: str, limit: int, data: Any):
    """Cache financial metrics (LRU eviction only)."""
    get_cache().set('financials', data, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_financials(ticker: str, end_date: str, limit: int) -> Optional[Any]:
    """Get cached financial metrics."""
    return get_cache().get('financials', ticker=ticker, end_date=end_date, limit=limit)


def cache_news(ticker: str, end_date: str, limit: int, data: Any):
    """Cache news data (LRU eviction only)."""
    get_cache().set('news', data, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_news(ticker: str, end_date: str, limit: int) -> Optional[Any]:
    """Get cached news data."""
    return get_cache().get('news', ticker=ticker, end_date=end_date, limit=limit)


def cache_insider_trades(ticker: str, end_date: str, limit: int, data: Any):
    """Cache insider trades (LRU eviction only)."""
    get_cache().set('insider_trades', data, ticker=ticker, end_date=end_date, limit=limit)


def get_cached_insider_trades(ticker: str, end_date: str, limit: int) -> Optional[Any]:
    """Get cached insider trades."""
    return get_cache().get('insider_trades', ticker=ticker, end_date=end_date, limit=limit)

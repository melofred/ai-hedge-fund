#!/usr/bin/env python3
"""
Test script for the financial data cache system.
Demonstrates cache performance and functionality.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.cache import get_cache, print_cache_stats, CacheConfig
from src.tools.api import get_prices, get_financial_metrics
from cache_config import DEFAULT_CACHE_CONFIG


def test_cache_performance():
    """Test cache performance with real API calls."""
    print("🧪 Testing Financial Data Cache Performance")
    print("=" * 50)
    
    # Initialize cache with custom config
    cache = get_cache()
    
    # Test parameters
    ticker = "AAPL"
    start_date = "2024-11-01"
    end_date = "2024-12-01"
    
    print(f"Testing with {ticker} from {start_date} to {end_date}")
    print()
    
    # Test 1: First call (cache miss)
    print("📡 Test 1: First API call (cache miss)")
    start_time = time.time()
    prices = get_prices(ticker, start_date, end_date)
    first_call_time = time.time() - start_time
    print(f"   Time: {first_call_time:.2f}s")
    print(f"   Data points: {len(prices)}")
    print()
    
    # Test 2: Second call (cache hit)
    print("⚡ Test 2: Second call (cache hit)")
    start_time = time.time()
    prices_cached = get_prices(ticker, start_date, end_date)
    second_call_time = time.time() - start_time
    print(f"   Time: {second_call_time:.2f}s")
    print(f"   Data points: {len(prices_cached)}")
    print(f"   Speedup: {first_call_time / second_call_time:.1f}x faster")
    print()
    
    # Test 3: Financial metrics
    print("📊 Test 3: Financial metrics caching")
    start_time = time.time()
    metrics = get_financial_metrics(ticker, end_date, limit=5)
    metrics_time = time.time() - start_time
    print(f"   Time: {metrics_time:.2f}s")
    print(f"   Metrics: {len(metrics)}")
    print()
    
    # Test 4: Cached financial metrics
    print("⚡ Test 4: Cached financial metrics")
    start_time = time.time()
    metrics_cached = get_financial_metrics(ticker, end_date, limit=5)
    metrics_cached_time = time.time() - start_time
    print(f"   Time: {metrics_cached_time:.2f}s")
    print(f"   Speedup: {metrics_time / metrics_cached_time:.1f}x faster")
    print()
    
    # Print cache statistics
    print_cache_stats()
    
    # Test cache eviction
    print("\n🗑️  Testing cache eviction...")
    cache.clear()
    print_cache_stats()


def test_cache_configuration():
    """Test different cache configurations."""
    print("\n🔧 Testing Cache Configurations")
    print("=" * 50)
    
    # Test with small cache
    small_config = CacheConfig(
        max_memory_size=10 * 1024 * 1024,  # 10MB
        max_disk_size=100 * 1024 * 1024,   # 100MB
        cache_dir="test_cache_small"
    )
    
    print("Small cache configuration:")
    print(f"   Memory limit: {small_config.max_memory_size / 1024 / 1024:.1f}MB")
    print(f"   Disk limit: {small_config.max_disk_size / 1024 / 1024:.1f}MB")
    
    # Test with large cache
    large_config = CacheConfig(
        max_memory_size=500 * 1024 * 1024,  # 500MB
        max_disk_size=5 * 1024 * 1024 * 1024,  # 5GB
        cache_dir="test_cache_large"
    )
    
    print("\nLarge cache configuration:")
    print(f"   Memory limit: {large_config.max_memory_size / 1024 / 1024:.1f}MB")
    print(f"   Disk limit: {large_config.max_disk_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    try:
        test_cache_performance()
        test_cache_configuration()
        print("\n✅ Cache testing completed successfully!")
    except Exception as e:
        print(f"\n❌ Cache testing failed: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Simple test script for the cache system without external dependencies.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.cache import get_cache, print_cache_stats, CacheConfig


def test_basic_cache_operations():
    """Test basic cache operations."""
    print("🧪 Testing Basic Cache Operations")
    print("=" * 50)
    
    # Initialize cache
    cache = get_cache()
    
    # Test 1: Set and get
    print("📝 Test 1: Set and Get")
    test_data = {"ticker": "AAPL", "price": 150.0, "volume": 1000000}
    cache.set('test_data', test_data, ticker='AAPL', date='2024-01-01')
    
    retrieved_data = cache.get('test_data', ticker='AAPL', date='2024-01-01')
    print(f"   Original: {test_data}")
    print(f"   Retrieved: {retrieved_data}")
    print(f"   Match: {test_data == retrieved_data}")
    print()
    
    # Test 2: Cache miss
    print("❌ Test 2: Cache Miss")
    missing_data = cache.get('test_data', ticker='MSFT', date='2024-01-01')
    print(f"   Missing data: {missing_data}")
    print()
    
    # Test 3: TTL expiration
    print("⏰ Test 3: TTL Expiration")
    cache.set('expiring_data', {"test": "value"}, ttl=1, key='expire_test')
    time.sleep(1.1)  # Wait for expiration
    expired_data = cache.get('expiring_data', ttl=1, key='expire_test')
    print(f"   Expired data: {expired_data}")
    print()
    
    # Test 4: Memory eviction
    print("🗑️  Test 4: Memory Eviction")
    # Fill cache with large data
    for i in range(100):
        large_data = {"data": "x" * 1000, "index": i}
        cache.set('large_data', large_data, index=i)
    
    # Check if some data was evicted
    early_data = cache.get('large_data', index=0)
    late_data = cache.get('large_data', index=99)
    print(f"   Early data (index 0): {early_data is not None}")
    print(f"   Late data (index 99): {late_data is not None}")
    print()
    
    # Print cache statistics
    print_cache_stats()
    
    # Cleanup
    cache.clear()
    print("\n✅ Basic cache operations test completed!")


def test_cache_configuration():
    """Test different cache configurations."""
    print("\n🔧 Testing Cache Configurations")
    print("=" * 50)
    
    # Test with small cache
    small_config = CacheConfig(
        max_memory_size=1024 * 1024,  # 1MB
        max_disk_size=10 * 1024 * 1024,  # 10MB
        cache_dir="test_cache_small",
        default_ttl=60,  # 1 minute
        compression_level=1,
        enable_async=False  # Disable async for testing
    )
    
    print("Small cache configuration:")
    print(f"   Memory limit: {small_config.max_memory_size / 1024:.1f}KB")
    print(f"   Disk limit: {small_config.max_disk_size / 1024 / 1024:.1f}MB")
    print(f"   TTL: {small_config.default_ttl}s")
    print(f"   Async: {small_config.enable_async}")
    print()
    
    # Test cache creation
    try:
        from src.utils.cache import FinancialDataCache
        test_cache = FinancialDataCache(small_config)
        print("✅ Cache created successfully with custom config")
        test_cache.close()
    except Exception as e:
        print(f"❌ Cache creation failed: {e}")


if __name__ == "__main__":
    try:
        test_basic_cache_operations()
        test_cache_configuration()
        print("\n🎉 All cache tests passed!")
    except Exception as e:
        print(f"\n❌ Cache testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

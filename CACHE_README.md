# Financial Data Caching System

A high-performance, intelligent caching system for financial data APIs that dramatically reduces latency for local LLM inference.

## 🚀 Features

- **In-Memory + Disk Persistence**: Hot data in memory, cold data on disk
- **LRU Eviction**: Configurable size limits with least-recently-used eviction
- **TTL Support**: Time-to-live for different data types
- **Async Operations**: Non-blocking disk writes
- **Compression**: Built-in data compression to save space
- **Thread-Safe**: Works with concurrent operations
- **Performance Monitoring**: Built-in statistics and hit rate tracking

## 📊 Performance Benefits

- **10-100x faster** for cached data vs API calls
- **Reduced API costs** by minimizing redundant requests
- **Better user experience** with near-instant data access
- **Offline capability** for previously fetched data

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   In-Memory     │    │   Disk Cache    │    │   API Fallback  │
│   LRU Cache     │◄──►│   (SQLite)      │◄──►│   (Network)     │
│   (Fastest)     │    │   (Persistent)  │    │   (Slowest)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

The cache system uses `diskcache` which is already included in the project dependencies:

```bash
# Install dependencies (if not already done)
poetry install
```

## 📝 Usage

### Basic Usage

The cache is automatically integrated into all API functions:

```python
from src.tools.api import get_prices, get_financial_metrics

# First call - fetches from API and caches
prices = get_prices("AAPL", "2024-11-01", "2024-12-01")

# Second call - returns from cache (much faster!)
prices = get_prices("AAPL", "2024-11-01", "2024-12-01")
```

### Cache Configuration

Customize cache settings in `cache_config.py`:

```python
from src.utils.cache import CacheConfig

# Custom configuration
config = CacheConfig(
    max_memory_size=500 * 1024 * 1024,  # 500MB in-memory
    max_disk_size=5 * 1024 * 1024 * 1024,  # 5GB on disk
    cache_dir="my_cache",
    default_ttl=7200,  # 2 hours
    compression_level=2,  # Better compression
    enable_async=True
)
```

### Cache Statistics

Monitor cache performance:

```python
from src.utils.cache import print_cache_stats

# Print current statistics
print_cache_stats()
```

Output:
```
📊 Cache Statistics:
   Hit Rate: 85.2% (127/149)
   Memory: 45.2MB
   Disk: 234.1MB
   Evictions: 12
```

## ⚙️ Configuration Options

### Cache Size Presets

```python
from cache_config import CACHE_SIZE_PRESETS

# Small setup (50MB memory, 500MB disk)
small_config = CacheConfig(**CACHE_SIZE_PRESETS['small'])

# Medium setup (200MB memory, 2GB disk) - Default
medium_config = CacheConfig(**CACHE_SIZE_PRESETS['medium'])

# Large setup (500MB memory, 5GB disk)
large_config = CacheConfig(**CACHE_SIZE_PRESETS['large'])
```

### TTL Settings

Different data types have different TTL values:

```python
from cache_config import TTL_SETTINGS

# TTL in seconds
TTL_SETTINGS = {
    'prices': 1800,        # 30 minutes
    'financials': 3600,    # 1 hour
    'news': 1800,          # 30 minutes
    'insider_trades': 3600, # 1 hour
    'market_cap': 1800,    # 30 minutes
}
```

## 🧪 Testing

Run the cache test script to see performance improvements:

```bash
python test_cache.py
```

Expected output:
```
🧪 Testing Financial Data Cache Performance
==================================================
Testing with AAPL from 2024-11-01 to 2024-12-01

📡 Test 1: First API call (cache miss)
   Time: 2.34s
   Data points: 22

⚡ Test 2: Second call (cache hit)
   Time: 0.02s
   Data points: 22
   Speedup: 117.0x faster
```

## 📁 Cache Directory Structure

```
cache/
├── cache.db          # SQLite database
├── cache.db-shm      # Shared memory
└── cache.db-wal      # Write-ahead log
```

## 🔧 Advanced Usage

### Custom Cache Keys

```python
from src.utils.cache import get_cache

cache = get_cache()

# Custom cache operations
data = cache.get('custom_data', ttl=3600, param1='value1', param2='value2')
cache.set('custom_data', my_data, ttl=3600, param1='value1', param2='value2')
```

### Cache Decorator

```python
from src.utils.cache import cached

@cached('my_function', ttl=1800)
def expensive_computation(param1, param2):
    # Expensive operation here
    return result
```

### Manual Cache Management

```python
from src.utils.cache import get_cache

cache = get_cache()

# Clear all cache
cache.clear()

# Close cache (cleanup)
cache.close()

# Get detailed statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1f}%")
```

## 🎯 Best Practices

1. **Size Limits**: Set appropriate memory/disk limits based on your system
2. **TTL Values**: Use shorter TTL for frequently changing data (prices)
3. **Monitoring**: Regularly check cache hit rates and adjust settings
4. **Cleanup**: Periodically clear old cache data if disk space is limited

## 🐛 Troubleshooting

### High Memory Usage
- Reduce `max_memory_size` in configuration
- Check for memory leaks in long-running processes

### Low Hit Rate
- Increase TTL values for stable data
- Check if cache keys are being generated consistently

### Disk Space Issues
- Reduce `max_disk_size` limit
- Clear cache periodically: `cache.clear()`

### Performance Issues
- Enable compression: `compression_level=2`
- Use async operations: `enable_async=True`

## 📈 Performance Monitoring

The cache system provides detailed metrics:

- **Hit Rate**: Percentage of requests served from cache
- **Memory Usage**: Current in-memory cache size
- **Disk Usage**: Current disk cache size
- **Evictions**: Number of items evicted due to size limits

Monitor these metrics to optimize cache performance for your use case.

## 🔄 Integration with Backtester

The cache is automatically used by the backtester, providing significant speed improvements:

```bash
# First run - builds cache
python src/backtester.py --tickers AAPL,MSFT,NVDA --ollama

# Subsequent runs - uses cache for massive speedup
python src/backtester.py --tickers AAPL,MSFT,NVDA --ollama
```

## 🚀 Future Enhancements

- **Distributed Caching**: Redis integration for multi-instance setups
- **Smart Prefetching**: Predictive data loading based on usage patterns
- **Compression Algorithms**: Multiple compression options
- **Cache Warming**: Pre-populate cache with common data
- **Analytics Dashboard**: Web interface for cache monitoring

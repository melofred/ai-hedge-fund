"""
Cache configuration for the AI Hedge Fund application.
Customize cache settings here.
"""

from src.utils.cache import CacheConfig

# Default cache configuration
DEFAULT_CACHE_CONFIG = CacheConfig(
    max_memory_size=200 * 1024 * 1024,  # 200MB in-memory
    max_disk_size=2 * 1024 * 1024 * 1024,  # 2GB on disk
    cache_dir="cache",
    default_ttl=3600,  # 1 hour default TTL
    compression_level=1,  # Fast compression
    enable_async=True
)

# TTL settings for different data types (in seconds)
TTL_SETTINGS = {
    'prices': 1800,        # 30 minutes - prices change frequently
    'financials': 3600,    # 1 hour - financial data is more stable
    'news': 1800,          # 30 minutes - news is time-sensitive
    'insider_trades': 3600, # 1 hour - insider trades are less frequent
    'market_cap': 1800,    # 30 minutes - market cap changes with price
}

# Cache size recommendations based on usage patterns
CACHE_SIZE_PRESETS = {
    'small': {
        'max_memory_size': 50 * 1024 * 1024,   # 50MB
        'max_disk_size': 500 * 1024 * 1024,    # 500MB
    },
    'medium': {
        'max_memory_size': 200 * 1024 * 1024,  # 200MB
        'max_disk_size': 2 * 1024 * 1024 * 1024,  # 2GB
    },
    'large': {
        'max_memory_size': 500 * 1024 * 1024,  # 500MB
        'max_disk_size': 5 * 1024 * 1024 * 1024,  # 5GB
    }
}

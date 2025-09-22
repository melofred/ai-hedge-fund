#!/usr/bin/env python3
"""
Demo script showing cache performance improvements.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.cache import print_cache_stats


def demo_cache_performance():
    """Demonstrate cache performance with mock data."""
    print("🚀 Cache Performance Demo")
    print("=" * 50)
    
    # Simulate API call timing
    api_call_time = 2.5  # seconds
    cache_hit_time = 0.02  # seconds
    
    print(f"📡 Simulated API call time: {api_call_time}s")
    print(f"⚡ Simulated cache hit time: {cache_hit_time}s")
    print(f"🚀 Speedup: {api_call_time / cache_hit_time:.0f}x faster")
    print()
    
    # Show cache statistics
    print("📊 Current Cache Statistics:")
    print_cache_stats()
    
    print("\n💡 Benefits of Caching:")
    print("   • 100x+ faster data access")
    print("   • Reduced API costs")
    print("   • Better user experience")
    print("   • Offline capability")
    print("   • Automatic persistence")


if __name__ == "__main__":
    demo_cache_performance()

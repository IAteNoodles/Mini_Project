"""
Configuration file for the parallel news scraper.
Adjust these settings based on your system's capabilities.
"""

import os
import psutil

# === PARALLEL PROCESSING CONFIGURATION ===

# Number of parallel processes (cores) to use
# Recommended: number of CPU cores, but can be adjusted based on system performance
NUM_CORES = 16

# Maximum concurrent browser windows per process
# Lower values = more stable, higher values = faster (if system can handle it)
# Recommended: 2-4 for most systems
MAX_BROWSER_WINDOWS = 4

# === SCRAPING CONFIGURATION ===

# Minimum content length to consider an article valid
MIN_CONTENT_LENGTH = 200

# Sleep time between scraping cycles (in seconds)
# 900 = 15 minutes, 1800 = 30 minutes, 3600 = 1 hour
SLEEP_TIME_SECONDS = 900

# Browser timeout settings (in milliseconds)
BROWSER_TIMEOUT = 60000  # 60 seconds
PAGE_WAIT_TIME = 2000    # 2 seconds for dynamic content

# === MONGODB CONFIGURATION ===

# Batch size for MongoDB operations
MONGO_BATCH_SIZE = 100

# === PERFORMANCE OPTIMIZATION SETTINGS ===

# Browser launch arguments for performance and stability
BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage', 
    '--disable-gpu',
    '--disable-extensions',
    '--disable-plugins',
    '--disable-images',  # Disable image loading for faster scraping
    '--disable-javascript',  # Disable JS if not needed for content extraction
]

# User agent string to avoid being blocked
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# === AUTO-CONFIGURATION BASED ON SYSTEM ===

def auto_configure():
    """Automatically configure settings based on system capabilities."""
    
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    global NUM_CORES, MAX_BROWSER_WINDOWS
    
    # Adjust core count based on available CPUs
    if cpu_count < 8:
        NUM_CORES = max(2, cpu_count - 1)  # Leave one core for system
        print(f"🔧 Auto-configured NUM_CORES to {NUM_CORES} (system has {cpu_count} cores)")
    elif cpu_count < 16:
        NUM_CORES = cpu_count
        print(f"🔧 Auto-configured NUM_CORES to {NUM_CORES} (using all {cpu_count} cores)")
    # else keep NUM_CORES = 16 as configured
    
    # Adjust browser windows based on available memory
    if memory_gb < 4:
        MAX_BROWSER_WINDOWS = 1
        print(f"🔧 Auto-configured MAX_BROWSER_WINDOWS to {MAX_BROWSER_WINDOWS} (low memory: {memory_gb:.1f}GB)")
    elif memory_gb < 8:
        MAX_BROWSER_WINDOWS = 2
        print(f"🔧 Auto-configured MAX_BROWSER_WINDOWS to {MAX_BROWSER_WINDOWS} (moderate memory: {memory_gb:.1f}GB)")
    elif memory_gb < 16:
        MAX_BROWSER_WINDOWS = 3
        print(f"🔧 Auto-configured MAX_BROWSER_WINDOWS to {MAX_BROWSER_WINDOWS} (good memory: {memory_gb:.1f}GB)")
    # else keep MAX_BROWSER_WINDOWS = 4 as configured
    
    return {
        'NUM_CORES': NUM_CORES,
        'MAX_BROWSER_WINDOWS': MAX_BROWSER_WINDOWS,
        'CPU_COUNT': cpu_count,
        'MEMORY_GB': memory_gb
    }

# === CONFIGURATION VALIDATION ===

def validate_configuration():
    """Validate the current configuration and provide warnings if needed."""
    
    warnings = []
    
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check if NUM_CORES is reasonable
    if NUM_CORES > cpu_count * 2:
        warnings.append(f"⚠️  NUM_CORES ({NUM_CORES}) is much higher than available cores ({cpu_count})")
    
    # Check if browser windows might overwhelm the system
    total_browsers = NUM_CORES * MAX_BROWSER_WINDOWS
    if total_browsers > 50:
        warnings.append(f"⚠️  Total browser instances ({total_browsers}) might overwhelm the system")
    
    # Check memory requirements
    estimated_memory_per_browser = 0.1  # 100MB per browser instance
    estimated_total_memory = total_browsers * estimated_memory_per_browser
    if estimated_total_memory > memory_gb * 0.8:
        warnings.append(f"⚠️  Estimated memory usage ({estimated_total_memory:.1f}GB) might exceed available RAM ({memory_gb:.1f}GB)")
    
    return warnings

# === TOPICS CONFIGURATION ===

# You can modify this list to focus on specific topics or add new ones
# Leave empty to use ALL topics
PRIORITY_TOPICS = [
    # "Indian Politics", "US Politics", "Global Politics",
    # "Covid Response India", "Climate Change", "Technology News"
]

# Topics to exclude from scraping (if you want to skip certain topics)
EXCLUDED_TOPICS = [
    # "Religious Extremism",  # Uncomment to exclude sensitive topics
    # "Communal Violence",
]

def get_filtered_topics(all_topics):
    """Filter topics based on priority and exclusion lists."""
    
    # If priority topics are defined, use only those
    if PRIORITY_TOPICS:
        filtered = [topic for topic in all_topics if any(priority in topic for priority in PRIORITY_TOPICS)]
        if filtered:
            print(f"Using {len(filtered)} priority topics out of {len(all_topics)} total topics")
            return filtered
    
    # Otherwise, use all topics except excluded ones
    filtered = [topic for topic in all_topics if topic not in EXCLUDED_TOPICS]
    
    if len(filtered) != len(all_topics):
        print(f"Excluded {len(all_topics) - len(filtered)} topics, using {len(filtered)} topics")
    else:
        print(f"Using ALL {len(filtered)} available topics for scraping")
    
    return filtered

if __name__ == "__main__":
    print("🔧 PARALLEL SCRAPER CONFIGURATION")
    print("=" * 50)
    
    # Show current configuration
    print(f"NUM_CORES: {NUM_CORES}")
    print(f"MAX_BROWSER_WINDOWS: {MAX_BROWSER_WINDOWS}")
    print(f"SLEEP_TIME_SECONDS: {SLEEP_TIME_SECONDS}")
    print(f"MIN_CONTENT_LENGTH: {MIN_CONTENT_LENGTH}")
    
    print("\n" + "=" * 50)
    
    # Auto-configure and show results
    auto_config = auto_configure()
    print(f"\nAuto-configuration results:")
    for key, value in auto_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # Validate configuration
    warnings = validate_configuration()
    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("✅ Configuration looks good!")
    
    print("\n" + "=" * 50)
    print("To use auto-configuration, import this module in news_fetcher.py:")
    print("from config_parallel import auto_configure")
    print("auto_configure()")

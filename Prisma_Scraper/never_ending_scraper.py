#!/usr/bin/env python3
"""
Never-Ending Google News Scraper Launcher
Simple launcher for the unlimited parallel scraper that runs forever
"""

import sys
import logging
import signal
import os
from datetime import datetime

# Setup signal handler for graceful shutdown
def signal_handler(sig, frame):
    print('\n🛑 Graceful shutdown initiated...')
    logging.info("🛑 Scraper shutdown by user signal")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    print("🚀 NEVER-ENDING GOOGLE NEWS SCRAPER")
    print("=" * 50)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄 This will run FOREVER until you press Ctrl+C")
    print("💪 Using all available CPU cores for parallel processing")
    print("📊 Statistics will be logged after each complete cycle")
    print("=" * 50)
    
    # Import and run the unlimited scraper
    try:
        from unlimited_parallel_scraper import unlimited_parallel_main
        unlimited_parallel_main()
    except KeyboardInterrupt:
        print("\n🛑 Scraper stopped by user")
    except ImportError as e:
        print(f"❌ Error importing scraper: {e}")
        print("Make sure unlimited_parallel_scraper.py is in the same directory")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        logging.error(f"Unexpected error in launcher: {e}")

if __name__ == "__main__":
    main()

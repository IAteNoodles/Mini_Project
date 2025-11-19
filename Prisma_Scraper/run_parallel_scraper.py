#!/usr/bin/env python3
"""
Simple launcher for the parallel news scraper without emoji characters.
This version works reliably on Windows systems.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the parallel news scraper."""
    print("\n" + "="*60)
    print("PARALLEL NEWS SCRAPER - 16 CORE VERSION")
    print("="*60)
    print("Starting parallel news scraping with enhanced performance...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Import and run the news fetcher
        from news_fetcher import main as run_scraper
        run_scraper()
    except KeyboardInterrupt:
        print("\n\nScraping stopped by user.")
        print("Thank you for using the parallel news scraper!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Try running: python test_parallel_scraper.py")

if __name__ == "__main__":
    main()

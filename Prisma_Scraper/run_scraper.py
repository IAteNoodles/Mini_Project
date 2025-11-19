#!/usr/bin/env python3
"""
Simple test to run the Indian news scraper
"""

import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run the scraper with simple output"""
    
    print("🚀 STARTING INDIAN NEWS SCRAPER")
    print("=" * 50)
    
    try:
        # Import our fixed scraper
        from fixed_indian_scraper import scrape_all_sites, DatabaseManager
        
        print("✅ Scraper modules imported successfully")
        
        # Test database connection
        db_manager = DatabaseManager()
        print(f"✅ Database initialized (MongoDB: {db_manager.use_mongodb})")
        
        # Get existing count
        existing_urls = db_manager.get_existing_urls()
        print(f"📊 Found {len(existing_urls)} existing articles")
        
        # Run the scraper with a limit
        print("\n🔍 Starting article collection...")
        print("⏱️ This may take a few minutes...")
        
        new_articles = scrape_all_sites(max_articles_per_site=5)  # Limit for testing
        
        print(f"\n✅ Scraping completed!")
        print(f"📰 Collected {new_articles} new articles")
        
        # Get final count
        final_urls = db_manager.get_existing_urls()
        print(f"📊 Total articles in database: {len(final_urls)}")
        
        if new_articles > 0:
            print("🎉 SUCCESS: Scraper is working correctly!")
        else:
            print("⚠️ No new articles found - this might be normal if sites were recently scraped")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Scraping interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = main()
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n✅ Ready to run full scraper with: python fixed_indian_scraper.py")
    else:
        print("\n❌ Please check the errors above")
    
    exit(0 if success else 1)

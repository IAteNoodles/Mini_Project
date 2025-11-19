#!/usr/bin/env python3
"""
Test script for Google News scraper
"""

import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_google_scraper():
    """Test the Google News scraper components"""
    
    try:
        from scrape_google_news import GoogleNewsScraper, DatabaseManager
        
        print("🧪 TESTING GOOGLE NEWS SCRAPER")
        print("=" * 50)
        
        # Test database connection
        db_manager = DatabaseManager()
        print(f"✅ Database initialized (MongoDB: {db_manager.use_mongodb})")
        
        existing_urls = db_manager.get_existing_urls()
        print(f"📊 Found {len(existing_urls)} existing articles in database")
        
        # Test RSS feed
        scraper = GoogleNewsScraper()
        print(f"✅ Scraper initialized")
        
        # Test fetching RSS for a topic
        print(f"\n🔍 Testing RSS feed fetch...")
        articles = scraper.get_google_news_rss("technology", region="IN", language="en")
        print(f"📰 Found {len(articles)} articles in RSS feed")
        
        if articles:
            print(f"📄 Sample article:")
            sample = articles[0]
            print(f"   Title: {sample['title'][:80]}...")
            print(f"   Link: {sample['link']}")
            print(f"   Source: {sample['source']}")
        
        # Test URL resolution (just one)
        if articles:
            print(f"\n🔗 Testing URL resolution...")
            test_url = articles[0]['link']
            resolved = scraper.resolve_url(test_url)
            print(f"   Original: {test_url}")
            print(f"   Resolved: {resolved}")
            
            # Test content extraction
            if resolved and scraper.is_valid_news_url(resolved):
                print(f"\n📄 Testing content extraction...")
                content = scraper.extract_content(resolved)
                if content:
                    print(f"   ✅ Content extracted: {len(content)} characters")
                    print(f"   Preview: {content[:200]}...")
                else:
                    print(f"   ❌ No content extracted")
            else:
                print(f"   ⏭️ Invalid URL for content extraction")
        
        print(f"\n🎉 Basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_small_scrape():
    """Test scraping a small number of articles"""
    
    try:
        from scrape_google_news import GoogleNewsScraper
        
        print("\n🚀 TESTING SMALL SCRAPE")
        print("=" * 50)
        
        scraper = GoogleNewsScraper()
        
        # Test scraping just 3 articles for "India politics"
        print("🔍 Scraping 3 articles for 'India politics'...")
        count = scraper.scrape_topic("India politics", max_articles=3)
        
        print(f"✅ Successfully scraped {count} articles")
        
        # Show updated database count
        final_urls = scraper.db_manager.get_existing_urls()
        print(f"📊 Total articles now in database: {len(final_urls)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Small scrape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test basic components
    basic_success = test_google_scraper()
    
    if basic_success:
        # Test actual scraping
        scrape_success = test_small_scrape()
        
        if scrape_success:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ Google News scraper is working correctly")
            print(f"💡 Run full scraper with: python scrape_google_news.py")
        else:
            print(f"\n⚠️ Basic tests passed, but scraping test failed")
    else:
        print(f"\n❌ Basic tests failed - check configuration")
    
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

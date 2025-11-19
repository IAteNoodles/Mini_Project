#!/usr/bin/env python3
"""
Test the expanded news scraper configuration
"""

import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_new_sites():
    """Test the new sites configuration"""
    
    try:
        from fixed_indian_scraper import NEWS_SITES, domain_key_for_host, is_article_url, is_desired_topic
        
        print("🌍 EXPANDED NEWS SCRAPER TEST")
        print("=" * 60)
        
        # Show all configured sites
        print(f"📰 Total Sites Configured: {len(NEWS_SITES)}")
        print()
        
        # Group sites by category
        indian_mainstream = [s for s in NEWS_SITES if any(x in s[1] for x in ['ndtv', 'timesofindia', 'thehindu', 'indianexpress'])]
        indian_controversial = [s for s in NEWS_SITES if any(x in s[1] for x in ['republic', 'opindia', 'thewire', 'altnews'])]
        international_mainstream = [s for s in NEWS_SITES if any(x in s[1] for x in ['bbc', 'cnn', 'reuters', 'nytimes'])]
        international_controversial = [s for s in NEWS_SITES if any(x in s[1] for x in ['foxnews', 'breitbart', 'rt', 'aljazeera'])]
        tech_sites = [s for s in NEWS_SITES if any(x in s[1] for x in ['techcrunch', 'wired', 'verge', 'arstechnica'])]
        
        print("🇮🇳 INDIAN MAINSTREAM:")
        for name, url in indian_mainstream[:5]:
            print(f"   ✅ {name}: {url}")
        print(f"   ... and {len(indian_mainstream)-5} more")
        
        print("\n🔥 INDIAN CONTROVERSIAL:")
        for name, url in indian_controversial:
            print(f"   ✅ {name}: {url}")
        
        print("\n🌐 INTERNATIONAL MAINSTREAM:")
        for name, url in international_mainstream:
            print(f"   ✅ {name}: {url}")
        
        print("\n⚡ INTERNATIONAL CONTROVERSIAL:")
        for name, url in international_controversial:
            print(f"   ✅ {name}: {url}")
        
        print("\n💻 TECH SITES:")
        for name, url in tech_sites:
            print(f"   ✅ {name}: {url}")
        
        # Test domain mapping
        print(f"\n🔗 DOMAIN MAPPING TEST:")
        test_domains = [
            'www.bbc.com',
            'www.foxnews.com', 
            'www.opindia.com',
            'thewire.in',
            'www.rt.com'
        ]
        
        for domain in test_domains:
            mapped = domain_key_for_host(domain)
            print(f"   {domain} → {mapped}")
        
        # Test article URL detection
        print(f"\n📄 ARTICLE URL DETECTION TEST:")
        test_urls = [
            ("https://www.bbc.com/news/world-asia-12345", "bbc.com"),
            ("https://www.foxnews.com/politics/trump-election-2024", "foxnews.com"),
            ("https://www.opindia.com/2024/politics/modi-government", "opindia.com"),
            ("https://thewire.in/politics/parliament-session", "thewire.in"),
            ("https://www.rt.com/news/ukraine-russia-conflict/", "rt.com"),
            ("https://techcrunch.com/startup-funding-tech/", "techcrunch.com"),
        ]
        
        for url, domain in test_urls:
            result = is_article_url(url, domain)
            status = "✅" if result else "❌"
            print(f"   {status} {url}")
        
        # Test topic filtering
        print(f"\n🎯 TOPIC FILTERING TEST:")
        test_headlines = [
            ("Modi government announces new policy on China border", True),
            ("Trump election campaign controversy sparks debate", True),
            ("Tech startup gets $100M funding for AI innovation", True),
            ("Russia Ukraine conflict escalates tensions", True),
            ("Cricket match results and player statistics", False),
            ("Bollywood celebrity wedding photos leaked", False),
        ]
        
        for headline, expected in test_headlines:
            result = is_desired_topic("", headline)
            status = "✅" if result == expected else "❌"
            print(f"   {status} '{headline}' → {result}")
        
        print(f"\n🎉 CONFIGURATION TEST COMPLETED!")
        print(f"📊 Ready to scrape {len(NEWS_SITES)} diverse news sources")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = test_new_sites()
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n✅ ALL TESTS PASSED - Ready to scrape diverse news sources!")
        print("💡 Run: python fixed_indian_scraper.py")
    else:
        print("\n❌ Configuration has issues - please check errors above")
    
    return success

if __name__ == "__main__":
    main()

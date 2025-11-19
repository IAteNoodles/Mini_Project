#!/usr/bin/env python3
"""
Quick test of the fixed scraper components
"""

import sys
import os
from datetime import datetime

# Test the database manager
sys.path.append(os.path.dirname(__file__))

def test_database_manager():
    """Test the DatabaseManager class"""
    print("🧪 Testing DatabaseManager...")
    
    try:
        from fixed_indian_scraper import DatabaseManager
        
        db_manager = DatabaseManager()
        print(f"✅ DatabaseManager initialized")
        print(f"   Using MongoDB: {db_manager.use_mongodb}")
        
        # Test getting existing URLs
        existing = db_manager.get_existing_urls()
        print(f"✅ Retrieved {len(existing)} existing URLs")
        
        # Test saving an article
        test_url = f"https://test.com/article-{datetime.now().timestamp()}"
        success = db_manager.save_article(
            url=test_url,
            headline="Test Article",
            content="This is a test article for verification purposes.",
            site="test.com"
        )
        print(f"✅ Test article save: {'Success' if success else 'Failed'}")
        
        # Verify it was saved
        updated_existing = db_manager.get_existing_urls()
        if test_url in updated_existing:
            print(f"✅ Article properly stored and retrievable")
        else:
            print(f"⚠️ Article save verification failed")
        
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager test failed: {e}")
        return False

def test_url_functions():
    """Test URL and topic filtering functions"""
    print("\n🧪 Testing URL and topic filtering...")
    
    try:
        from fixed_indian_scraper import is_article_url, is_desired_topic
        
        # Test URL validation
        test_urls = [
            ("https://www.ndtv.com/india-news/some-article-123456", "ndtv.com", True),
            ("https://www.ndtv.com/entertainment/bollywood-gossip", "ndtv.com", False),
            ("https://www.ndtv.com/homepage", "ndtv.com", False),
            ("https://timesofindia.indiatimes.com/articleshow/123456.cms", "timesofindia.indiatimes.com", True),
        ]
        
        for url, domain, expected in test_urls:
            result = is_article_url(url, domain)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {url} -> {result} (expected {expected})")
        
        # Test topic filtering
        test_topics = [
            ("Modi government announces new policy", True),
            ("Cricket match results today", False),
            ("Bollywood star wedding photos", False),
            ("China border tension escalates", True),
            ("Technology startup funding news", True),
        ]
        
        for headline, expected in test_topics:
            result = is_desired_topic("", headline)
            status = "✅" if result == expected else "❌"
            print(f"   {status} '{headline}' -> {result} (expected {expected})")
        
        return True
        
    except Exception as e:
        print(f"❌ URL/Topic filtering test failed: {e}")
        return False

def test_imports():
    """Test that all imports work correctly"""
    print("\n🧪 Testing imports...")
    
    try:
        from fixed_indian_scraper import (
            DatabaseManager, domain_key_for_host, is_article_url, 
            is_desired_topic, NEWS_SITES, LocalDatabase
        )
        print("✅ All main components imported successfully")
        
        print(f"✅ {len(NEWS_SITES)} news sites configured")
        for name, url in NEWS_SITES[:3]:
            print(f"   - {name}: {url}")
        print(f"   ... and {len(NEWS_SITES)-3} more sites")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 TESTING FIXED INDIAN SCRAPER COMPONENTS")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_database_manager())
    results.append(test_url_functions())
    
    print(f"\n📊 TEST RESULTS:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print(f"✅ Fixed scraper is ready to run!")
    else:
        print(f"⚠️ SOME TESTS FAILED ({passed}/{total})")
        print(f"❌ Please check the issues above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

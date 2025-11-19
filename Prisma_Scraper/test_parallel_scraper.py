#!/usr/bin/env python3
"""
Test script for the parallel news fetcher to verify it works correctly.
"""

import os
import sys
from dotenv import load_dotenv
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def _test_function(x):
    """Test function for multiprocessing (needs to be at module level for Windows)."""
    time.sleep(0.1)  # Simulate work
    return x * x

def test_configuration():
    """Test if all required environment variables are set."""
    print("Testing configuration...")
    
    mongo_uri = os.getenv("MONGO_URI")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not mongo_uri or mongo_uri == "YOUR_MONGO_CONNECTION_STRING_HERE":
        print("MONGO_URI not properly configured")
        return False
    else:
        print("MONGO_URI configured")
    
    if not news_api_key:
        print("NEWS_API_KEY not found")
        return False
    else:
        print("NEWS_API_KEY configured")
    
    return True

def test_mongodb_connection():
    """Test MongoDB connection."""
    print("\nTesting MongoDB connection...")
    
    try:
        from news_fetcher import get_db_collection
        collection = get_db_collection()
        
        # Try to count documents
        count = collection.count_documents({})
        print(f"MongoDB connection successful. Found {count} existing articles.")
        return True
        
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

def test_news_api():
    """Test News API connection."""
    print("\nTesting News API...")
    
    try:
        import requests
        api_key = os.getenv("NEWS_API_KEY")
        url = f"https://newsapi.org/v2/everything?q=test&apiKey={api_key}&pageSize=1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok":
            print("News API connection successful")
            return True
        else:
            print(f"News API returned error: {data}")
            return False
            
    except Exception as e:
        print(f"News API connection failed: {e}")
        return False

def test_playwright():
    """Test Playwright installation."""
    print("\nTesting Playwright...")
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://example.com", timeout=30000)
            title = page.title()
            browser.close()
            
        print(f"Playwright working. Test page title: {title}")
        return True
        
    except Exception as e:
        print(f"Playwright test failed: {e}")
        return False

def test_parallel_processing():
    """Test that multiprocessing setup works."""
    print("\nTesting parallel processing setup...")
    
    try:
        from multiprocessing import Pool
        
        start_time = time.time()
        with Pool(processes=4) as pool:
            results = pool.map(_test_function, [1, 2, 3, 4])
        end_time = time.time()
        
        expected_results = [1, 4, 9, 16]
        if results == expected_results:
            print(f"Parallel processing working. Time taken: {end_time - start_time:.2f}s")
            return True
        else:
            print(f"Parallel processing returned wrong results: {results}")
            return False
            
    except Exception as e:
        print(f"Parallel processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Parallel News Fetcher Tests")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_mongodb_connection,
        test_news_api,
        test_playwright,
        test_parallel_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The parallel scraper should work correctly.")
        print("\nYou can now run: python news_fetcher.py")
    else:
        print("Some tests failed. Please fix the issues before running the scraper.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

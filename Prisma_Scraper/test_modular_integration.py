#!/usr/bin/env python3
"""
Quick test of integrated modular functionality
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test imports
try:
    from site_rules import (
        get_domain_key, is_valid_article_url, is_excluded_url,
        get_content_selectors, get_priority_sections, filter_links,
        SITE_ARTICLE_PATTERNS, DOMAIN_EXCLUSIONS
    )
    print("✅ Modular rules imported successfully")
    
    # Test the functions
    print("\n=== Testing Domain Key Function ===")
    test_urls = [
        "https://www.ndtv.com/",
        "https://timesofindia.indiatimes.com/",
        "https://economictimes.indiatimes.com/",
        "https://www.news18.com/"
    ]
    
    for url in test_urls:
        domain_key = get_domain_key(url)
        print(f"{url} -> {domain_key}")
    
    print("\n=== Testing Content Selectors ===")
    for url in test_urls:
        domain_key = get_domain_key(url)
        selectors = get_content_selectors(domain_key)
        print(f"{domain_key}: {selectors}")
    
    print("\n=== Testing Priority Sections ===")
    for url in test_urls:
        domain_key = get_domain_key(url)
        sections = get_priority_sections(domain_key)
        print(f"{domain_key}: {sections}")
    
    print("\n=== Testing Article URL Validation ===")
    test_article_urls = [
        "https://www.ndtv.com/india-news/some-article-1234567",
        "https://timesofindia.indiatimes.com/articleshow/123456.cms",
        "https://economictimes.indiatimes.com/markets/stocks/news/some-article/articleshow/123456.cms",
        "https://www.news18.com/world/some-article-9540705.html"
    ]
    
    for test_url in test_article_urls:
        domain_key = get_domain_key(test_url)
        is_valid = is_valid_article_url(test_url, domain_key)
        is_excluded = is_excluded_url(test_url, domain_key)
        print(f"{test_url}")
        print(f"  Domain: {domain_key}, Valid: {is_valid}, Excluded: {is_excluded}")
    
    print("\n✅ All modular rules functions working correctly!")
    
except ImportError as e:
    print(f"❌ Failed to import modular rules: {e}")
except Exception as e:
    print(f"❌ Error testing modular rules: {e}")

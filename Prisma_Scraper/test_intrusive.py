#!/usr/bin/env python3
"""
Quick test of intrusive rule-based search on working sites
"""
import os
import logging
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import patterns from main file
import sys
sys.path.append('.')
from indian_news_fetcher import (
    SITE_ARTICLE_PATTERNS, collect_links_for_site, domain_key_for_host,
    get_db_collection, get_existing_urls, PER_SITE_LIMIT
)

# Test just the working sites
WORKING_SITES = [
    ("Indian Express", "https://indianexpress.com/"),
    ("Business Standard", "https://www.business-standard.com/"),
    ("News18", "https://www.news18.com/"),
]

def main():
    collection = get_db_collection()
    existing = get_existing_urls(collection)
    logging.info(f"Existing URLs in DB: {len(existing)}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        
        for name, site_url in WORKING_SITES:
            allowed_domain = (urlparse(site_url).hostname or '').lower()
            
            logging.info(f"\n[{name}] Testing intrusive patterns on {site_url}")
            page = browser.new_page()
            
            try:
                page.goto(site_url, timeout=90000)
                page.wait_for_load_state('domcontentloaded', timeout=20000)
                time.sleep(2)
                
                candidates = collect_links_for_site(page, site_url, allowed_domain)
                logging.info(f"[{name}] Found {len(candidates)} candidates with intrusive patterns")
                
                # Show top candidates
                for i, c in enumerate(candidates[:5]):
                    logging.info(f"  {i+1}. Score {c.get('importance_score', 0)}: {c['headline'][:60]}...")
                
            except Exception as e:
                logging.error(f"[{name}] Error: {e}")
            finally:
                try:
                    page.close()
                except:
                    pass
        
        browser.close()

if __name__ == "__main__":
    main()

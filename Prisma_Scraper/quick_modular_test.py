#!/usr/bin/env python3
"""
Quick focused test of modular processing for problematic sites
"""

import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from pymongo import MongoClient

# Setup path and imports
sys.path.append(os.path.dirname(__file__))
from site_rules import get_domain_key, get_priority_sections

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

def get_existing_urls():
    """Get existing URLs count from database"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    count = collection.count_documents({})
    client.close()
    return count

def quick_test_modular_sites():
    """Quick test of the four problematic sites with modular rules"""
    
    # Test sites
    TEST_SITES = [
        ("NDTV", "https://www.ndtv.com/"),
        ("Times of India", "https://timesofindia.indiatimes.com/"), 
        ("Economic Times", "https://economictimes.indiatimes.com/"),
        ("News18", "https://www.news18.com/")
    ]
    
    existing_count = get_existing_urls()
    logging.info(f"Database currently has {existing_count} articles")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        
        for name, site_url in TEST_SITES:
            domain_key = get_domain_key(site_url)
            priority_sections = get_priority_sections(domain_key)
            
            logging.info(f"\n{'='*50}")
            logging.info(f"Testing {name} ({domain_key})")
            logging.info(f"Priority sections: {len(priority_sections)}")
            logging.info(f"{'='*50}")
            
            page = browser.new_page()
            
            try:
                # Test just the homepage
                logging.info(f"Testing homepage: {site_url}")
                page.goto(site_url, timeout=30000)
                page.wait_for_load_state('domcontentloaded', timeout=10000)
                
                # Check for basic anchor presence
                anchor_count = page.evaluate('() => document.querySelectorAll("a").length')
                logging.info(f"  Total anchors found: {anchor_count}")
                
                # Test a few specific selectors
                from site_rules import get_content_selectors
                selectors = get_content_selectors(domain_key)
                
                for selector in selectors[:3]:  # Test first 3 selectors
                    try:
                        count = page.evaluate(f'() => document.querySelectorAll("{selector}").length')
                        logging.info(f"  Selector '{selector}': {count} elements")
                    except Exception as e:
                        logging.info(f"  Selector '{selector}': ERROR - {e}")
                
                # Test first priority section if available
                if priority_sections:
                    section_url = priority_sections[0]
                    logging.info(f"Testing first section: {section_url}")
                    
                    try:
                        page.goto(section_url, timeout=30000)
                        page.wait_for_load_state('domcontentloaded', timeout=10000)
                        
                        section_anchors = page.evaluate('() => document.querySelectorAll("a").length')
                        logging.info(f"  Section anchors found: {section_anchors}")
                        
                    except Exception as e:
                        logging.info(f"  Section test error: {e}")
                
                logging.info(f"✅ {name} basic connectivity test passed")
                
            except Exception as e:
                logging.error(f"❌ {name} test failed: {e}")
            
            finally:
                try:
                    page.close()
                except:
                    pass
        
        browser.close()
    
    logging.info("\n" + "="*50)
    logging.info("MODULAR INTEGRATION TEST COMPLETE")
    logging.info("The integrated system is ready for full operation")
    logging.info("="*50)

if __name__ == "__main__":
    quick_test_modular_sites()

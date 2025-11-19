#!/usr/bin/env python3
"""
Specialized scraper for problematic Indian news sites
Focuses on NDTV, Times of India, Economic Times, News18

Uses modular rules and intelligent link extraction
"""

import os
import logging
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin
from pymongo import MongoClient

# Import our modular rules
from site_rules import (
    get_domain_key, is_valid_article_url, is_excluded_url,
    get_content_selectors, get_priority_sections, filter_links,
    SITE_ARTICLE_PATTERNS, DOMAIN_EXCLUSIONS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

# Target sites for this specialized scraper
TARGET_SITES = [
    ("NDTV", "https://www.ndtv.com/"),
    ("Times of India", "https://timesofindia.indiatimes.com/"),
    ("Economic Times", "https://economictimes.indiatimes.com/"),
    ("News18", "https://www.news18.com/"),
]

def get_db_collection():
    """Get MongoDB collection"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def get_existing_urls(collection):
    """Get set of existing URLs in database"""
    return set(doc["url"] for doc in collection.find({}, {"url": 1}))

def extract_all_links(page, base_url, domain_key):
    """Extract all links from page using intelligent selectors"""
    all_links = []
    
    # Get content-specific selectors for this domain
    selectors = get_content_selectors(domain_key)
    
    logging.info(f"Using selectors for {domain_key}: {selectors}")
    
    for selector in selectors:
        try:
            # Wait a bit for dynamic content
            page.wait_for_timeout(1000)
            
            # Get all matching elements
            elements = page.query_selector_all(selector)
            logging.info(f"  Selector '{selector}': {len(elements)} elements")
            
            for element in elements:
                try:
                    href = element.get_attribute('href')
                    text = element.inner_text().strip()
                    
                    if href:
                        # Convert relative URLs to absolute
                        full_url = urljoin(base_url, href)
                        
                        all_links.append({
                            'url': full_url,
                            'text': text,
                            'selector': selector
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logging.warning(f"Error with selector '{selector}': {e}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in all_links:
        if link['url'] not in seen:
            seen.add(link['url'])
            unique_links.append(link)
    
    logging.info(f"Total unique links found: {len(unique_links)}")
    return unique_links

def analyze_and_filter_links(links, domain_key, base_url):
    """Analyze links and apply intelligent filtering"""
    logging.info(f"\n=== ANALYZING LINKS FOR {domain_key.upper()} ===")
    
    # Step 1: Remove external domains
    internal_links = []
    for link in links:
        link_domain = get_domain_key(link['url'])
        if domain_key in link_domain or link_domain in domain_key:
            internal_links.append(link)
    
    logging.info(f"Step 1 - Internal links: {len(internal_links)}/{len(links)}")
    
    # Step 2: Apply domain exclusions
    non_excluded = []
    for link in internal_links:
        if not is_excluded_url(link['url'], domain_key):
            non_excluded.append(link)
    
    logging.info(f"Step 2 - After exclusions: {len(non_excluded)}/{len(internal_links)}")
    
    # Step 3: Apply article pattern matching
    article_links = []
    for link in non_excluded:
        if is_valid_article_url(link['url'], domain_key):
            article_links.append(link)
    
    logging.info(f"Step 3 - Valid articles: {len(article_links)}/{len(non_excluded)}")
    
    # Step 4: Show sample results
    logging.info(f"\n=== SAMPLE VALID ARTICLES (first 10) ===")
    for i, link in enumerate(article_links[:10]):
        logging.info(f"  {i+1}. {link['url']}")
        logging.info(f"      Text: {link['text'][:80]}...")
        logging.info(f"      Selector: {link['selector']}")
    
    return article_links

def scrape_site(browser, site_name, site_url, existing_urls):
    """Scrape a single site with intelligent link extraction"""
    domain_key = get_domain_key(site_url)
    logging.info(f"\n{'='*60}")
    logging.info(f"SCRAPING {site_name.upper()} ({domain_key})")
    logging.info(f"{'='*60}")
    
    all_candidates = []
    
    # List of URLs to visit (homepage + priority sections)
    urls_to_visit = [site_url] + get_priority_sections(domain_key)
    
    for i, url in enumerate(urls_to_visit):
        page_type = "Homepage" if i == 0 else f"Section {i}"
        logging.info(f"\n--- {page_type}: {url} ---")
        
        page = browser.new_page()
        try:
            # Navigate to page
            page.goto(url, timeout=60000)
            page.wait_for_load_state('domcontentloaded', timeout=15000)
            time.sleep(3)  # Wait for dynamic content
            
            # Extract all links
            links = extract_all_links(page, url, domain_key)
            
            # Analyze and filter
            filtered_links = analyze_and_filter_links(links, domain_key, url)
            
            all_candidates.extend(filtered_links)
            logging.info(f"{page_type} contributed {len(filtered_links)} candidates")
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
        finally:
            try:
                page.close()
            except:
                pass
    
    # Final deduplication
    seen = set()
    unique_candidates = []
    for candidate in all_candidates:
        if candidate['url'] not in seen:
            seen.add(candidate['url'])
            unique_candidates.append(candidate)
    
    logging.info(f"\n{site_name} SUMMARY:")
    logging.info(f"  Total unique candidates: {len(unique_candidates)}")
    
    # Filter against database
    new_candidates = [c for c in unique_candidates if c['url'] not in existing_urls]
    logging.info(f"  New candidates (not in DB): {len(new_candidates)}")
    
    return new_candidates

def main():
    """Main execution function"""
    logging.info("Starting specialized scraper for problematic Indian news sites")
    logging.info(f"Target sites: {[name for name, _ in TARGET_SITES]}")
    
    # Connect to database
    collection = get_db_collection()
    existing_urls = get_existing_urls(collection)
    logging.info(f"Existing URLs in database: {len(existing_urls)}")
    
    # Launch browser
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible for debugging
        
        results = {}
        
        for site_name, site_url in TARGET_SITES:
            try:
                candidates = scrape_site(browser, site_name, site_url, existing_urls)
                results[site_name] = candidates
                
                # Small delay between sites
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Failed to scrape {site_name}: {e}")
                results[site_name] = []
        
        browser.close()
    
    # Summary report
    logging.info(f"\n{'='*60}")
    logging.info("FINAL SUMMARY REPORT")
    logging.info(f"{'='*60}")
    
    total_found = 0
    for site_name, candidates in results.items():
        count = len(candidates)
        total_found += count
        logging.info(f"{site_name:20}: {count:4} new articles found")
        
        # Show top 3 from each site
        if candidates:
            logging.info(f"  Top articles from {site_name}:")
            for i, candidate in enumerate(candidates[:3]):
                logging.info(f"    {i+1}. {candidate['text'][:60]}...")
                logging.info(f"       {candidate['url']}")
    
    logging.info(f"\nTOTAL NEW ARTICLES FOUND: {total_found}")
    
    # Save results to file for analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"scraping_results_{timestamp}.json"
    
    # Convert to serializable format
    serializable_results = {}
    for site_name, candidates in results.items():
        serializable_results[site_name] = [
            {
                'url': c['url'],
                'text': c['text'],
                'selector': c['selector']
            }
            for c in candidates
        ]
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

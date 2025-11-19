#!/usr/bin/env python3
"""
Unlimited Parallel Google News Scraper - No Limits Edition
Utilizes all available cores for maximum scraping power
"""

import os
import time
import logging
import json
import feedparser
from urllib.parse import quote, urlparse
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import re
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from queue import Queue
import random

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# Content extraction
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

# Selenium imports - Firefox support
try:
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from dotenv import load_dotenv

# Enhanced logging setup
LOG_FILE = "google_news_scraper.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Disable excessive logs from other libraries
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('selenium').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('undetected_chromedriver').setLevel(logging.ERROR)
logging.getLogger('selenium.webdriver.remote.remote_connection').setLevel(logging.ERROR)
logging.getLogger('selenium.webdriver.common.service').setLevel(logging.ERROR)

# Suppress Selenium DevTools and Chrome logs
import os
os.environ['WDM_LOG'] = '0'

# ----------------- UNLIMITED CONFIG -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

# Enhanced configuration with retry settings
MAX_ARTICLES_PER_TOPIC = None  # UNLIMITED!
MIN_CONTENT_LENGTH = 100  # Lower threshold for more articles
REQUEST_TIMEOUT = 30  # Longer timeout for patience
GOOGLE_NEWS_WAIT_TIME = 15  # Even more patience for redirects

# Connection pool and retry settings
MAX_RETRIES = 3
RETRY_DELAY = (1, 5)  # Random delay between retries
CONNECTION_POOL_SIZE = 20  # Larger connection pool
MAX_REDIRECTS = 10

# Enhanced User Agent - Firefox
FIREFOX_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0"

# Parallel processing settings
MAX_WORKERS = min(16, multiprocessing.cpu_count())  # Use up to 16 cores
MAX_SELENIUM_INSTANCES = 8  # Multiple Chrome instances
TOPICS_PER_BATCH = 4  # Process topics in batches

# Delays for being respectful
ARTICLE_DELAY = (1, 3)  # Random delay between 1-3 seconds per article
TOPIC_DELAY = (2, 5)    # Random delay between 2-5 seconds per topic
BATCH_DELAY = (5, 10)   # Random delay between 5-10 seconds per batch

# Local storage
LOCAL_DB_FILE = "unlimited_google_news_backup.json"

# Thread-safe counters and cycle tracking with enhanced statistics
article_counter = threading.Lock()
cycle_stats_lock = threading.Lock()
error_stats_lock = threading.Lock()

# Global statistics
total_scraped = 0
total_saved = 0
total_failed = 0
current_cycle = 0
start_time = None

# Enhanced error tracking
error_stats = {
    'url_resolution_failed': 0,
    'content_extraction_failed': 0,
    'mongodb_save_failed': 0,
    'local_save_failed': 0,
    'http_403_errors': 0,
    'http_timeout_errors': 0,
    'connection_errors': 0,
    'selenium_errors': 0,
    'unknown_errors': 0
}

# Performance tracking
cycle_performance = []

def log_error_stat(error_type: str):
    """Thread-safe error statistics logging"""
    global error_stats
    with error_stats_lock:
        if error_type in error_stats:
            error_stats[error_type] += 1
        else:
            error_stats['unknown_errors'] += 1

def log_detailed_statistics():
    """Log comprehensive statistics dashboard"""
    global total_scraped, total_saved, total_failed, error_stats, start_time
    
    total_runtime = datetime.now() - start_time if start_time else None
    
    logging.info("\n" + "="*60)
    logging.info("📊 COMPREHENSIVE SCRAPER STATISTICS DASHBOARD")
    logging.info("="*60)
    
    # Basic metrics
    logging.info(f"📈 PERFORMANCE METRICS:")
    logging.info(f"   📄 Total articles processed: {total_scraped:,}")
    logging.info(f"   ✅ Successfully saved: {total_saved:,}")
    logging.info(f"   ❌ Failed: {total_failed:,}")
    
    if total_scraped > 0:
        success_rate = (total_saved / total_scraped) * 100
        logging.info(f"   📊 Success rate: {success_rate:.2f}%")
    
    if total_runtime:
        hours = total_runtime.total_seconds() / 3600
        if hours > 0:
            articles_per_hour = total_saved / hours
            logging.info(f"   ⏱️  Articles per hour: {articles_per_hour:.1f}")
            logging.info(f"   ⏰ Total runtime: {total_runtime}")
    
    # Error breakdown
    logging.info(f"\n🚨 ERROR BREAKDOWN:")
    for error_type, count in error_stats.items():
        if count > 0:
            percentage = (count / total_scraped) * 100 if total_scraped > 0 else 0
            logging.info(f"   {error_type.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    # System info
    logging.info(f"\n🖥️  SYSTEM CONFIGURATION:")
    logging.info(f"   Threads: {MAX_WORKERS}")
    logging.info(f"   Selenium instances: {MAX_SELENIUM_INSTANCES}")
    logging.info(f"   Connection pool size: {CONNECTION_POOL_SIZE}")
    logging.info(f"   Request timeout: {REQUEST_TIMEOUT}s")
    
    logging.info("="*60)

def log_cycle_summary(cycle_num: int, cycle_saved: int, cycle_duration, total_articles_db: int):
    """Log cycle completion summary"""
    global total_saved, start_time, cycle_performance
    
    # Track cycle performance
    cycle_info = {
        'cycle': cycle_num,
        'saved': cycle_saved,
        'duration': cycle_duration.total_seconds(),
        'timestamp': datetime.now()
    }
    cycle_performance.append(cycle_info)
    
    # Keep only last 10 cycles for performance tracking
    if len(cycle_performance) > 10:
        cycle_performance = cycle_performance[-10:]
    
    logging.info(f"\n🎉 CYCLE #{cycle_num} COMPLETED! 🎉")
    logging.info(f"📊 Cycle #{cycle_num} Summary:")
    logging.info(f"   ⏱️  Duration: {cycle_duration}")
    logging.info(f"   📄 Articles saved this cycle: {cycle_saved:,}")
    logging.info(f"   🗃️ Total database size: {total_articles_db:,}")
    logging.info(f"   ✅ Total saved across all cycles: {total_saved:,}")
    
    # Calculate average cycle performance
    if len(cycle_performance) >= 2:
        recent_cycles = cycle_performance[-5:]  # Last 5 cycles
        avg_duration = sum(c['duration'] for c in recent_cycles) / len(recent_cycles)
        avg_saved = sum(c['saved'] for c in recent_cycles) / len(recent_cycles)
        logging.info(f"   📈 Recent average: {avg_saved:.1f} articles/cycle, {avg_duration/60:.1f} min/cycle")
    
    if start_time:
        total_runtime = datetime.now() - start_time
        hours = total_runtime.total_seconds() / 3600
        if hours > 0:
            articles_per_hour = total_saved / hours
            logging.info(f"   ⚡ Current rate: {articles_per_hour:.1f} articles/hour")

class ThreadSafeLocalDatabase:
    """Thread-safe local JSON database"""
    
    def __init__(self, db_file=LOCAL_DB_FILE):
        self.db_file = db_file
        self.lock = threading.Lock()
        self.data = self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"articles": [], "metadata": {"created": datetime.now().isoformat()}}
        return {"articles": [], "metadata": {"created": datetime.now().isoformat()}}
    
    def _save_data(self):
        with self.lock:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_existing_urls(self):
        with self.lock:
            return {article['url'] for article in self.data['articles']}
    
    def save_article(self, url, headline, content, source="Google News"):
        with self.lock:
            article = {
                'url': url,
                'headline': headline,
                'content': content,
                'source': source,
                'scraped_at': datetime.now().isoformat(),
                'content_length': len(content)
            }
            self.data['articles'].append(article)
            self.data['metadata']['last_updated'] = datetime.now().isoformat()
            self.data['metadata']['total_articles'] = len(self.data['articles'])
            self._save_data()
            return True

class ThreadSafeDatabaseManager:
    """Thread-safe database manager"""
    
    def __init__(self):
        self.use_mongodb = False
        self.collection = None
        self.local_db = ThreadSafeLocalDatabase()
        self.lock = threading.Lock()
        
        if MONGO_AVAILABLE and MONGO_URI:
            try:
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                db = client[DB_NAME]
                self.collection = db[COLLECTION_NAME]
                self.use_mongodb = True
                logging.info("✅ Connected to MongoDB Atlas")
            except Exception as e:
                logging.warning(f"❌ MongoDB connection failed: {e}")
                logging.info("🔄 Falling back to local JSON storage")
        else:
            logging.info("📝 Using local JSON storage")
    
    def get_existing_urls(self) -> set:
        if self.use_mongodb and self.collection is not None:
            try:
                with self.lock:
                    urls = {doc['url'] for doc in self.collection.find({}, {"url": 1})}
                logging.info(f"📊 MongoDB: {len(urls)} existing URLs")
                return urls
            except Exception as e:
                logging.error(f"MongoDB read error: {e}")
                self.use_mongodb = False
        
        urls = self.local_db.get_existing_urls()
        logging.info(f"📊 Local DB: {len(urls)} existing URLs")
        return urls
    
    def save_article(self, url, headline, content, source="Google News"):
        success = False
        
        # Try MongoDB with retry logic
        if self.use_mongodb and self.collection is not None:
            for attempt in range(MAX_RETRIES):
                try:
                    with self.lock:
                        self.collection.replace_one(
                            {'url': url},
                            {
                                'url': url,
                                'headline': headline,
                                'content': content,
                                'source': source,
                                'scraped_at': datetime.now()
                            },
                            upsert=True
                        )
                    success = True
                    logging.debug(f"✅ Saved to MongoDB: {headline[:50]}...")
                    break
                except Exception as e:
                    logging.warning(f"🔄 MongoDB save attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                    if attempt == MAX_RETRIES - 1:
                        logging.error(f"❌ MongoDB save failed after {MAX_RETRIES} attempts: {e}")
                        self.use_mongodb = False
                        logging.info("🔄 Switching to local-only storage mode")
                    else:
                        time.sleep(random.uniform(*RETRY_DELAY))
        
        # Always save to local as backup/fallback
        try:
            self.local_db.save_article(url, headline, content, source)
            logging.debug(f"💾 Saved to local backup: {headline[:50]}...")
        except Exception as e:
            logging.error(f"❌ Local save failed: {e}")
            
        return success or True

class ParallelGoogleNewsScraper:
    """Unlimited parallel Google News scraper"""
    
    def __init__(self):
        self.db_manager = ThreadSafeDatabaseManager()
        self.existing_urls = self.db_manager.get_existing_urls()
        self.existing_urls_lock = threading.Lock()
        
        # Create multiple session pools for parallel requests with enhanced settings
        self.session_pool = Queue()
        for _ in range(MAX_WORKERS):
            session = requests.Session()
            
            # Enhanced adapter settings for better connection pooling
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=MAX_RETRIES,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
            
            adapter = HTTPAdapter(
                pool_connections=CONNECTION_POOL_SIZE,
                pool_maxsize=CONNECTION_POOL_SIZE,
                max_retries=retry_strategy,
                pool_block=False
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            session.headers.update({
                'User-Agent': FIREFOX_USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'DNT': '1',
                'Sec-GPC': '1'
            })
            self.session_pool.put(session)
        
        # Selenium semaphore for controlled instances
        self.selenium_semaphore = threading.Semaphore(MAX_SELENIUM_INSTANCES)
        
    def get_session(self):
        """Get a session from the pool"""
        return self.session_pool.get()
    
    def return_session(self, session):
        """Return session to the pool"""
        self.session_pool.put(session)
    
    def get_all_google_news_rss(self, topic: str = None, region: str = "IN", language: str = "en") -> List[Dict]:
        """Fetch ALL articles from Google News RSS feed - NO LIMITS"""
        try:
            if topic:
                encoded_topic = quote(topic)
                rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl={language}&gl={region}&ceid={region}:{language}"
                logging.info(f"🔍 Unlimited search for topic: '{topic}'")
            else:
                rss_url = f'https://news.google.com/rss?hl={language}&gl={region}&ceid={region}:{language}'
                logging.info(f"📰 Unlimited general news fetch")
            
            logging.info(f"RSS URL: {rss_url}")
            
            feed = feedparser.parse(rss_url)
            logging.info(f"📄 Found {len(feed.entries)} entries - SCRAPING ALL!")
            
            articles = []
            for entry in feed.entries:  # NO LIMITS - process ALL entries
                summary = entry.get('summary', 'No summary available')
                soup = BeautifulSoup(summary, 'html.parser')
                summary_text = soup.get_text().strip()
                
                article = {
                    'title': entry.get('title', 'No Title'),
                    'link': entry.get('link', ''),
                    'summary': summary_text,
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown Source')
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logging.error(f"Error fetching RSS: {e}")
            return []
    
    def extract_url_with_unlimited_patience(self, google_url: str) -> Optional[str]:
        """Extract URL with unlimited patience and multiple methods"""
        if not google_url or 'news.google.com' not in google_url:
            return google_url if self.is_valid_news_url(google_url) else None
        
        session = self.get_session()
        try:
            # Method 1: Multiple patient redirect attempts with enhanced error handling
            for attempt in range(MAX_RETRIES):
                try:
                    timeout = 20 + (attempt * 10)  # Progressive timeout
                    response = session.get(
                        google_url, 
                        allow_redirects=True, 
                        timeout=timeout,
                        verify=False  # Sometimes SSL issues cause problems
                    )
                    
                    # Check for success response codes
                    if response.status_code == 200:
                        final_url = response.url
                        if final_url != google_url and 'news.google.com' not in final_url:
                            if self.is_valid_news_url(final_url):
                                logging.debug(f"✅ Method 1 success: {final_url}")
                                return final_url
                    elif response.status_code == 403:
                        logging.warning(f"🚫 403 Forbidden for {google_url} - trying next method")
                        log_error_stat('http_403_errors')
                        break  # Move to next method immediately
                    elif response.status_code in [429, 503, 504]:
                        logging.warning(f"⏳ Rate limited ({response.status_code}) - waiting...")
                        wait_time = random.uniform(5, 15)
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"⚠️ HTTP {response.status_code} for {google_url}")
                    
                    if attempt < MAX_RETRIES - 1:
                        wait_time = 5 + (attempt * 2)
                        time.sleep(wait_time)
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
                    log_error_stat('http_timeout_errors')
                except requests.exceptions.ConnectionError as e:
                    logging.warning(f"🔌 Connection error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                    log_error_stat('connection_errors')
                except requests.exceptions.RequestException as e:
                    logging.warning(f"📡 Request error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                except Exception as e:
                    logging.error(f"💥 Unexpected error in Method 1: {e}")
                    log_error_stat('unknown_errors')
                    
                if attempt < MAX_RETRIES - 1:
                    time.sleep(random.uniform(*RETRY_DELAY))
            
            # Method 2: Firefox Selenium with enhanced error handling
            if SELENIUM_AVAILABLE:
                acquired = self.selenium_semaphore.acquire(timeout=30)
                if acquired:
                    driver = None
                    try:
                        # Try Firefox first
                        options = FirefoxOptions()
                        options.add_argument("--headless")
                        options.add_argument("--no-sandbox")
                        options.add_argument("--disable-dev-shm-usage")
                        options.add_argument("--disable-gpu")
                        options.add_argument("--window-size=1920,1080")
                        options.add_argument("--disable-extensions")
                        options.add_argument("--disable-notifications")
                        options.add_argument("--disable-logging")
                        options.add_argument("--silent")
                        options.set_preference("general.useragent.override", FIREFOX_USER_AGENT)
                        options.set_preference("dom.webdriver.enabled", False)
                        options.set_preference("useAutomationExtension", False)
                        
                        try:
                            driver = webdriver.Firefox(options=options)
                        except:
                            # Fallback to Chrome if Firefox fails
                            logging.debug("Firefox not available, falling back to Chrome")
                            chrome_options = ChromeOptions()
                            chrome_options.add_argument("--headless")
                            chrome_options.add_argument("--no-sandbox")
                            chrome_options.add_argument("--disable-dev-shm-usage")
                            chrome_options.add_argument("--disable-gpu")
                            chrome_options.add_argument("--window-size=1920,1080")
                            chrome_options.add_argument("--disable-extensions")
                            chrome_options.add_argument("--disable-notifications")
                            chrome_options.add_argument("--disable-logging")
                            chrome_options.add_argument("--silent")
                            chrome_options.add_argument(f"--user-agent={FIREFOX_USER_AGENT}")
                            chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
                            chrome_options.add_experimental_option('useAutomationExtension', False)
                            driver = webdriver.Chrome(options=chrome_options)
                        
                        driver.set_page_load_timeout(45)
                        driver.implicitly_wait(10)
                        
                        driver.get(google_url)
                        time.sleep(GOOGLE_NEWS_WAIT_TIME)
                        
                        final_url = driver.current_url
                        
                        if final_url != google_url and 'news.google.com' not in final_url:
                            if self.is_valid_news_url(final_url):
                                logging.debug(f"✅ Method 2 (Selenium) success: {final_url}")
                                return final_url
                                
                    except TimeoutException:
                        logging.warning(f"⏰ Selenium timeout for {google_url}")
                        log_error_stat('selenium_errors')
                    except WebDriverException as e:
                        logging.warning(f"🚗 Selenium WebDriver error: {e}")
                        log_error_stat('selenium_errors')
                    except Exception as e:
                        logging.error(f"💥 Selenium error: {e}")
                        log_error_stat('selenium_errors')
                    finally:
                        if driver:
                            try:
                                driver.quit()
                            except:
                                pass
                        self.selenium_semaphore.release()
            
            # Method 3: Header redirects with better error handling
            try:
                response = session.get(google_url, allow_redirects=False, timeout=15)
                if response.status_code in [301, 302, 303, 307, 308] and 'Location' in response.headers:
                    redirect_url = response.headers['Location']
                    if self.is_valid_news_url(redirect_url):
                        logging.debug(f"✅ Method 3 (Header redirect) success: {redirect_url}")
                        return redirect_url
            except Exception as e:
                logging.debug(f"Method 3 failed: {e}")
            
            # Method 4: Content parsing with enhanced selectors
            try:
                response = session.get(google_url, timeout=20)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    selectors = [
                        'a[href*="http"]:not([href*="google.com"])',
                        'a[data-n-tid]', 'article a', '.article a',
                        '[data-article-url]', 'a[target="_blank"]',
                        'a[href*="://"]', 'a[rel="nofollow"]',
                        '.g a', 'h3 a', '.r a'  # Additional Google-specific selectors
                    ]
                    
                    for selector in selectors:
                        links = soup.select(selector)
                        for link in links[:20]:
                            href = link.get('href') or link.get('data-article-url')
                            if href and self.is_valid_news_url(href):
                                logging.debug(f"✅ Method 4 (Content parsing) success: {href}")
                                return href
            except Exception as e:
                logging.debug(f"Method 4 failed: {e}")
            
            # Method 5: Enhanced base64 decoding
            try:
                if any(pattern in google_url for pattern in ['CBM', 'CAI', 'CCAiC', 'CCM']):
                    patterns = [r'CBM([^?&]+)', r'CAI([^?&]+)', r'CCAiC([^?&]+)', r'CCM([^?&]+)']
                    for pattern in patterns:
                        matches = re.findall(pattern, google_url)
                        for match in matches:
                            try:
                                padded = match + '=' * (4 - len(match) % 4)
                                decoded = base64.b64decode(padded, validate=True)
                                decoded_str = decoded.decode('utf-8', errors='ignore')
                                
                                url_patterns = [
                                    r'https?://[^\s<>"\']+[a-zA-Z0-9/]',
                                    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                                ]
                                
                                for url_pattern in url_patterns:
                                    urls = re.findall(url_pattern, decoded_str)
                                    for url in urls:
                                        if self.is_valid_news_url(url):
                                            logging.debug(f"✅ Method 5 (Base64) success: {url}")
                                            return url
                            except Exception as decode_error:
                                logging.debug(f"Base64 decode failed: {decode_error}")
                                continue
            except Exception as e:
                logging.debug(f"Method 5 failed: {e}")
            
            logging.warning(f"❌ All methods failed for: {google_url}")
            return None
            
        finally:
            self.return_session(session)
    
    def is_valid_news_url(self, url: str) -> bool:
        """Enhanced URL validation"""
        if not url or not url.startswith('http'):
            return False
        
        skip_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'pinterest.com', 'reddit.com',
            'news.google.com', 'google.co', 'googlenews.com', 'tiktok.com'
        ]
        
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        
        if any(skip in domain.lower() for skip in skip_domains):
            return False
        
        if not domain or '.' not in domain:
            return False
        
        path = parsed.path.lower()
        news_indicators = [
            '/news/', '/article/', '/story/', '/post/', '/politics/',
            '/world/', '/business/', '/tech/', '/opinion/', '/breaking/',
            'news', 'article', 'story', 'report', '/2024/', '/2025/',
            '/sports/', '/entertainment/', '/health/', '/science/'
        ]
        
        has_news_pattern = any(indicator in path or indicator in domain.lower() for indicator in news_indicators)
        
        news_domains = [
            'bbc', 'cnn', 'reuters', 'ap', 'guardian', 'nytimes', 'washingtonpost',
            'wsj', 'bloomberg', 'ndtv', 'timesofindia', 'thehindu', 'indianexpress',
            'hindustantimes', 'livemint', 'economictimes', 'firstpost', 'scroll',
            'thewire', 'opindia', 'aljazeera', 'rt', 'foxnews', 'breitbart',
            'reuters', 'politico', 'newsweek', 'forbes', 'cnbc', 'abcnews',
            'indiatoday', 'news18', 'republicworld', 'zeenews', 'aajtak'
        ]
        
        is_news_domain = any(news_domain in domain.lower() for news_domain in news_domains)
        
        return has_news_pattern or is_news_domain
    
    def extract_content_unlimited(self, url: str) -> Optional[str]:
        """Extract content with aggressive fallback chain: BS4 → Trafilatura → Selenium"""
        session = self.get_session()
        content = None
        
        try:
            logging.debug(f"🔍 Starting content extraction for: {url}")
            
            # Method 1: BeautifulSoup with enhanced extraction
            logging.debug(f"📄 Method 1: BeautifulSoup extraction for {url}")
            for attempt in range(MAX_RETRIES):
                try:
                    headers = {
                        'User-Agent': FIREFOX_USER_AGENT,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Referer': 'https://news.google.com/',
                        'DNT': '1',
                        'Sec-GPC': '1'
                    }
                    
                    response = session.get(
                        url, 
                        timeout=REQUEST_TIMEOUT,
                        headers=headers,
                        allow_redirects=True,
                        verify=False
                    )
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Remove unwanted elements
                        for element in soup(["script", "style", "nav", "footer", "header", "aside", 
                                           "menu", "form", "noscript", "iframe", "object", "embed"]):
                            element.decompose()
                        
                        # Enhanced content selectors - most specific first
                        content_selectors = [
                            # Main content containers
                            'article div[data-module="ArticleBody"]',
                            'article div[class*="article-body"]',
                            'article div[class*="story-body"]',
                            'article div[class*="content-body"]',
                            'div[class*="article-content"]',
                            'div[class*="story-content"]',
                            'div[class*="post-content"]',
                            'div[class*="entry-content"]',
                            'div[class*="news-content"]',
                            'div[class*="article-text"]',
                            'div[class*="story-text"]',
                            'div[class*="post-text"]',
                            
                            # Semantic elements
                            'article', '[role="main"]', 'main',
                            
                            # Content sections
                            '.content', '.text', '.description',
                            '.field-name-body', '.node-content',
                            '.article-wrap', '.story-wrap', '.post-wrap',
                            
                            # ID-based selectors
                            '#content', '#main-content', '#article-content',
                            '#story-content', '#post-content',
                            
                            # Class-based selectors
                            'div[class*="content"]', 'div[class*="article"]',
                            'div[class*="story"]', 'div[class*="text"]',
                            
                            # Paragraph containers
                            '.article p', '.story p', '.content p',
                            'article p', 'main p'
                        ]
                        
                        content = ""
                        for selector in content_selectors:
                            try:
                                elements = soup.select(selector)
                                if elements:
                                    # Try the first element
                                    element_text = elements[0].get_text(separator=' ', strip=True)
                                    if len(element_text) >= MIN_CONTENT_LENGTH:
                                        content = element_text
                                        logging.debug(f"✅ BS4 selector '{selector}' found {len(content)} chars")
                                        break
                            except:
                                continue
                        
                        # Fallback: All paragraphs
                        if not content or len(content) < MIN_CONTENT_LENGTH:
                            paragraphs = soup.find_all('p')
                            if paragraphs:
                                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                                logging.debug(f"✅ BS4 paragraph fallback: {len(content)} chars")
                        
                        # Last resort: full text
                        if not content or len(content) < MIN_CONTENT_LENGTH:
                            content = soup.get_text(separator=' ', strip=True)
                            logging.debug(f"✅ BS4 full text fallback: {len(content)} chars")
                        
                        if content and len(content) >= MIN_CONTENT_LENGTH:
                            logging.info(f"✅ BS4 extracted {len(content)} chars from {url}")
                            return content
                        else:
                            logging.debug(f"⚠️ BS4 content too short ({len(content) if content else 0} chars)")
                            
                    elif response.status_code == 403:
                        logging.warning(f"🚫 403 Forbidden for BS4 extraction: {url}")
                        log_error_stat('http_403_errors')
                        break  # Don't retry 403 errors
                    elif response.status_code in [429, 503, 504]:
                        logging.warning(f"⏳ Rate limited ({response.status_code}) for BS4: {url}")
                        wait_time = random.uniform(5, 15) * (attempt + 1)
                        time.sleep(wait_time)
                    else:
                        logging.warning(f"⚠️ HTTP {response.status_code} for BS4: {url}")
                        
                except requests.exceptions.Timeout:
                    logging.warning(f"⏰ BS4 timeout on attempt {attempt + 1}/{MAX_RETRIES}: {url}")
                    log_error_stat('http_timeout_errors')
                except requests.exceptions.ConnectionError:
                    logging.warning(f"🔌 BS4 connection error on attempt {attempt + 1}/{MAX_RETRIES}: {url}")
                    log_error_stat('connection_errors')
                except Exception as e:
                    logging.warning(f"� BS4 error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    wait_time = random.uniform(*RETRY_DELAY) * (attempt + 1)
                    time.sleep(wait_time)
            
            # Method 2: Trafilatura (if BS4 failed)
            if not content or len(content) < MIN_CONTENT_LENGTH:
                logging.debug(f"📰 Method 2: Trafilatura extraction for {url}")
                if TRAFILATURA_AVAILABLE:
                    try:
                        downloaded = trafilatura.fetch_url(url)
                        if downloaded:
                            trafilatura_content = trafilatura.extract(
                                downloaded, 
                                include_comments=False,
                                include_tables=True,
                                include_formatting=True,
                                favor_precision=True
                            )
                            if trafilatura_content and len(trafilatura_content) >= MIN_CONTENT_LENGTH:
                                logging.info(f"✅ Trafilatura extracted {len(trafilatura_content)} chars from {url}")
                                return trafilatura_content
                            else:
                                logging.debug(f"⚠️ Trafilatura content too short ({len(trafilatura_content) if trafilatura_content else 0} chars)")
                    except Exception as e:
                        logging.debug(f"Trafilatura failed for {url}: {e}")
            
            # Method 3: Selenium (last resort)
            if not content or len(content) < MIN_CONTENT_LENGTH:
                logging.debug(f"� Method 3: Selenium extraction for {url}")
                if SELENIUM_AVAILABLE:
                    acquired = self.selenium_semaphore.acquire(timeout=30)
                    if acquired:
                        driver = None
                        try:
                            # Try Firefox first
                            options = FirefoxOptions()
                            options.add_argument("--headless")
                            options.add_argument("--no-sandbox")
                            options.add_argument("--disable-dev-shm-usage")
                            options.add_argument("--disable-gpu")
                            options.add_argument("--window-size=1920,1080")
                            options.set_preference("general.useragent.override", FIREFOX_USER_AGENT)
                            
                            try:
                                driver = webdriver.Firefox(options=options)
                            except:
                                # Fallback to Chrome
                                chrome_options = ChromeOptions()
                                chrome_options.add_argument("--headless")
                                chrome_options.add_argument("--no-sandbox")
                                chrome_options.add_argument("--disable-dev-shm-usage")
                                chrome_options.add_argument("--disable-gpu")
                                chrome_options.add_argument(f"--user-agent={FIREFOX_USER_AGENT}")
                                driver = webdriver.Chrome(options=chrome_options)
                            
                            driver.set_page_load_timeout(30)
                            driver.get(url)
                            time.sleep(5)  # Wait for dynamic content
                            
                            # Extract content from rendered page
                            page_source = driver.page_source
                            soup = BeautifulSoup(page_source, 'html.parser')
                            
                            # Remove unwanted elements
                            for element in soup(["script", "style", "nav", "footer", "header", "aside", "menu"]):
                                element.decompose()
                            
                            # Try same selectors as BS4
                            selenium_content = ""
                            content_selectors = [
                                'article', '[role="main"]', '.content', '.article-content',
                                '.story-content', '.post-content', 'main', '.entry-content',
                                '.article-body', '.story-body', '.post-body', '.news-content'
                            ]
                            
                            for selector in content_selectors:
                                elements = soup.select(selector)
                                if elements:
                                    selenium_content = elements[0].get_text(strip=True)
                                    if len(selenium_content) >= MIN_CONTENT_LENGTH:
                                        break
                            
                            if not selenium_content or len(selenium_content) < MIN_CONTENT_LENGTH:
                                selenium_content = soup.get_text(strip=True)
                            
                            if selenium_content and len(selenium_content) >= MIN_CONTENT_LENGTH:
                                logging.info(f"✅ Selenium extracted {len(selenium_content)} chars from {url}")
                                return selenium_content
                            else:
                                logging.debug(f"⚠️ Selenium content too short ({len(selenium_content) if selenium_content else 0} chars)")
                                
                        except Exception as e:
                            logging.warning(f"💥 Selenium content extraction error: {e}")
                            log_error_stat('selenium_errors')
                        finally:
                            if driver:
                                try:
                                    driver.quit()
                                except:
                                    pass
                            self.selenium_semaphore.release()
            
            logging.warning(f"❌ All content extraction methods failed for: {url}")
            log_error_stat('content_extraction_failed')
            return None
            
        finally:
            self.return_session(session)
    
    def process_single_article(self, article_data):
        """Process a single article - designed for parallel execution"""
        global total_scraped, total_saved, total_failed, current_cycle
        
        article, topic, thread_id = article_data
        google_url = article['link']
        title = article['title']
        
        with article_counter:
            total_scraped += 1
            current_count = total_scraped
        
        logging.info(f"[C{current_cycle}-{thread_id}] 📄 Processing #{current_count}: {title[:50]}...")
        
        # Resolve URL with unlimited patience
        final_url = None
        for attempt in range(3):  # Multiple attempts
            final_url = self.extract_url_with_unlimited_patience(google_url)
            if final_url:
                break
            elif attempt < 2:
                delay = random.uniform(3, 7)
                time.sleep(delay)
        
        if not final_url:
            with article_counter:
                total_failed += 1
            log_error_stat('url_resolution_failed')
            logging.warning(f"[C{current_cycle}-{thread_id}] ⏭️ Could not resolve URL: {google_url}")
            return False
        
        # Check if already exists
        with self.existing_urls_lock:
            if final_url in self.existing_urls:
                logging.info(f"[C{current_cycle}-{thread_id}] ⏭️ Article already exists: {final_url}")
                return False
        
        # Extract content with enhanced fallback chain
        content = None
        logging.debug(f"[C{current_cycle}-{thread_id}] 🔍 Starting content extraction for: {final_url}")
        
        # Single attempt - the extract_content_unlimited method has its own retry logic
        content = self.extract_content_unlimited(final_url)
        
        if not content or len(content) < MIN_CONTENT_LENGTH:
            with article_counter:
                total_failed += 1
            log_error_stat('content_extraction_failed')
            logging.warning(f"[C{current_cycle}-{thread_id}] ⏭️ No content extracted after all methods: {final_url}")
            return False
        
        # Save article
        source_name = f"Google News - {topic}" if topic else "Google News"
        success = self.db_manager.save_article(
            url=final_url,
            headline=title,
            content=content,
            source=source_name
        )
        
        if success:
            with article_counter:
                total_saved += 1
            with self.existing_urls_lock:
                self.existing_urls.add(final_url)
            logging.info(f"[C{current_cycle}-{thread_id}] ✅ Saved: {title[:50]}...")
            
            # Random delay to be respectful
            delay = random.uniform(*ARTICLE_DELAY)
            time.sleep(delay)
            return True
        else:
            with article_counter:
                total_failed += 1
            logging.warning(f"[C{current_cycle}-{thread_id}] ❌ Failed to save: {final_url}")
            return False
    
    def scrape_topic_unlimited(self, topic: str = None) -> int:
        """Scrape ALL articles for a topic using parallel processing"""
        global current_cycle
        
        topic_name = topic or "General News"
        logging.info(f"🚀 C{current_cycle} UNLIMITED scrape starting for: '{topic_name}'")
        
        # Get ALL RSS articles
        rss_articles = self.get_all_google_news_rss(topic)
        if not rss_articles:
            logging.warning(f"C{current_cycle} No articles found for: '{topic_name}'")
            return 0
        
        logging.info(f"🎯 C{current_cycle} Processing ALL {len(rss_articles)} articles for '{topic_name}'")
        
        # Prepare article data for parallel processing
        article_data_list = []
        for i, article in enumerate(rss_articles):
            article_data_list.append((article, topic, f"{topic_name[:3]}-{i+1}"))
        
        saved_count = 0
        
        # Process articles in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_article = {
                executor.submit(self.process_single_article, article_data): article_data
                for article_data in article_data_list
            }
            
            for future in as_completed(future_to_article):
                try:
                    result = future.result()
                    if result:
                        saved_count += 1
                except Exception as e:
                    logging.error(f"C{current_cycle} Article processing error: {e}")
        
        logging.info(f"🎉 C{current_cycle} Topic '{topic_name}' completed: {saved_count} articles saved from {len(rss_articles)} total")
        
        # Topic delay
        delay = random.uniform(*TOPIC_DELAY)
        time.sleep(delay)
        
        return saved_count

def unlimited_parallel_main():
    """NEVER-ENDING LOOP - Main unlimited scraping function with parallel processing"""
    global total_scraped, total_saved, total_failed
    
    logging.info("🚀🚀🚀 STARTING NEVER-ENDING UNLIMITED PARALLEL GOOGLE NEWS SCRAPER 🚀🚀🚀")
    logging.info(f"� ENHANCED VERSION: Firefox User Agent + Aggressive Content Extraction")
    logging.info(f"�💪 Using {MAX_WORKERS} threads with {MAX_SELENIUM_INSTANCES} Selenium instances")
    logging.info(f"🦊 User Agent: {FIREFOX_USER_AGENT}")
    logging.info(f"📄 Content extraction order: BeautifulSoup → Trafilatura → Selenium")
    logging.info("🔄 THIS WILL RUN FOREVER - Press Ctrl+C to stop")
    
    scraper = ParallelGoogleNewsScraper()
    
    # UNLIMITED TOPIC LIST - Add as many as you want!
    all_topics = [
        None,  # General news
        "politics", "international news", "business", "technology", "india news",
        "world news", "economy", "government policy", "china", "usa", "russia",
        "ukraine", "climate change", "artificial intelligence", "sports", "cricket",
        "football", "entertainment", "bollywood", "hollywood", "health", "covid",
        "science", "space", "research", "education", "finance", "stock market",
        "cryptocurrency", "blockchain", "startups", "investing", "real estate",
        "automobiles", "electric vehicles", "renewable energy", "oil prices",
        "agriculture", "food", "travel", "tourism", "aviation", "railways",
        "infrastructure", "urban development", "environment", "pollution",
        "wildlife", "conservation", "human rights", "law", "supreme court",
        "elections", "parliament", "defense", "military", "terrorism",
        "security", "cyber security", "data privacy", "social media",
        "internet", "mobile phones", "gadgets", "apps", "gaming",
        "fashion", "lifestyle", "culture", "religion", "festivals",
        "literature", "books", "art", "music", "movies", "television",
        "streaming", "photography", "design", "architecture", "history",
        "archaeology", "philosophy", "psychology", "sociology", "economics",
        "mathematics", "physics", "chemistry", "biology", "medicine",
        "pharmacy", "biotechnology", "genetics", "neuroscience", "astronomy",
        "geology", "weather", "natural disasters", "earthquakes", "floods",
        "drought", "climate science", "oceanography", "marine biology"
    ]
    
    logging.info(f"🎯 Processing {len(all_topics)} topics with UNLIMITED scraping - FOREVER!")
    
    cycle_count = 0
    total_cycle_saved = 0
    
    # NEVER-ENDING LOOP
    while True:
        try:
            cycle_count += 1
            current_cycle = cycle_count  # Update global cycle counter
            cycle_start_time = datetime.now()
            cycle_saved = 0
            
            # Set global start time on first cycle
            global start_time
            if start_time is None:
                start_time = cycle_start_time
            
            logging.info(f"\n🔄🔄🔄 STARTING CYCLE #{cycle_count} 🔄🔄🔄")
            logging.info(f"⏰ Cycle started at: {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Refresh existing URLs count for each cycle
            scraper.existing_urls = scraper.db_manager.get_existing_urls()
            
            # Process topics in batches for better resource management
            for batch_start in range(0, len(all_topics), TOPICS_PER_BATCH):
                batch_end = min(batch_start + TOPICS_PER_BATCH, len(all_topics))
                batch_topics = all_topics[batch_start:batch_end]
                
                logging.info(f"\n🔥 CYCLE #{cycle_count} - BATCH {batch_start//TOPICS_PER_BATCH + 1}: Processing topics {batch_start+1}-{batch_end}")
                
                # Process batch topics in parallel
                with ThreadPoolExecutor(max_workers=min(len(batch_topics), MAX_WORKERS//2)) as executor:
                    future_to_topic = {
                        executor.submit(scraper.scrape_topic_unlimited, topic): topic
                        for topic in batch_topics
                    }
                    
                    for future in as_completed(future_to_topic):
                        topic = future_to_topic[future]
                        try:
                            saved_count = future.result()
                            cycle_saved += saved_count
                            logging.info(f"✅ Cycle #{cycle_count} - Topic '{topic}' completed: {saved_count} articles")
                        except Exception as e:
                            logging.error(f"❌ Cycle #{cycle_count} - Error processing topic '{topic}': {e}")
                
                # Batch delay for server respect
                if batch_end < len(all_topics):
                    delay = random.uniform(*BATCH_DELAY)
                    logging.info(f"😴 Cycle #{cycle_count} - Batch complete, resting for {delay:.1f} seconds...")
                    time.sleep(delay)
            
            # Cycle completion statistics with enhanced logging
            total_cycle_saved += cycle_saved
            cycle_end_time = datetime.now()
            cycle_duration = cycle_end_time - cycle_start_time
            total_duration = cycle_end_time - start_time
            
            existing_urls = scraper.db_manager.get_existing_urls()
            
            # Use enhanced logging function
            log_cycle_summary(cycle_count, cycle_saved, cycle_duration, len(existing_urls))
            
            # Log detailed statistics every 5 cycles
            if cycle_count % 5 == 0:
                log_detailed_statistics()
            
            # Rest between cycles (longer delay to let news refresh)
            cycle_rest_delay = random.uniform(60, 120)  # 1-2 minutes between full cycles
            logging.info(f"😴 Resting {cycle_rest_delay:.1f} seconds before starting cycle #{cycle_count + 1}...")
            time.sleep(cycle_rest_delay)
            
        except KeyboardInterrupt:
            logging.info(f"\n🛑 NEVER-ENDING LOOP INTERRUPTED BY USER!")
            logging.info(f"📊 Final Statistics After {cycle_count} Cycles:")
            logging.info(f"   📄 Total articles processed: {total_scraped}")
            logging.info(f"   ✅ Total articles saved: {total_saved}")
            logging.info(f"   ❌ Total failed: {total_failed}")
            logging.info(f"   📈 Success rate: {(total_saved/total_scraped*100) if total_scraped > 0 else 0:.1f}%")
            logging.info(f"   🗃️ Final database size: {len(scraper.existing_urls)}")
            logging.info(f"   🔄 Cycles completed: {cycle_count}")
            break
        except Exception as e:
            logging.error(f"💥 Unexpected error in cycle #{cycle_count}: {e}")
            logging.info("⏳ Waiting 30 seconds before retrying...")
            time.sleep(30)
            continue  # Continue with next cycle even if one fails

if __name__ == "__main__":
    unlimited_parallel_main()

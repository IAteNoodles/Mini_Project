#!/usr/bin/env python3
"""
Google News Scraper with MongoDB Storage
Simplified version that scrapes Google News and saves to MongoDB using the same system as Indian scraper
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

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# Browser automation
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import WebDriverException, TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Content extraction
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Config -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

# Scraping settings
MAX_ARTICLES_PER_TOPIC = 10
MAX_CONCURRENT_BROWSERS = 3
PAGE_LOAD_TIMEOUT = 30
MIN_CONTENT_LENGTH = 200
HEADLESS = True

# Local storage file
LOCAL_DB_FILE = "google_news_backup.json"

# ChromeDriver path (update this)
CHROMEDRIVER_PATH = r"C:\Tools\chromedriver.exe"

class LocalDatabase:
    """Local JSON database as MongoDB fallback"""
    
    def __init__(self, db_file=LOCAL_DB_FILE):
        self.db_file = db_file
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
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_existing_urls(self):
        return {article['url'] for article in self.data['articles']}
    
    def save_article(self, url, headline, content, source="Google News"):
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
    
    def count_articles(self):
        return len(self.data['articles'])

class DatabaseManager:
    """Unified database manager with MongoDB primary and local fallback"""
    
    def __init__(self):
        self.use_mongodb = False
        self.collection = None
        self.local_db = LocalDatabase()
        
        if MONGO_AVAILABLE and MONGO_URI:
            try:
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')  # Test connection
                db = client[DB_NAME]
                self.collection = db[COLLECTION_NAME]
                self.use_mongodb = True
                logging.info("✅ Connected to MongoDB Atlas")
            except Exception as e:
                logging.warning(f"❌ MongoDB connection failed: {e}")
                logging.info("🔄 Falling back to local JSON storage")
        else:
            logging.info("📝 Using local JSON storage (MongoDB not available)")
    
    def get_existing_urls(self) -> set:
        if self.use_mongodb and self.collection is not None:
            try:
                urls = {doc['url'] for doc in self.collection.find({}, {"url": 1})}
                logging.info(f"📊 MongoDB: {len(urls)} existing URLs")
                return urls
            except Exception as e:
                logging.error(f"MongoDB read error: {e}, falling back to local")
                self.use_mongodb = False
        
        urls = self.local_db.get_existing_urls()
        logging.info(f"📊 Local DB: {len(urls)} existing URLs")
        return urls
    
    def save_article(self, url, headline, content, source="Google News"):
        success = False
        
        if self.use_mongodb and self.collection is not None:
            try:
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
            except Exception as e:
                logging.error(f"MongoDB save error: {e}, using local backup")
                self.use_mongodb = False
        
        # Always save to local as backup
        self.local_db.save_article(url, headline, content, source)
        
        return success or True  # Success if either worked

class GoogleNewsScraper:
    """Google News scraper with URL resolution"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.existing_urls = self.db_manager.get_existing_urls()
        
    def get_google_news_rss(self, topic: str = None, region: str = "IN", language: str = "en") -> List[Dict]:
        """Fetch articles from Google News RSS feed"""
        try:
            if topic:
                # Search for specific topic
                encoded_topic = quote(topic)
                rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl={language}&gl={region}&ceid={region}:{language}"
                logging.info(f"🔍 Searching Google News for topic: '{topic}'")
            else:
                # General news feed
                rss_url = f'https://news.google.com/rss?hl={language}&gl={region}&ceid={region}:{language}'
                logging.info(f"📰 Fetching general Google News feed")
            
            logging.info(f"RSS URL: {rss_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            logging.info(f"📄 Found {len(feed.entries)} entries in RSS feed")
            
            articles = []
            for entry in feed.entries:
                # Clean summary HTML
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
            logging.error(f"Error fetching Google News RSS: {e}")
            return []
    
    def resolve_url_playwright(self, url: str) -> Optional[str]:
        """Resolve redirected URL using Playwright"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=HEADLESS)
                page = browser.new_page()
                page.set_default_timeout(PAGE_LOAD_TIMEOUT * 1000)
                
                # Navigate to the Google News URL
                page.goto(url)
                page.wait_for_load_state('networkidle', timeout=10000)
                
                # Wait for redirects and get final URL
                final_url = page.url
                
                # If still on Google News, try to find the actual article link
                if 'news.google.com' in final_url:
                    try:
                        # Look for article links on the page
                        article_links = page.query_selector_all('a[href*="http"]')
                        for link in article_links:
                            href = link.get_attribute('href')
                            if href and not any(domain in href for domain in ['google.com', 'youtube.com', 'facebook.com']):
                                # Found a potential article link
                                final_url = href
                                break
                    except:
                        pass
                
                browser.close()
                return final_url
                
        except Exception as e:
            logging.error(f"Playwright URL resolution failed for {url}: {e}")
            return None
    
    def resolve_url_selenium(self, url: str) -> Optional[str]:
        """Resolve redirected URL using Selenium"""
        driver = None
        try:
            options = Options()
            if HEADLESS:
                options.add_argument("--headless")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-notifications")
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            service = Service(executable_path=CHROMEDRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            driver.implicitly_wait(5)
            
            # Navigate to Google News URL
            driver.get(url)
            time.sleep(5)  # Wait for redirects and page load
            
            final_url = driver.current_url
            
            # If still on Google News, try to find article links
            if 'news.google.com' in final_url:
                try:
                    # Look for article links
                    links = driver.find_elements("css selector", "a[href*='http']")
                    for link in links:
                        href = link.get_attribute('href')
                        if href and not any(domain in href for domain in ['google.com', 'youtube.com', 'facebook.com']):
                            final_url = href
                            break
                except:
                    pass
            
            return final_url
            
        except Exception as e:
            logging.error(f"Selenium URL resolution failed for {url}: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def extract_url_from_google_news(self, google_url: str) -> Optional[str]:
        """Extract actual article URL from Google News redirect URL"""
        try:
            # Try to decode the URL from Google News format
            if 'news.google.com/rss/articles/' in google_url:
                # Use requests to follow redirects with proper headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
                
                # Try multiple redirect following strategies
                session = requests.Session()
                session.headers.update(headers)
                
                # Method 1: Follow redirects normally
                try:
                    response = session.get(google_url, allow_redirects=True, timeout=15)
                    final_url = response.url
                    
                    if final_url != google_url and 'news.google.com' not in final_url:
                        if self.is_valid_news_url(final_url):
                            logging.info(f"✅ URL resolved via redirects: {final_url}")
                            return final_url
                except Exception as e:
                    logging.debug(f"Redirect method failed: {e}")
                
                # Method 2: Parse response content for article links
                try:
                    response = session.get(google_url, allow_redirects=False, timeout=10)
                    
                    # Check for redirect in headers
                    if 'Location' in response.headers:
                        redirect_url = response.headers['Location']
                        if self.is_valid_news_url(redirect_url):
                            logging.info(f"✅ URL found in Location header: {redirect_url}")
                            return redirect_url
                    
                    # Parse HTML content for article links
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for various types of article links
                    selectors = [
                        'a[href*="http"]',
                        'a[data-n-tid]',
                        'article a',
                        '.article a',
                        '[data-article-url]'
                    ]
                    
                    for selector in selectors:
                        links = soup.select(selector)
                        for link in links:
                            href = link.get('href') or link.get('data-article-url')
                            if href and href.startswith('http') and self.is_valid_news_url(href):
                                logging.info(f"✅ URL found via selector {selector}: {href}")
                                return href
                                
                except Exception as e:
                    logging.debug(f"Content parsing method failed: {e}")
                
                # Method 3: Try to extract from Google News URL structure
                try:
                    import base64
                    import urllib.parse
                    
                    # Google News URLs sometimes contain encoded information
                    if 'CBM' in google_url:
                        # Try to decode the CBM parameter
                        parts = google_url.split('CBM')
                        if len(parts) > 1:
                            encoded_part = parts[1].split('?')[0]
                            try:
                                # Attempt base64 decoding
                                decoded = base64.b64decode(encoded_part + '==')
                                decoded_str = decoded.decode('utf-8', errors='ignore')
                                
                                # Look for HTTP URLs in the decoded string
                                import re
                                urls = re.findall(r'https?://[^\s<>"]+', decoded_str)
                                for url in urls:
                                    if self.is_valid_news_url(url):
                                        logging.info(f"✅ URL extracted from CBM encoding: {url}")
                                        return url
                            except:
                                pass
                                
                except Exception as e:
                    logging.debug(f"URL structure extraction failed: {e}")
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to extract URL from Google News: {e}")
            return None
    
    def resolve_url(self, url: str) -> Optional[str]:
        """Resolve URL using multiple methods"""
        # First try to extract directly from Google News URL
        extracted_url = self.extract_url_from_google_news(url)
        if extracted_url and self.is_valid_news_url(extracted_url):
            return extracted_url
        
        # Fallback to browser automation
        if PLAYWRIGHT_AVAILABLE:
            resolved = self.resolve_url_playwright(url)
            if resolved and self.is_valid_news_url(resolved):
                return resolved
        
        if SELENIUM_AVAILABLE:
            resolved = self.resolve_url_selenium(url)
            if resolved and self.is_valid_news_url(resolved):
                return resolved
        
        # Last resort: use original URL if it's valid
        if self.is_valid_news_url(url):
            return url
        
        return None
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract article content using trafilatura or requests"""
        try:
            if TRAFILATURA_AVAILABLE:
                # Use trafilatura for better content extraction
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    content = trafilatura.extract(downloaded, 
                                                include_comments=False,
                                                include_tables=False)
                    if content and len(content) >= MIN_CONTENT_LENGTH:
                        return content
            
            # Fallback to requests + BeautifulSoup
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', '[role="main"]', '.content', '.article-content',
                '.story-content', '.post-content', 'main', '.entry-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(strip=True)
                    break
            
            if not content:
                # Fallback to body text
                content = soup.get_text(strip=True)
            
            return content if len(content) >= MIN_CONTENT_LENGTH else None
            
        except Exception as e:
            logging.error(f"Content extraction failed for {url}: {e}")
            return None
    
    def is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news article"""
        if not url or not url.startswith('http'):
            return False
        
        # Skip Google/YouTube/social media URLs
        skip_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'pinterest.com', 'reddit.com',
            'news.google.com', 'google.co'
        ]
        
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        
        if any(skip in domain.lower() for skip in skip_domains):
            return False
        
        # Check for news-like patterns
        path = parsed.path.lower()
        news_indicators = [
            '/news/', '/article/', '/story/', '/post/', '/politics/',
            '/world/', '/business/', '/tech/', '/opinion/', '/breaking/',
            'news', 'article', 'story', 'report'
        ]
        
        # URL should have news indicators or be from a known news domain
        has_news_pattern = any(indicator in path or indicator in domain.lower() for indicator in news_indicators)
        
        # Known news domains (partial list)
        news_domains = [
            'bbc', 'cnn', 'reuters', 'ap', 'guardian', 'nytimes', 'washingtonpost',
            'wsj', 'bloomberg', 'ndtv', 'timesofindia', 'thehindu', 'indianexpress',
            'hindustantimes', 'livemint', 'economictimes', 'firstpost', 'scroll',
            'thewire', 'opindia', 'aljazeera', 'rt', 'foxnews', 'breitbart'
        ]
        
        is_news_domain = any(news_domain in domain.lower() for news_domain in news_domains)
        
        return has_news_pattern or is_news_domain
    
    def scrape_topic(self, topic: str, max_articles: int = MAX_ARTICLES_PER_TOPIC) -> int:
        """Scrape articles for a specific topic"""
        logging.info(f"🔍 Starting scrape for topic: '{topic}' (max {max_articles} articles)")
        
        # Get RSS articles
        rss_articles = self.get_google_news_rss(topic)
        if not rss_articles:
            logging.warning(f"No RSS articles found for topic: '{topic}'")
            return 0
        
        saved_count = 0
        processed_count = 0
        
        for article in rss_articles[:max_articles * 2]:  # Get extra to account for filtering
            if saved_count >= max_articles:
                break
                
            processed_count += 1
            google_url = article['link']
            title = article['title']
            
            logging.info(f"📄 Processing {processed_count}: {title[:60]}...")
            
            # Resolve the final URL
            final_url = self.resolve_url(google_url)
            if not final_url or not self.is_valid_news_url(final_url):
                logging.info(f"⏭️ Skipping invalid URL: {final_url}")
                continue
            
            # Check if already exists
            if final_url in self.existing_urls:
                logging.info(f"⏭️ Article already exists: {final_url}")
                continue
            
            # Extract content
            content = self.extract_content(final_url)
            if not content:
                logging.info(f"⏭️ No content extracted from: {final_url}")
                continue
            
            # Save article
            success = self.db_manager.save_article(
                url=final_url,
                headline=title,
                content=content,
                source=f"Google News - {topic}" if topic else "Google News"
            )
            
            if success:
                saved_count += 1
                self.existing_urls.add(final_url)
                logging.info(f"✅ Saved article {saved_count}/{max_articles}: {title[:60]}...")
            else:
                logging.warning(f"❌ Failed to save article: {final_url}")
        
        logging.info(f"✅ Topic '{topic}' completed: {saved_count} articles saved")
        return saved_count
    
    def scrape_general_news(self, max_articles: int = MAX_ARTICLES_PER_TOPIC) -> int:
        """Scrape general news articles"""
        return self.scrape_topic(None, max_articles)

def main():
    """Main scraper function"""
    logging.info("🚀 Starting Google News Scraper")
    
    scraper = GoogleNewsScraper()
    
    # Define topics to scrape
    topics = [
        "politics",
        "international news", 
        "business",
        "technology",
        "india news",
        "world news",
        "economy",
        "government policy",
        "election",
        "china",
        "usa",
        "russia",
        "ukraine",
        "climate change",
        "artificial intelligence",
        "cryptocurrency"
    ]
    
    total_saved = 0
    
    # Scrape general news first
    logging.info("📰 Scraping general news...")
    general_count = scraper.scrape_general_news(20)
    total_saved += general_count
    
    # Scrape specific topics
    for topic in topics:
        try:
            count = scraper.scrape_topic(topic, 15)
            total_saved += count
            time.sleep(2)  # Brief pause between topics
        except Exception as e:
            logging.error(f"Error scraping topic '{topic}': {e}")
            continue
    
    logging.info(f"🎉 Scraping completed! Total articles saved: {total_saved}")
    
    # Show database stats
    existing_urls = scraper.db_manager.get_existing_urls()
    logging.info(f"📊 Total articles in database: {len(existing_urls)}")

if __name__ == "__main__":
    main()

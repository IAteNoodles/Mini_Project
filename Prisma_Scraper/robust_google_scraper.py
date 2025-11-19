#!/usr/bin/env python3
"""
Robust Google News Scraper - Focus on what works
Simplified version that prioritizes the working URL resolution methods
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

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Config -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

# Scraping settings
MAX_ARTICLES_PER_TOPIC = 15
MIN_CONTENT_LENGTH = 200
REQUEST_TIMEOUT = 25  # Increased from 15 to 25 seconds for Google News
GOOGLE_NEWS_WAIT_TIME = 12  # Wait time for JavaScript redirects (from your original code)

# Local storage file
LOCAL_DB_FILE = "google_news_backup.json"

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

class RobustGoogleNewsScraper:
    """Google News scraper focusing on robust URL resolution"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.existing_urls = self.db_manager.get_existing_urls()
        self.session = requests.Session()
        
        # Setup session with proper headers (more comprehensive like your original)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
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
    
    def extract_url_from_google_news(self, google_url: str) -> Optional[str]:
        """Extract actual article URL using multiple robust methods with patience"""
        if not google_url or 'news.google.com' not in google_url:
            return google_url if self.is_valid_news_url(google_url) else None
        
        logging.debug(f"🔍 Resolving Google News URL: {google_url}")
        
        # Method 1: Patient redirect following with multiple attempts
        for attempt in range(3):
            try:
                # Increase timeout progressively (Google News can be slow)
                timeout = 15 + (attempt * 5)  # 15s, 20s, 25s
                logging.debug(f"Attempt {attempt + 1}/3: Following redirects (timeout: {timeout}s)")
                
                response = self.session.get(google_url, allow_redirects=True, timeout=timeout)
                final_url = response.url
                
                if final_url != google_url and 'news.google.com' not in final_url:
                    if self.is_valid_news_url(final_url):
                        logging.debug(f"✅ URL resolved via redirects (attempt {attempt + 1}): {final_url}")
                        return final_url
                
                # If still on Google News, wait and try again
                if 'news.google.com' in final_url and attempt < 2:
                    wait_time = 3 + attempt  # 3s, 4s
                    logging.debug(f"Still on Google News, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logging.debug(f"Redirect attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)  # Brief pause before retry
        
        # Method 2: Selenium-style approach - Load page and wait for JavaScript redirects
        try:
            # Import selenium components only if needed
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.common.exceptions import TimeoutException, WebDriverException
            
            logging.debug("Trying Selenium approach for JavaScript redirects...")
            
            # Setup Chrome options (headless)
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-web-security")
            options.add_argument("--allow-running-insecure-content")
            
            # Try to find Chrome driver
            try:
                driver = webdriver.Chrome(options=options)
            except:
                # If Chrome not available, skip this method
                logging.debug("Chrome driver not available, skipping Selenium method")
                raise Exception("Chrome driver not available")
            
            try:
                driver.set_page_load_timeout(30)  # Give it 30 seconds
                driver.implicitly_wait(5)
                
                logging.debug(f"Loading Google News URL with Selenium...")
                driver.get(google_url)
                
                # Wait for potential JavaScript redirects (like your original code)
                logging.debug("Waiting for JavaScript redirects...")
                time.sleep(12)  # Wait longer than your original 4s - Google News can be slow
                
                final_url = driver.current_url
                
                if final_url != google_url and 'news.google.com' not in final_url:
                    if self.is_valid_news_url(final_url):
                        logging.debug(f"✅ URL resolved via Selenium: {final_url}")
                        return final_url
                
            finally:
                try:
                    driver.quit()
                except:
                    pass
                    
        except Exception as e:
            logging.debug(f"Selenium method failed: {e}")
        
        # Method 3: Check redirect headers with patience
        try:
            response = self.session.get(google_url, allow_redirects=False, timeout=20)
            
            if 'Location' in response.headers:
                redirect_url = response.headers['Location']
                if self.is_valid_news_url(redirect_url):
                    logging.debug(f"✅ URL found in Location header: {redirect_url}")
                    return redirect_url
        except Exception as e:
            logging.debug(f"Header redirect method failed: {e}")
        
        # Method 4: Try alternative Google News URL formats
        try:
            # Some Google News URLs can be converted to direct links
            if '/articles/' in google_url:
                # Try removing some Google News parameters
                clean_url = google_url.split('?oc=')[0]  # Remove oc parameter
                if clean_url != google_url:
                    response = self.session.get(clean_url, allow_redirects=True, timeout=15)
                    if response.url != clean_url and self.is_valid_news_url(response.url):
                        logging.debug(f"✅ URL resolved via clean URL: {response.url}")
                        return response.url
        except Exception as e:
            logging.debug(f"Clean URL method failed: {e}")
        
        # Method 5: Parse page content for article links
        try:
            response = self.session.get(google_url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for article links with various selectors
            selectors = [
                'a[href*="http"]:not([href*="google.com"])',
                'a[data-n-tid]',
                'article a',
                '.article a',
                '[data-article-url]',
                'a[target="_blank"]',
                'a[href*="://"]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links[:10]:  # Check more links
                    href = link.get('href') or link.get('data-article-url')
                    if href and self.is_valid_news_url(href):
                        logging.debug(f"✅ URL found via selector {selector}: {href}")
                        return href
        except Exception as e:
            logging.debug(f"Content parsing method failed: {e}")
        
        # Method 6: Advanced base64 decoding
        try:
            if any(pattern in google_url for pattern in ['CBM', 'CAI', 'CCAiC', 'CCM']):
                # Extract encoded parts with multiple patterns
                patterns = [r'CBM([^?&]+)', r'CAI([^?&]+)', r'CCAiC([^?&]+)', r'CCM([^?&]+)']
                for pattern in patterns:
                    matches = re.findall(pattern, google_url)
                    for match in matches:
                        try:
                            # Try base64 decoding with padding
                            padded = match + '=' * (4 - len(match) % 4)
                            decoded = base64.b64decode(padded, validate=True)
                            decoded_str = decoded.decode('utf-8', errors='ignore')
                            
                            # Look for URLs in decoded string
                            url_patterns = [
                                r'https?://[^\s<>"\']+[a-zA-Z0-9/]',
                                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                            ]
                            
                            for url_pattern in url_patterns:
                                urls = re.findall(url_pattern, decoded_str)
                                for url in urls:
                                    if self.is_valid_news_url(url):
                                        logging.debug(f"✅ URL decoded from base64: {url}")
                                        return url
                        except:
                            continue
        except Exception as e:
            logging.debug(f"URL decoding method failed: {e}")
        
        logging.debug(f"❌ Could not resolve Google News URL: {google_url}")
        return None
    
    def is_valid_news_url(self, url: str) -> bool:
        """Check if URL is a valid news article"""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted domains
        skip_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'linkedin.com', 'pinterest.com', 'reddit.com',
            'news.google.com', 'google.co', 'googlenews.com'
        ]
        
        parsed = urlparse(url)
        domain = parsed.hostname or ''
        
        if any(skip in domain.lower() for skip in skip_domains):
            return False
        
        # Must have a valid domain
        if not domain or '.' not in domain:
            return False
        
        # Check for news-like patterns
        path = parsed.path.lower()
        news_indicators = [
            '/news/', '/article/', '/story/', '/post/', '/politics/',
            '/world/', '/business/', '/tech/', '/opinion/', '/breaking/',
            'news', 'article', 'story', 'report', '/2024/', '/2025/'
        ]
        
        has_news_pattern = any(indicator in path or indicator in domain.lower() for indicator in news_indicators)
        
        # Known news domains
        news_domains = [
            'bbc', 'cnn', 'reuters', 'ap', 'guardian', 'nytimes', 'washingtonpost',
            'wsj', 'bloomberg', 'ndtv', 'timesofindia', 'thehindu', 'indianexpress',
            'hindustantimes', 'livemint', 'economictimes', 'firstpost', 'scroll',
            'thewire', 'opindia', 'aljazeera', 'rt', 'foxnews', 'breitbart',
            'reuters', 'politico', 'newsweek', 'forbes', 'cnbc', 'abcnews'
        ]
        
        is_news_domain = any(news_domain in domain.lower() for news_domain in news_domains)
        
        return has_news_pattern or is_news_domain
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract article content"""
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
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article', '[role="main"]', '.content', '.article-content',
                '.story-content', '.post-content', 'main', '.entry-content',
                '.article-body', '.story-body', '.post-body'
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
    
    def scrape_topic(self, topic: str = None, max_articles: int = MAX_ARTICLES_PER_TOPIC) -> int:
        """Scrape articles for a specific topic with improved patience"""
        topic_name = topic or "General News"
        logging.info(f"🔍 Starting scrape for topic: '{topic_name}' (max {max_articles} articles)")
        
        # Get RSS articles
        rss_articles = self.get_google_news_rss(topic)
        if not rss_articles:
            logging.warning(f"No RSS articles found for topic: '{topic_name}'")
            return 0
        
        saved_count = 0
        processed_count = 0
        failed_count = 0
        
        for article in rss_articles:
            if saved_count >= max_articles:
                break
                
            processed_count += 1
            google_url = article['link']
            title = article['title']
            
            logging.info(f"📄 Processing {processed_count}: {title[:60]}...")
            
            # Resolve the final URL with patience
            final_url = None
            for attempt in range(2):  # Try twice for stubborn URLs
                final_url = self.extract_url_from_google_news(google_url)
                if final_url:
                    break
                elif attempt == 0:
                    # Wait a bit before second attempt (Google News might need time)
                    logging.debug(f"First attempt failed, waiting 5s before retry...")
                    time.sleep(5)
            
            if not final_url:
                failed_count += 1
                logging.info(f"⏭️ Could not resolve URL after 2 attempts: {google_url}")
                # Continue to next article instead of giving up entirely
                continue
            
            # Check if already exists
            if final_url in self.existing_urls:
                logging.info(f"⏭️ Article already exists: {final_url}")
                continue
            
            # Extract content with retries
            content = None
            for attempt in range(2):  # Try twice for content extraction
                content = self.extract_content(final_url)
                if content:
                    break
                elif attempt == 0:
                    logging.debug(f"Content extraction failed, waiting 3s before retry...")
                    time.sleep(3)
            
            if not content:
                logging.info(f"⏭️ No content extracted after 2 attempts from: {final_url}")
                continue
            
            # Save article
            source_name = f"Google News - {topic}" if topic else "Google News"
            success = self.db_manager.save_article(
                url=final_url,
                headline=title,
                content=content,
                source=source_name
            )
            
            if success:
                saved_count += 1
                self.existing_urls.add(final_url)
                logging.info(f"✅ Saved article {saved_count}/{max_articles}: {title[:60]}...")
            else:
                logging.warning(f"❌ Failed to save article: {final_url}")
            
            # Pause between articles to be respectful (increased from 1s)
            time.sleep(2)
        
        success_rate = (saved_count / processed_count * 100) if processed_count > 0 else 0
        logging.info(f"✅ Topic '{topic_name}' completed: {saved_count} articles saved, {failed_count} failed URLs, {success_rate:.1f}% success rate")
        return saved_count

def main():
    """Main scraper function"""
    logging.info("🚀 Starting Robust Google News Scraper")
    
    scraper = RobustGoogleNewsScraper()
    
    # Define topics to scrape
    topics = [
        None,  # General news
        "politics",
        "international news", 
        "business",
        "technology",
        "india news",
        "world news",
        "economy",
        "government policy",
        "china",
        "usa",
        "russia",
        "ukraine",
        "climate change",
        "artificial intelligence"
    ]
    
    total_saved = 0
    
    for i, topic in enumerate(topics, 1):
        try:
            logging.info(f"\n📰 [{i}/{len(topics)}] Processing topic: {topic or 'General News'}")
            count = scraper.scrape_topic(topic, 10)  # 10 articles per topic
            total_saved += count
            
            # Brief pause between topics
            if i < len(topics):
                time.sleep(2)
                
        except KeyboardInterrupt:
            logging.info("🛑 Scraping interrupted by user")
            break
        except Exception as e:
            logging.error(f"Error scraping topic '{topic}': {e}")
            continue
    
    logging.info(f"\n🎉 Scraping completed! Total articles saved: {total_saved}")
    
    # Show database stats
    existing_urls = scraper.db_manager.get_existing_urls()
    logging.info(f"📊 Total articles in database: {len(existing_urls)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fixed Indian News Scraper with Local Storage Fallback
Handles MongoDB connection issues by using local JSON storage as backup
"""

import os
import time
import logging
import re
import json
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import trafilatura

# Try MongoDB, fallback to local storage
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# Import modular rules for problematic sites
try:
    from site_rules import (
        get_domain_key, is_valid_article_url, is_excluded_url,
        get_content_selectors, get_priority_sections, filter_links,
        SITE_ARTICLE_PATTERNS, DOMAIN_EXCLUSIONS
    )
    MODULAR_RULES_AVAILABLE = True
except ImportError as e:
    MODULAR_RULES_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------- Config -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"
COLLECTION_NAME = "articles"

HEADLESS = True
SLEEP_TIME_SECONDS = 1800  # 30 minutes
PER_SITE_LIMIT = 25
MIN_CONTENT_LENGTH = 200
NAV_TIMEOUT = 60000  # Reduced timeout
WAIT_SELECTOR_TIMEOUT = 15000

# Local storage file
LOCAL_DB_FILE = "local_articles_backup.json"

# News sites to scrape - Mix of Indian, International, Controversial & Biased Sources
NEWS_SITES: List[Tuple[str, str]] = [
    # ===== EXISTING INDIAN MAINSTREAM =====
    ("NDTV", "https://www.ndtv.com/"),
    ("Times of India", "https://timesofindia.indiatimes.com/"),
    ("The Hindu", "https://www.thehindu.com/"),
    ("Indian Express", "https://indianexpress.com/"),
    ("Hindustan Times", "https://www.hindustantimes.com/"),
    ("LiveMint", "https://www.livemint.com/"),
    ("Economic Times", "https://economictimes.indiatimes.com/"),
    ("Business Standard", "https://www.business-standard.com/"),
    ("India Today", "https://www.indiatoday.in/"),
    ("Deccan Herald", "https://www.deccanherald.com/"),
    ("Scroll", "https://scroll.in/"),
    ("News18", "https://www.news18.com/"),
    
    # ===== INDIAN CONTROVERSIAL & BIASED =====
    ("Republic TV", "https://www.republicworld.com/"),
    ("OpIndia", "https://www.opindia.com/"),
    ("The Wire", "https://thewire.in/"),
    ("Alt News", "https://www.altnews.in/"),
    ("Swarajya", "https://swarajyamag.com/"),
    ("Zee News", "https://zeenews.india.com/"),
    ("India TV", "https://www.indiatv.in/"),
    ("Times Now", "https://www.timesnownews.com/"),
    ("NewsLaundry", "https://www.newslaundry.com/"),
    ("The Print", "https://theprint.in/"),
    ("Firstpost", "https://www.firstpost.com/"),
    ("DNA India", "https://www.dnaindia.com/"),
    
    # ===== INTERNATIONAL MAINSTREAM =====
    ("BBC", "https://www.bbc.com/news"),
    ("CNN", "https://www.cnn.com/"),
    ("Reuters", "https://www.reuters.com/"),
    ("Associated Press", "https://apnews.com/"),
    ("The Guardian", "https://www.theguardian.com/"),
    ("New York Times", "https://www.nytimes.com/"),
    ("Washington Post", "https://www.washingtonpost.com/"),
    ("Wall Street Journal", "https://www.wsj.com/"),
    ("Financial Times", "https://www.ft.com/"),
    ("Bloomberg", "https://www.bloomberg.com/"),
    
    # ===== INTERNATIONAL CONTROVERSIAL & BIASED =====
    ("Fox News", "https://www.foxnews.com/"),
    ("Breitbart", "https://www.breitbart.com/"),
    ("Daily Mail", "https://www.dailymail.co.uk/"),
    ("RT (Russia Today)", "https://www.rt.com/"),
    ("Al Jazeera", "https://www.aljazeera.com/"),
    ("Xinhua", "https://english.news.cn/"),
    ("CGTN", "https://www.cgtn.com/"),
    ("Press TV", "https://www.presstv.ir/"),
    ("The Intercept", "https://theintercept.com/"),
    ("Jacobin", "https://jacobin.com/"),
    ("Common Dreams", "https://www.commondreams.org/"),
    ("Truthout", "https://truthout.org/"),
    
    # ===== MIDDLE EAST & REGIONAL CONTROVERSIAL =====
    ("Times of Israel", "https://www.timesofisrael.com/"),
    ("Haaretz", "https://www.haaretz.com/"),
    ("Middle East Eye", "https://www.middleeasteye.net/"),
    ("Arab News", "https://www.arabnews.com/"),
    ("The National (UAE)", "https://www.thenationalnews.com/"),
    
    # ===== ALTERNATIVE & INDEPENDENT =====
    ("Zero Hedge", "https://www.zerohedge.com/"),
    ("The Grayzone", "https://thegrayzone.com/"),
    ("MintPress News", "https://www.mintpressnews.com/"),
    ("Strategic Culture", "https://www.strategic-culture.org/"),
    ("Global Research", "https://www.globalresearch.ca/"),
    
    # ===== TECH & BUSINESS FOCUSED =====
    ("TechCrunch", "https://techcrunch.com/"),
    ("Ars Technica", "https://arstechnica.com/"),
    ("The Verge", "https://www.theverge.com/"),
    ("Wired", "https://www.wired.com/"),
    ("Politico", "https://www.politico.com/"),
    
    # ===== REGIONAL INDIAN (CONTROVERSIAL) =====
    ("Asianet News", "https://english.asianetnews.com/"),
    ("News Minute", "https://www.thenewsminute.com/"),
    ("Catch News", "https://www.catchnews.com/"),
    ("Quint", "https://www.thequint.com/"),
]

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
    
    def save_article(self, url, headline, content, site):
        article = {
            'url': url,
            'headline': headline,
            'content': content,
            'site': site,
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
    
    def save_article(self, url, headline, content, site):
        success = False
        
        if self.use_mongodb and self.collection is not None:
            try:
                self.collection.replace_one(
                    {'url': url},
                    {
                        'url': url,
                        'headline': headline,
                        'content': content,
                        'site': site,
                        'scraped_at': datetime.now()
                    },
                    upsert=True
                )
                success = True
            except Exception as e:
                logging.error(f"MongoDB save error: {e}, using local backup")
                self.use_mongodb = False
        
        # Always save to local as backup
        self.local_db.save_article(url, headline, content, site)
        
        return success or True  # Success if either worked

def domain_key_for_host(hostname: str) -> str:
    """Get domain key for hostname"""
    if not hostname:
        return ""
    
    domain_mappings = {
        # ===== EXISTING INDIAN MAINSTREAM =====
        'ndtv.com': 'ndtv.com',
        'www.ndtv.com': 'ndtv.com',
        'timesofindia.indiatimes.com': 'timesofindia.indiatimes.com',
        'economictimes.indiatimes.com': 'economictimes.indiatimes.com',
        'www.thehindu.com': 'thehindu.com',
        'indianexpress.com': 'indianexpress.com',
        'www.hindustantimes.com': 'hindustantimes.com',
        'www.livemint.com': 'livemint.com',
        'www.business-standard.com': 'business-standard.com',
        'www.indiatoday.in': 'indiatoday.in',
        'www.deccanherald.com': 'deccanherald.com',
        'scroll.in': 'scroll.in',
        'www.news18.com': 'news18.com',
        
        # ===== INDIAN CONTROVERSIAL & BIASED =====
        'www.republicworld.com': 'republicworld.com',
        'republicworld.com': 'republicworld.com',
        'www.opindia.com': 'opindia.com',
        'opindia.com': 'opindia.com',
        'thewire.in': 'thewire.in',
        'www.thewire.in': 'thewire.in',
        'www.altnews.in': 'altnews.in',
        'altnews.in': 'altnews.in',
        'swarajyamag.com': 'swarajyamag.com',
        'www.swarajyamag.com': 'swarajyamag.com',
        'zeenews.india.com': 'zeenews.india.com',
        'www.indiatv.in': 'indiatv.in',
        'indiatv.in': 'indiatv.in',
        'www.timesnownews.com': 'timesnownews.com',
        'timesnownews.com': 'timesnownews.com',
        'www.newslaundry.com': 'newslaundry.com',
        'newslaundry.com': 'newslaundry.com',
        'theprint.in': 'theprint.in',
        'www.theprint.in': 'theprint.in',
        'www.firstpost.com': 'firstpost.com',
        'firstpost.com': 'firstpost.com',
        'www.dnaindia.com': 'dnaindia.com',
        'dnaindia.com': 'dnaindia.com',
        
        # ===== INTERNATIONAL MAINSTREAM =====
        'www.bbc.com': 'bbc.com',
        'bbc.com': 'bbc.com',
        'www.cnn.com': 'cnn.com',
        'cnn.com': 'cnn.com',
        'www.reuters.com': 'reuters.com',
        'reuters.com': 'reuters.com',
        'apnews.com': 'apnews.com',
        'www.apnews.com': 'apnews.com',
        'www.theguardian.com': 'theguardian.com',
        'theguardian.com': 'theguardian.com',
        'www.nytimes.com': 'nytimes.com',
        'nytimes.com': 'nytimes.com',
        'www.washingtonpost.com': 'washingtonpost.com',
        'washingtonpost.com': 'washingtonpost.com',
        'www.wsj.com': 'wsj.com',
        'wsj.com': 'wsj.com',
        'www.ft.com': 'ft.com',
        'ft.com': 'ft.com',
        'www.bloomberg.com': 'bloomberg.com',
        'bloomberg.com': 'bloomberg.com',
        
        # ===== INTERNATIONAL CONTROVERSIAL & BIASED =====
        'www.foxnews.com': 'foxnews.com',
        'foxnews.com': 'foxnews.com',
        'www.breitbart.com': 'breitbart.com',
        'breitbart.com': 'breitbart.com',
        'www.dailymail.co.uk': 'dailymail.co.uk',
        'dailymail.co.uk': 'dailymail.co.uk',
        'www.rt.com': 'rt.com',
        'rt.com': 'rt.com',
        'www.aljazeera.com': 'aljazeera.com',
        'aljazeera.com': 'aljazeera.com',
        'english.news.cn': 'xinhua.com',
        'www.cgtn.com': 'cgtn.com',
        'cgtn.com': 'cgtn.com',
        'www.presstv.ir': 'presstv.ir',
        'presstv.ir': 'presstv.ir',
        'theintercept.com': 'theintercept.com',
        'www.theintercept.com': 'theintercept.com',
        'jacobin.com': 'jacobin.com',
        'www.jacobin.com': 'jacobin.com',
        'www.commondreams.org': 'commondreams.org',
        'commondreams.org': 'commondreams.org',
        'truthout.org': 'truthout.org',
        'www.truthout.org': 'truthout.org',
        
        # ===== MIDDLE EAST & REGIONAL =====
        'www.timesofisrael.com': 'timesofisrael.com',
        'timesofisrael.com': 'timesofisrael.com',
        'www.haaretz.com': 'haaretz.com',
        'haaretz.com': 'haaretz.com',
        'www.middleeasteye.net': 'middleeasteye.net',
        'middleeasteye.net': 'middleeasteye.net',
        'www.arabnews.com': 'arabnews.com',
        'arabnews.com': 'arabnews.com',
        'www.thenationalnews.com': 'thenationalnews.com',
        'thenationalnews.com': 'thenationalnews.com',
        
        # ===== ALTERNATIVE & INDEPENDENT =====
        'www.zerohedge.com': 'zerohedge.com',
        'zerohedge.com': 'zerohedge.com',
        'thegrayzone.com': 'thegrayzone.com',
        'www.thegrayzone.com': 'thegrayzone.com',
        'www.mintpressnews.com': 'mintpressnews.com',
        'mintpressnews.com': 'mintpressnews.com',
        'www.strategic-culture.org': 'strategic-culture.org',
        'strategic-culture.org': 'strategic-culture.org',
        'www.globalresearch.ca': 'globalresearch.ca',
        'globalresearch.ca': 'globalresearch.ca',
        
        # ===== TECH & BUSINESS =====
        'techcrunch.com': 'techcrunch.com',
        'www.techcrunch.com': 'techcrunch.com',
        'arstechnica.com': 'arstechnica.com',
        'www.arstechnica.com': 'arstechnica.com',
        'www.theverge.com': 'theverge.com',
        'theverge.com': 'theverge.com',
        'www.wired.com': 'wired.com',
        'wired.com': 'wired.com',
        'www.politico.com': 'politico.com',
        'politico.com': 'politico.com',
        
        # ===== REGIONAL INDIAN =====
        'english.asianetnews.com': 'asianetnews.com',
        'www.thenewsminute.com': 'thenewsminute.com',
        'thenewsminute.com': 'thenewsminute.com',
        'www.catchnews.com': 'catchnews.com',
        'catchnews.com': 'catchnews.com',
        'www.thequint.com': 'thequint.com',
        'thequint.com': 'thequint.com',
    }
    
    return domain_mappings.get(hostname, hostname)

def is_article_url(url: str, allowed_domain: str) -> bool:
    """Check if URL looks like an article"""
    if not url:
        return False
    
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ''
        
        # Must be from allowed domain
        if allowed_domain not in hostname.lower():
            return False
        
        path = parsed.path.lower()
        
        # Enhanced article indicators for diverse international sites
        article_indicators = [
            # Standard patterns
            '/news/', '/article/', '/story/', '/post/', '/articles/',
            '/world/', '/india/', '/business/', '/politics/', '/opinion/',
            '/sports/', '/tech/', '/science/', '/national/', '/international/',
            '-news-', '-article-', '/analysis/', '/report/', '/investigation/',
            
            # International patterns
            '/europe/', '/asia/', '/americas/', '/africa/', '/middle-east/',
            '/global/', '/foreign/', '/domestic/', '/breaking/', '/latest/',
            '/updates/', '/live/', '/feature/', '/in-depth/', '/special/',
            
            # Site-specific patterns
            '/articleshow/', '/cms/', '/photogallery/', '/videos/', '/blogs/',
            '/columns/', '/edit/', '/comment/', '/debate/', '/explained/',
            '/magazine/', '/weekly/', '/daily/', '/trending/', '/viral/',
            
            # Controversial/Opinion patterns
            '/conspiracy/', '/alternative/', '/independent/', '/truth/',
            '/expose/', '/investigation/', '/leak/', '/scandal/', '/corruption/',
            '/propaganda/', '/media/', '/fake-news/', '/fact-check/',
            
            # Regional patterns
            '/city/', '/state/', '/region/', '/local/', '/metro/', '/urban/',
            '/rural/', '/village/', '/district/', '/constituency/',
            
            # Topic-specific patterns
            '/covid/', '/pandemic/', '/health/', '/environment/', '/climate/',
            '/energy/', '/space/', '/defense/', '/military/', '/security/',
            '/cyber/', '/crypto/', '/blockchain/', '/ai/', '/machine-learning/',
        ]
        
        # URL should have article indicators
        if any(indicator in path for indicator in article_indicators):
            return True
        
        # Check for numeric IDs (common in news articles)
        if re.search(r'/\d{4,}', path) or re.search(r'-\d{4,}', path):
            return True
        
        return False
        
    except Exception:
        return False

def is_desired_topic(url: str, headline: str) -> bool:
    """Check if article is about desired topics"""
    content = f"{url} {headline}".lower()
    
    # Enhanced desired topics for controversial and international coverage
    desired = [
        # Political & Government
        'politics', 'government', 'election', 'policy', 'minister', 'parliament',
        'democracy', 'authoritarian', 'dictatorship', 'regime', 'coup', 'revolution',
        'protest', 'demonstration', 'rally', 'activism', 'dissent', 'opposition',
        
        # International Relations
        'world', 'international', 'china', 'usa', 'russia', 'pakistan', 'border', 'war',
        'conflict', 'diplomatic', 'sanctions', 'treaty', 'alliance', 'nato', 'un',
        'nuclear', 'missile', 'terrorism', 'intelligence', 'espionage', 'cyber-attack',
        
        # Economy & Business
        'economy', 'business', 'market', 'trade', 'inflation', 'gdp', 'rupee',
        'recession', 'growth', 'debt', 'crisis', 'bailout', 'stimulus', 'budget',
        'tax', 'regulation', 'monopoly', 'corruption', 'fraud', 'scandal',
        
        # Technology & Innovation
        'technology', 'tech', 'ai', 'startup', 'innovation', 'digital', 'internet',
        'social media', 'privacy', 'surveillance', 'data', 'algorithm', 'blockchain',
        'cryptocurrency', 'cyber', 'hacking', 'breach', 'leak', 'censorship',
        
        # Social Issues
        'culture', 'society', 'social', 'education', 'health', 'environment',
        'climate', 'pollution', 'energy', 'renewable', 'sustainability',
        'employment', 'jobs', 'unemployment', 'workers', 'labor', 'union', 'strike',
        'gender', 'women', 'equality', 'rights', 'justice', 'discrimination',
        'minority', 'religion', 'caste', 'race', 'immigration', 'refugee',
        
        # Controversial Topics
        'conspiracy', 'cover-up', 'whistleblower', 'propaganda', 'bias', 'manipulation',
        'fake news', 'misinformation', 'disinformation', 'fact-check', 'media',
        'freedom', 'censorship', 'ban', 'restriction', 'crackdown', 'suppression',
        
        # Crisis & Emergency
        'pandemic', 'covid', 'disaster', 'emergency', 'crisis', 'outbreak',
        'violence', 'attack', 'bombing', 'shooting', 'accident', 'tragedy',
        
        # Regional Specific
        'kashmir', 'tibet', 'taiwan', 'hong kong', 'ukraine', 'syria', 'iraq',
        'afghanistan', 'israel', 'palestine', 'iran', 'north korea', 'venezuela'
    ]
    
    # Undesired topics (avoid entertainment, sports, lifestyle)
    avoid = [
        'cricket', 'football', 'sport', 'match', 'player', 'team',
        'bollywood', 'celebrity', 'actor', 'actress', 'film', 'movie',
        'astrology', 'horoscope', 'fashion', 'beauty', 'lifestyle',
        'recipe', 'cooking', 'food', 'travel', 'tourism',
        'entertainment', 'gossip', 'personal life'
    ]
    
    # Check for desired topics
    has_desired = any(topic in content for topic in desired)
    has_avoid = any(topic in content for topic in avoid)
    
    # Prefer articles with desired topics and without avoided topics
    if has_desired and not has_avoid:
        return True
    elif has_desired and has_avoid:
        return True  # Still include if has important topics
    elif not has_avoid:
        return True  # Include if no avoided topics
    
    return False

def scrape_article_content(browser, url: str) -> Tuple[str, str]:
    """Scrape article content from URL"""
    try:
        page = browser.new_page()
        
        # Navigate with timeout
        page.goto(url, timeout=NAV_TIMEOUT)
        page.wait_for_load_state('domcontentloaded', timeout=10000)
        
        # Get page content
        html = page.content()
        page.close()
        
        # Extract with trafilatura
        content = trafilatura.extract(html, include_comments=False, include_tables=False)
        
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return url, content
        else:
            return url, ""
            
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        try:
            page.close()
        except:
            pass
        return url, ""

def collect_links_from_page(page, site_url: str, domain_key: str) -> List[Dict[str, str]]:
    """Collect article links from a page"""
    try:
        # Wait for page to load
        page.wait_for_selector('a', timeout=10000)
        page.wait_for_timeout(2000)  # Wait for dynamic content
        
        # Site-specific selectors for better article detection
        domain_key = domain_key_for_host(urlparse(site_url).hostname or '')
        
        selectors = {
            # Indian Sites
            'ndtv.com': ['article a', '.story-card a', '.news_Itm a', 'h2 a', 'h3 a'],
            'timesofindia.indiatimes.com': ['span[itemprop="headline"] a', '.w_tle a', 'h2 a'],
            'thehindu.com': ['.story-card a', 'h3 a', '.element a'],
            'indianexpress.com': ['.articles a', 'h2 a', '.title a'],
            'republicworld.com': ['.story-card a', '.news-card a', 'h2 a', 'h3 a'],
            'opindia.com': ['.post-title a', '.entry-title a', 'h2 a'],
            'thewire.in': ['.post-title a', '.entry-title a', 'h2 a'],
            'altnews.in': ['.entry-title a', '.post-title a', 'h2 a'],
            'swarajyamag.com': ['.article-title a', '.post-title a', 'h2 a'],
            'theprint.in': ['.story-card a', '.post-title a', 'h2 a'],
            
            # International Mainstream
            'bbc.com': ['.media__link', '.gs-c-promo-heading a', 'h3 a'],
            'cnn.com': ['.container__link', '.cd__headline a', 'h3 a'],
            'reuters.com': ['.media-story-card__headline__link', 'h3 a', '.story-title a'],
            'apnews.com': ['.Page-headline a', '.Component-headline a', 'h2 a'],
            'theguardian.com': ['.u-faux-block-link__overlay', '.fc-item__link', 'h3 a'],
            'nytimes.com': ['.css-9mylee', '.story-link', 'h2 a', 'h3 a'],
            'washingtonpost.com': ['.headline a', '.pb-headline a', 'h2 a'],
            'wsj.com': ['.WSJTheme--headline a', '.headline a', 'h2 a'],
            'bloomberg.com': ['.story-list-story__link', '.headline a', 'h2 a'],
            
            # International Controversial
            'foxnews.com': ['.title a', '.headline a', 'h2 a', 'h3 a'],
            'breitbart.com': ['.title a', '.headline a', 'h2 a'],
            'dailymail.co.uk': ['.linkro-darkred', '.article a', 'h2 a'],
            'rt.com': ['.link_hover', '.main-promo__link', 'h2 a'],
            'aljazeera.com': ['.u-clickable-card__link', '.article-card__link', 'h2 a'],
            'presstv.ir': ['.title a', '.headline a', 'h2 a'],
            'theintercept.com': ['.Post-title a', '.headline a', 'h2 a'],
            'jacobin.com': ['.post-title a', '.headline a', 'h2 a'],
            'zerohedge.com': ['.teaser-title a', '.headline a', 'h2 a'],
            
            # Tech Sites
            'techcrunch.com': ['.post-block__title__link', '.headline a', 'h2 a'],
            'arstechnica.com': ['.headline a', '.post-title a', 'h2 a'],
            'theverge.com': ['.c-entry-box--compact__title a', '.headline a', 'h2 a'],
            'wired.com': ['.card__headline a', '.headline a', 'h2 a'],
            'politico.com': ['.headline a', '.story-headline a', 'h2 a'],
        }
        
        # Get site-specific selectors or use default
        site_selectors = selectors.get(domain_key, ['a'])
        
        # Try each selector until we find links
        all_links = []
        for selector in site_selectors:
            try:
                links = page.evaluate(f'''() => {{
                    const links = Array.from(document.querySelectorAll('{selector}'));
                    return links.map(a => {{
                        const linkElement = a.tagName === 'A' ? a : a.querySelector('a') || a.closest('a');
                        if (!linkElement) return null;
                        return {{
                            url: linkElement.href,
                            text: (linkElement.textContent || linkElement.innerText || '').trim()
                        }};
                    }}).filter(link => link && link.url && link.text);
                }}''')
                
                if links:
                    all_links.extend(links)
                    break  # Found links with this selector
                    
            except Exception as e:
                continue
        
        # Fallback to generic selector if no site-specific links found
        if not all_links:
            all_links = page.evaluate('''() => {
                const links = Array.from(document.querySelectorAll('a'));
                return links.map(a => ({
                    url: a.href,
                    text: a.textContent.trim()
                })).filter(link => link.url && link.text);
            }''')
        
        # Remove duplicates
        seen_urls = set()
        unique_links = []
        for link in all_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)
        
        # Filter links
        filtered_links = []
        for link in unique_links:
            url = link['url']
            text = link['text']
            
            # Basic filtering
            if is_article_url(url, domain_key) and is_desired_topic(url, text):
                filtered_links.append({
                    'url': url,
                    'headline': text
                })
        
        return filtered_links[:PER_SITE_LIMIT]  # Limit results
        
    except Exception as e:
        logging.error(f"Error collecting links from {site_url}: {e}")
        return []

def scrape_single_site(browser, site_name: str, site_url: str, existing_urls: set, db_manager: DatabaseManager):
    """Scrape a single news site"""
    domain_key = domain_key_for_host(urlparse(site_url).hostname or '')
    
    logging.info(f"\n🔍 [{site_name}] Starting scrape...")
    
    page = browser.new_page()
    candidates = []
    articles_saved = 0
    
    try:
        # Navigate to homepage
        page.goto(site_url, timeout=NAV_TIMEOUT)
        page.wait_for_load_state('domcontentloaded', timeout=10000)
        
        # Collect links
        candidates = collect_links_from_page(page, site_url, domain_key)
        logging.info(f"📰 [{site_name}] Found {len(candidates)} candidate articles")
        
        # Filter out existing URLs
        new_candidates = [c for c in candidates if c['url'] not in existing_urls]
        logging.info(f"🆕 [{site_name}] {len(new_candidates)} new articles to scrape")
        
        # Debug: Show some sample URLs if no new candidates found
        if len(new_candidates) == 0 and len(candidates) > 0:
            logging.info(f"🔍 [{site_name}] DEBUG: Sample existing URLs found:")
            for i, candidate in enumerate(candidates[:3]):
                logging.info(f"   {i+1}. {candidate['url']}")
            logging.info(f"📊 [{site_name}] Total URLs in database: {len(existing_urls)}")
        
        # Scrape new articles
        for i, candidate in enumerate(new_candidates[:PER_SITE_LIMIT]):
            url = candidate['url']
            headline = candidate['headline']
            
            logging.info(f"📄 [{site_name}] ({i+1}/{len(new_candidates)}) {headline[:60]}...")
            
            final_url, content = scrape_article_content(browser, url)
            
            if content and len(content) >= MIN_CONTENT_LENGTH:
                success = db_manager.save_article(final_url, headline, content, domain_key)
                if success:
                    articles_saved += 1
                    existing_urls.add(final_url)
                    logging.info(f"✅ [{site_name}] Saved article")
                else:
                    logging.warning(f"⚠️ [{site_name}] Failed to save article")
            else:
                logging.info(f"⏭️ [{site_name}] Skipped (insufficient content)")
        
        logging.info(f"✅ [{site_name}] Completed: {articles_saved} articles saved")
        
    except Exception as e:
        logging.error(f"❌ [{site_name}] Error: {e}")
    
    finally:
        try:
            page.close()
        except:
            pass
    
    return articles_saved

def main():
    """Main scraper function"""
    logging.info("🚀 Starting Indian News Scraper with Robust Error Handling")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Launch browser
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        
        try:
            while True:
                logging.info(f"\n🔄 Starting scraping cycle at {datetime.now()}")
                
                # Get existing URLs
                existing_urls = db_manager.get_existing_urls()
                
                total_saved = 0
                
                # Scrape each site
                for site_name, site_url in NEWS_SITES:
                    try:
                        saved = scrape_single_site(browser, site_name, site_url, existing_urls, db_manager)
                        total_saved += saved
                        time.sleep(2)  # Brief pause between sites
                    except Exception as e:
                        logging.error(f"Site error for {site_name}: {e}")
                        continue
                
                logging.info(f"\n🎉 Cycle complete! Total articles saved: {total_saved}")
                logging.info(f"💤 Sleeping for {SLEEP_TIME_SECONDS/60:.1f} minutes...")
                
                time.sleep(SLEEP_TIME_SECONDS)
                
        except KeyboardInterrupt:
            logging.info("👋 Scraper stopped by user")
        except Exception as e:
            logging.error(f"💥 Unexpected error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()

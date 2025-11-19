import os
import time
import logging
import re
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Tuple
from datetime import datetime

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import trafilatura
from pymongo import MongoClient

# Import modular rules for problematic sites
try:
    from site_rules import (
        get_domain_key, is_valid_article_url, is_excluded_url,
        get_content_selectors, get_priority_sections, filter_links,
        SITE_ARTICLE_PATTERNS, DOMAIN_EXCLUSIONS
    )
    MODULAR_RULES_AVAILABLE = True
    logging.info("Modular site rules imported successfully")
except ImportError as e:
    MODULAR_RULES_AVAILABLE = False
    logging.warning(f"Modular site rules not available: {e}")

# ----------------- Config -----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "Prisma"  # Explicit DB
COLLECTION_NAME = "articles"

HEADLESS = False  # keep browser visible
SLEEP_TIME_SECONDS = 1800  # 30 minutes between cycles
PER_SITE_LIMIT = 25  # cap per site per cycle
MIN_CONTENT_LENGTH = 200
NAV_TIMEOUT = 90000
WAIT_SELECTOR_TIMEOUT = 20000

# Site-specific article URL regex patterns
SITE_ARTICLE_PATTERNS = {
    "ndtv.com": [
        r"/news/[^/]+/[^/]+-\d+$",  # Main news articles
        r"/india-news/[^/]+-\d+$", r"/world-news/[^/]+-\d+$",
        r"/\d{4}/\d{2}/\d{2}/",  # Date-based URLs
        r"/opinion/[^/]+-\d+$",  # Opinion pieces
        # INTELLIGENT PATTERNS for NDTV based on actual structure:
        r"/india-news/[^/]+-\d{7,}",      # /india-news/article-title-1234567
        r"/world-news/[^/]+-\d{7,}",      # /world-news/article-title-1234567
        r"/business/[^/]+-\d{7,}",        # /business/article-title-1234567
        r"/opinion/[^/]+-\d{7,}",         # /opinion/article-title-1234567
        r"/offbeat/[^/]+-\d{7,}",         # /offbeat/article-title-1234567
        r"/[^/]*-news/[^/]+-\d{7,}",      # Any news section with 7+ digit ID
        r"\.com/[^/]+-\d{7,}$",           # Direct article URLs with 7+ digit ID
    ],
    "indiatimes.com": [
        r"/articleshow/\d+\.cms$",  # Main articles
        r"/city/.+?/\d+\.cms$",  # City news
        # INTRUSIVE PATTERNS for Economic Times:
        r"/news/",            # News section
        r"/india/",           # India news
        r"/world/",           # World news
        r"/politics/",        # Politics section
        r"/industry/",        # Industry news
        r"/markets/",         # Market news
        r"/jobs/",            # Employment news
    ],
    "timesofindia.indiatimes.com": [
        r"/articleshow/\d+\.cms$",  # Main articles
        r"/city/.+?/\d+\.cms$",  # City news
        # INTRUSIVE PATTERNS for Times of India:
        r"/india/",           # India section
        r"/world/",           # World section
        r"/business/",        # Business section
        r"/blogs/",           # Blog posts
        r"/[^/]+-news/",     # Various news sections
    ],
    "thehindu.com": [
        r"/\d{4}/\d{2}/\d{2}/",  # Date-based articles
        r"/news/[^/]+/article\d+\.ece$",  # Article format
        r"/opinion/[^/]+/article\d+\.ece$",  # Opinion
        # INTRUSIVE PATTERNS for The Hindu:
        r"/news/national/",   # National news section (detected)
        r"/news/international/", # International news (detected)
        r"/news/cities/",     # City news (detected)
        r"/business/",        # Business section
        r"/sci-tech/",        # Science & Technology (detected)
        r"/society/",         # Society section
        r"/thread/",          # Thread articles
        r"/todays-paper/",    # Today's paper articles
    ],
    "indianexpress.com": [
        r"/article/[^/]+/.+/?$",  # Main articles
        r"/\d{4}/\d{2}/\d{2}/",  # Date-based
        # INTRUSIVE PATTERNS for Indian Express:
        r"/india/",           # India section
        r"/world/",           # World section
        r"/cities/",          # Cities section
        r"/business/",        # Business section
        r"/explained/",       # Explained articles
        r"/opinion/",         # Opinion pieces
        r"/technology/",      # Technology section
        r"/education/",       # Education section
    ],
    "hindustantimes.com": [
        r"/[a-z-]+-news/.+-\d+\.html$",  # News articles
        r"/\d{4}/\d{2}/\d{2}/",  # Date-based
        r"/opinion/.+-\d+\.html$",  # Opinion pieces
        # INTRUSIVE PATTERNS for Hindustan Times:
        r"/india-news/",      # India news section
        r"/world-news/",      # World news section
        r"/cities/",          # Cities section
        r"/business/",        # Business section
        r"/tech/",            # Technology
        r"/opinion/",         # Opinion articles
        r"/education/",       # Education section
    ],
    "livemint.com": [
        r"/(news|companies|industry)/.+\.html$",  # Business/news
        r"/\d{4}/\d{2}/\d{2}/",  # Date-based
        r"/opinion/.+\.html$",  # Opinion
        # INTRUSIVE PATTERNS for LiveMint:
        r"/news/",            # News section
        r"/politics/",        # Politics section
        r"/economy/",         # Economy section
        r"/industry/",        # Industry news
        r"/market/",          # Market news
        r"/opinion/",         # Opinion pieces
        r"/technology/",      # Technology section
        r"/ai/",              # AI section
        r"/startups/",        # Startup news
    ],
    "economictimes.indiatimes.com": [
        r"/\d+\.cms$",  # Main articles
        r"/industry/.+?/\d+\.cms$",  # Industry news
        r"/markets/.+?/\d+\.cms$",  # Market news
        # INTRUSIVE PATTERNS for Economic Times:
        r"/news/",            # News section
        r"/india/",           # India news
        r"/world/",           # World news
        r"/politics/",        # Politics section
        r"/economy/",         # Economy section
        r"/tech/",            # Technology
        r"/jobs/",            # Employment news
    ],
    "business-standard.com": [
        r"/article/[^/]+/.+\.htm$",  # Main articles
        # INTRUSIVE PATTERNS for Business Standard:
        r"/economy/",         # Economy section
        r"/finance/",         # Finance section
        r"/politics/",        # Politics section
        r"/current-affairs/", # Current affairs
        r"/international/",   # International news
        r"/technology/",      # Technology section
        r"/markets/",         # Market news
        r"/companies/",       # Company news
        r"/opinion/",         # Opinion articles
        r"/specials/",        # Special reports
    ],
    "indiatoday.in": [
        r"/.+/story/",  # Story format
        r"/\d{4}-\d{2}-\d{2}/",  # Date format
        # INTRUSIVE PATTERNS for India Today:
        r"/india/",           # India section
        r"/world/",           # World section
        r"/business/",        # Business section
        r"/technology/",      # Technology section
        r"/education/",       # Education section
        r"/science/",         # Science section
    ],
    "deccanherald.com": [
        r"/.+/\d{4}-\d{2}-\d{2}/",  # Date-based
        r"/opinion/.+/\d{4}-\d{2}-\d{2}/",  # Opinion
        # INTRUSIVE PATTERNS for Deccan Herald:
        r"/india/",           # India section
        r"/world/",           # World section
        r"/business/",        # Business section
        r"/state/",           # State news
        r"/national/",        # National news
        r"/international/",   # International news
        r"/opinion/",         # Opinion articles
        r"/specials/",        # Special reports
    ],
    "scroll.in": [
        r"/article/\d+/",  # Article format
        # INTRUSIVE PATTERNS for Scroll:
        r"/latest/",          # Latest section
        r"/field/",           # Field reports
        r"/video/",           # Video articles
        r"/reel/",            # Reel section
        r"/magazine/",        # Magazine articles
        r"/topic/",           # Topic-based articles
        r"/author/",          # Author articles
        r"/india/",           # India section
        r"/world/",           # World section
    ],
    "news18.com": [
        r"/news/[^/]+/.+/\d{4}/\d{2}/\d{2}/",  # Date-based news
        r"/politics/.+/\d{4}/\d{2}/\d{2}/",  # Politics
        # INTRUSIVE PATTERNS for News18:
        r"/india/",           # India section
        r"/world/",           # World section
        r"/business/",        # Business section
        r"/politics/",        # Politics section
        r"/tech/",            # Technology section
        r"/education/",       # Education section
        r"/auto/",            # Automobile section
        r"/movies/",          # Movies section
        r"/cricket/",         # Cricket section
    ],
}

# Site-specific exclusion patterns - these override article patterns
SITE_EXCLUSION_PATTERNS = {
    "ndtv.com": [
        r"/entertainment/",
        r"/movies/",
        r"/cricket/",
        r"/sports/",
        r"/food/",
        r"/auto/",
        r"/gadgets360/",
        r"/profit/",
        r"/webstories/",
        r"/photos/",
        r"/videos/",
        r"/trends/",
        r"/offbeat/",
        r"/lifestyle/",
    ],
    "indiatimes.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/travel/",
        r"/food/",
        r"/auto/",
        r"/technology/",
        r"/trending/",
        r"/worth/",
        r"/health/",
        r"/relationships/",
        r"/astrology/",
        r"/movies/",
        r"/cricket/",
    ],
    "timesofindia.indiatimes.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/travel/",
        r"/food/",
        r"/auto/",
        r"/technology/",
        r"/trending/",
        r"/astrology/",
        r"/cricket/",
        r"/movies/",
        r"/tv/",
        r"/web-series/",
    ],
    "thehindu.com": [
        r"/sport/",
        r"/entertainment/",
        r"/life-and-style/",
        r"/food/",
        r"/travel/",
        r"/books/",
        r"/cinema/",
        r"/music/",
        r"/dance/",
        r"/theatre/",
        r"/cartoons/",
        r"/crossword/",
        r"/multimedia/",
        r"/young-world/",
    ],
    "indianexpress.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/food/",
        r"/travel/",
        r"/books/",
        r"/health/",
        r"/parenting/",
        r"/horoscope/",
        r"/cricket/",
        r"/football/",
        r"/bollywood/",
        r"/television/",
        r"/explained/",  # Often simplified explainers
    ],
    "hindustantimes.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/fashion/",
        r"/health/",
        r"/travel/",
        r"/food/",
        r"/astrology/",
        r"/cricket/",
        r"/football/",
        r"/bollywood/",
        r"/tv/",
        r"/music/",
        r"/art-and-culture/",  # Often event coverage
        r"/photos/",
        r"/videos/",
    ],
    "livemint.com": [
        r"/sports/",
        r"/lifestyle/",
        r"/fashion/",
        r"/food/",
        r"/books/",
        r"/art/",
        r"/lounge/",  # Lifestyle section
        r"/mint-lounge/",
        r"/leisure/",
        r"/entertainment/",
        r"/cricket/",
        r"/football/",
        r"/auto-news/",
        r"/photos/",
        r"/videos/",
    ],
    "economictimes.indiatimes.com": [
        r"/magazines/",
        r"/panache/",  # Lifestyle
        r"/brands/",
        r"/slideshows/",
        r"/photos/",
        r"/videos/",
        r"/travel/",
        r"/wealth/",  # Personal finance tips
        r"/personal-finance/",
        r"/most-popular/",
        r"/trending/",
    ],
    "business-standard.com": [
        r"/lifestyle/",
        r"/sports/",
        r"/entertainment/",
        r"/leisure/",
        r"/photos/",
        r"/videos/",
        r"/cricket/",
        r"/football/",
        r"/auto/",
        r"/travel/",
        r"/books/",
        r"/specials/",  # Often promotional
    ],
    "indiatoday.in": [
        r"/lifestyle/",
        r"/sports/",
        r"/entertainment/",
        r"/movies/",
        r"/television/",
        r"/music/",
        r"/books/",
        r"/food/",
        r"/fashion/",
        r"/health/",
        r"/travel/",
        r"/auto/",
        r"/gadgets/",
        r"/cricket/",
        r"/football/",
        r"/photos/",
        r"/videos/",
        r"/art-culture/",
        r"/magazine/",
        r"/web-stories/",
    ],
    "deccanherald.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/food/",
        r"/travel/",
        r"/books/",
        r"/cinema/",
        r"/music/",
        r"/cricket/",
        r"/football/",
        r"/photos/",
        r"/videos/",
        r"/supplements/",
        r"/spectrum/",  # Sunday supplement
        r"/metrolife/",  # Lifestyle
    ],
    "scroll.in": [
        r"/reel/",  # Entertainment
        r"/field/",  # Sports
        r"/culture/",  # Often arts/books
        r"/photos/",
        r"/videos/",
    ],
    "news18.com": [
        r"/entertainment/",
        r"/sports/",
        r"/lifestyle/",
        r"/movies/",
        r"/cricket/",
        r"/football/",
        r"/auto/",
        r"/tech/",
        r"/gadgets/",
        r"/photos/",
        r"/videos/",
        r"/web-stories/",
        r"/buzz/",  # Trending/viral
        r"/trending/",
        r"/food/",
        r"/travel/",
        r"/health/",
        r"/books/",
    ],
}

# Site-specific quality indicators - what makes content substantial on each site
SITE_QUALITY_INDICATORS = {
    "ndtv.com": {
        "sections": ["/news/", "/india-news/", "/world-news/", "/opinion/"],
        "quality_words": ["investigation", "exclusive", "ground report", "analysis", "special report"],
        "avoid_headlines": ["photos", "pics", "watch", "video", "gallery"],
    },
    "thehindu.com": {
        "sections": ["/news/", "/opinion/", "/editorial/"],
        "quality_words": ["editorial", "analysis", "explained", "perspective", "comment"],
        "avoid_headlines": ["in pictures", "photo gallery", "watch", "video"],
    },
    "indianexpress.com": {
        "sections": ["/article/", "/opinion/", "/editorial/"],
        "quality_words": ["investigation", "ground zero", "analysis", "editorial", "opinion"],
        "avoid_headlines": ["photos", "pics", "watch", "gallery", "in pictures"],
    },
    "scroll.in": {
        "sections": ["/article/"],
        "quality_words": ["analysis", "investigation", "reportage", "ground report"],
        "avoid_headlines": ["photos", "gallery", "watch", "video"],
    },
    "livemint.com": {
        "sections": ["/news/", "/opinion/", "/politics/", "/industry/"],
        "quality_words": ["analysis", "interview", "exclusive", "investigation"],
        "avoid_headlines": ["photos", "gallery", "personal finance", "wealth"],
    },
}

# Site-specific CSS selectors for better anchor detection - INTELLIGENT SYSTEM
SITE_SELECTORS = {
    "timesofindia.indiatimes.com": [
        'a',
        'article a',
        '.storylisting a',
        'h2 a', 'h3 a',
    ],
    "www.indiatoday.in": [
        'a',
        'article a',
        '.story-block a',
        'h2 a', 'h3 a',
    ],
    "www.thehindu.com": [
        'a',
        'article a',
        '.story-card a',
        'h1 a', 'h2 a', 'h3 a',
        '.lead-story a',
    ],
    "economictimes.indiatimes.com": [
        'a',
        'article a',
        '.eachStory a',
        'h2 a', 'h3 a',
    ],
    "www.ndtv.com": [
        'a',
        'article a',
        '.story-list a',
        '.story-content a',
        'h2 a', 'h3 a',
        '.main-story a',
        '.story_content a',
    ],
    "indianexpress.com": [
        'a',
        'article a',
        '.articles a',
        'h2 a', 'h3 a',
    ],
    "www.livemint.com": [
        'a',
        'article a',
        '.story a',
        'h2 a', 'h3 a',
    ],
    "www.hindustantimes.com": [
        'a',
        'article a',
        '.story-listing a',
        'h2 a', 'h3 a',
    ],
    "scroll.in": [
        'a',
        'article a',
        '.story-card a',
        'h2 a', 'h3 a',
    ],
    "www.business-standard.com": [
        'a',
        'article a',
        '.story-list a',
        'h2 a', 'h3 a',
    ],
    "www.deccanherald.com": [
        'a',
        'article a',
        '.story a',
        'h2 a', 'h3 a',
    ],
    "www.news18.com": [
        'a',
        'article a',
        '.story-listing a',
        'h2 a', 'h3 a',
    ],
}

IMPORTANT_URL_TAGS = {
    # Political and governance
    "/politics/", "/political/", "/government/", "/parliament/", "/election/", "/policy/",
    "/minister/", "/cabinet/", "/opposition/", "/party/", "/campaign/", "/vote/", "/voting/",
    
    # Geographic and local news
    "/city/", "/cities/", "/state/", "/states/", "/national/", "/india/", "/delhi/", 
    "/mumbai/", "/bangalore/", "/chennai/", "/kolkata/", "/hyderabad/", "/pune/",
    "/ahmedabad/", "/lucknow/", "/jaipur/", "/chandigarh/", "/region/", "/regional/",
    
    # Content type indicators
    "/article/", "/news/", "/story/", "/report/", "/analysis/", "/opinion/", "/editorial/",
    "/interview/", "/exclusive/", "/investigation/", "/probe/", "/expose/",
    
    # International and world affairs
    "/world/", "/international/", "/global/", "/foreign/", "/diplomacy/", "/bilateral/",
    "/multilateral/", "/asia/", "/china/", "/pakistan/", "/usa/", "/europe/", "/africa/",
    
    # Economic and business
    "/economy/", "/economic/", "/business/", "/finance/", "/financial/", "/market/", 
    "/markets/", "/industry/", "/companies/", "/corporate/", "/banking/", "/trade/",
    "/budget/", "/gdp/", "/inflation/", "/investment/", "/startup/", "/technology/",
    
    # Social issues and justice
    "/social/", "/society/", "/justice/", "/law/", "/legal/", "/court/", "/rights/",
    "/women/", "/gender/", "/minority/", "/caste/", "/reservation/", "/equality/",
    "/discrimination/", "/harassment/", "/violence/", "/crime/", "/corruption/",
    
    # Environment and sustainability
    "/environment/", "/climate/", "/pollution/", "/energy/", "/renewable/", "/carbon/",
    "/sustainability/", "/conservation/", "/wildlife/", "/forest/", "/water/",
    
    # Education and research
    "/education/", "/university/", "/research/", "/science/", "/scientific/", "/study/",
    "/academic/", "/scholar/", "/innovation/", "/discovery/", "/technology/",
    
    # Health and public welfare
    "/health/", "/healthcare/", "/medical/", "/hospital/", "/medicine/", "/pandemic/",
    "/vaccine/", "/disease/", "/public-health/", "/welfare/", "/scheme/", "/program/",
    
    # Security and defense
    "/security/", "/defense/", "/military/", "/army/", "/navy/", "/airforce/", "/border/",
    "/terrorism/", "/cyber/", "/cybersecurity/", "/intelligence/", "/surveillance/",
    
    # Infrastructure and development
    "/infrastructure/", "/development/", "/urban/", "/rural/", "/transport/", "/railway/",
    "/highway/", "/airport/", "/port/", "/construction/", "/housing/", "/smart-city/",
}

# URL tags that indicate less substantial content (to deprioritize)
DEPRIORITY_URL_TAGS = {
    "/entertainment/", "/bollywood/", "/hollywood/", "/celebrity/", "/gossip/", "/movies/",
    "/sports/", "/cricket/", "/football/", "/tennis/", "/olympics/", "/match/", "/score/",
    "/lifestyle/", "/fashion/", "/beauty/", "/style/", "/trends/", "/trending/",
    "/food/", "/recipe/", "/cooking/", "/restaurant/", "/cuisine/", "/dining/",
    "/travel/", "/tourism/", "/destination/", "/vacation/", "/hotel/", "/flight/",
    "/auto/", "/cars/", "/bike/", "/vehicle/", "/gadgets/", "/mobile/", "/smartphone/",
    "/photos/", "/photo/", "/gallery/", "/pics/", "/images/", "/slideshow/",
    "/videos/", "/video/", "/watch/", "/live/", "/streaming/", "/youtube/",
    "/astrology/", "/horoscope/", "/zodiac/", "/predictions/", "/fortune/",
    "/tips/", "/hacks/", "/guide/", "/tutorial/", "/how-to/", "/diy/", "/listicle/",
}

# Curated Indian news homepages to crawl (name, url)
NEWS_SITES: List[Tuple[str, str]] = [
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
]

# Additional section URLs for sites where homepage has navigation instead of articles
SITE_SECTION_URLS = {
    "ndtv.com": [
        "https://www.ndtv.com/india-news",     # NDTV uses /india-news/ structure
        "https://www.ndtv.com/world-news",     # NDTV uses /world-news/ structure  
        "https://www.ndtv.com/business",       # NDTV business section
        "https://www.ndtv.com/opinion",        # NDTV opinion pieces
        "https://www.ndtv.com/offbeat",        # NDTV offbeat stories
    ],
    "thehindu.com": [
        "https://www.thehindu.com/news/national/",
        "https://www.thehindu.com/news/international/", 
        "https://www.thehindu.com/business/",
        "https://www.thehindu.com/sci-tech/",
        "https://www.thehindu.com/opinion/",
    ],
    "timesofindia.indiatimes.com": [
        "https://timesofindia.indiatimes.com/india/",
        "https://timesofindia.indiatimes.com/world/",
        "https://timesofindia.indiatimes.com/business/",
    ],
    "indiatoday.in": [
        "https://www.indiatoday.in/india/",
        "https://www.indiatoday.in/world/",
        "https://www.indiatoday.in/business/",
        "https://www.indiatoday.in/technology/",
        "https://www.indiatoday.in/education/",
        "https://www.indiatoday.in/science/",
    ],
}

# Exclude obvious non-article areas
EXCLUDED_PATH_KEYWORDS = (
    "/photos/", "/photo/", "/pictures/", "/gallery/", "/galleries/",
    "/videos/", "/video/", "/live-tv/", "/livetv/", "/live-tv", "/subscribe", "/privacy",
    "/terms", "/about", "/contact", "/advertise", "/horoscope/", "/astrology/",
    "/lifestyle/", "/tips/", "/fashion/", "/beauty/", "/travel/", "/food/", "/recipe/",
    "/entertainment/bollywood/", "/celebrity/", "/gossip/", "/quiz/", "/games/",
)

# Enhanced positive hints for substantial content
ARTICLE_POSITIVE_HINTS = (
    "news", "india", "world", "business", "economy", "tech", "technology", "politics",
    "nation", "city", "opinion", "science", "health", "environment", "policy", "government",
    "parliament", "election", "international", "global", "foreign", "diplomacy",
    "analysis", "investigation", "report", "research", "study", "survey",
    "corruption", "scam", "controversy", "debate", "protest", "movement",
    "climate", "energy", "agriculture", "infrastructure", "education", "healthcare",
    "security", "defense", "military", "terrorism", "cyber", "digital",
    "employment", "unemployment", "jobs", "labour", "workforce", "industry",
    "manufacturing", "startup", "innovation", "ai", "artificial intelligence",
    "blockchain", "cryptocurrency", "data", "privacy", "regulation",
    "gender", "women", "rights", "equality", "justice", "law", "court", "legal",
    "religion", "communal", "secular", "minority", "caste", "reservation",
    "poverty", "inequality", "welfare", "scheme", "budget", "tax", "finance",
    "banking", "market", "stock", "economy", "gdp", "inflation", "trade",
)

DISALLOWED_HOST_SUFFIXES = (
    "google.com", "gstatic.com", "youtube.com", "twitter.com", "t.co", "facebook.com",
    "instagram.com", "linkedin.com", "pinterest.com", "reddit.com"
)

# Topic filters: focus-only and exclude lists
ALLOWED_TOPIC_KEYWORDS = (
    # core focus areas (match stems where useful)
    "politic", "government", "policy", "parliament", "election", "minister", "cabinet",
    "world", "international", "global", "foreign", "diplomacy", "geopolitic",
    "culture", "society", "heritage", "community", "social",
    "ideolog", "religion", "communal", "secular", "minority", "caste",
    "tech", "technology", "ai", "cyber", "digital", "innovation", "startup", "data",
    "gender", "women", "lgbt", "queer", "rights", "equality", "justice",
    "employment", "job", "jobs", "labour", "labor", "workforce", "unemployment", "hiring",
    "economy", "economic", "finance", "financial", "budget", "tax", "market", "trade",
    "environment", "climate", "energy", "pollution", "sustainability",
    "health", "healthcare", "medical", "pandemic", "disease", "public health",
    "education", "university", "research", "science", "scientific",
    "infrastructure", "transport", "agriculture", "farming", "rural",
    "security", "defense", "military", "terrorism", "cyber", "crime", "law", "legal", "court",
    "corruption", "scam", "investigation", "probe", "controversy", "scandal",
    "protest", "movement", "strike", "demonstration", "activism",
)

EXCLUDED_TOPIC_KEYWORDS = (
    # Fluff and non-substantive content
    "astrology", "horoscope", "zodiac", "vastu", "numerology", "palmistry",
    "learning", "tips", "life hack", "lifehack", "lifestyle", "fashion", "beauty",
    "recipe", "cooking", "food", "travel", "tourism", "vacation",
    "celebrity", "bollywood", "gossip", "entertainment", "movie", "film",
    "cricket", "sports", "match", "game", "quiz", "puzzle",
    "relationship", "dating", "love", "marriage", "wedding",
    "shopping", "deal", "offer", "discount", "product review",
    "fitness", "workout", "diet", "weight loss", "skincare",
    "festival", "celebration", "party", "event", "cultural program",
)

# --------------- Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------- Helpers ------------------

def get_db_collection():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI not set in .env")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    logging.info("Connected to MongoDB.")
    return db[COLLECTION_NAME]


def get_existing_urls(collection) -> set:
    urls = {doc['url'] for doc in collection.find({}, {"url": 1})}
    logging.info(f"Existing URLs in DB: {len(urls)}")
    return urls


def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        # Drop tracking and nav params
        drop_params = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "pfrom", "ocid"}
        if p.query:
            q = "&".join(k for k in p.query.split("&") if k and k.split("=")[0] not in drop_params)
        else:
            q = ""
        return p._replace(query=q, fragment="").geturl()
    except Exception:
        return url


def domain_key_for_host(host: str) -> str:
    host = (host or "").lower()
    # Handle Times of India subdomain
    if "timesofindia" in host:
        return "timesofindia.indiatimes.com"
    for key in SITE_ARTICLE_PATTERNS.keys():
        if host == key or host.endswith("." + key):
            return key
    return host


def get_url_importance_score(url: str) -> int:
    """Score URL based on important tags. Higher score = more important content."""
    try:
        path = (urlparse(url).path or '').lower()
        
        # Count important tags (positive score)
        important_score = sum(1 for tag in IMPORTANT_URL_TAGS if tag in path)
        
        # Count depriority tags (negative score)
        depriority_score = sum(1 for tag in DEPRIORITY_URL_TAGS if tag in path)
        
        # Additional boost for multiple important tags
        if important_score >= 2:
            important_score += 1
            
        # Penalty for depriority content
        final_score = important_score - (depriority_score * 2)
        
        return max(0, final_score)  # Don't go negative
    except Exception:
        return 0


def is_excluded_by_site_pattern(url: str, domain_key: str) -> bool:
    """Check if URL matches site-specific exclusion patterns."""
    try:
        exclusions = SITE_EXCLUSION_PATTERNS.get(domain_key, [])
        path = (urlparse(url).path or '').lower()
        return any(re.search(pattern, path) for pattern in exclusions)
    except Exception:
        return False


def matches_site_quality_indicators(url: str, headline: str, domain_key: str) -> bool:
    """Check if content matches site-specific quality indicators."""
    try:
        indicators = SITE_QUALITY_INDICATORS.get(domain_key, {})
        if not indicators:
            return True  # No specific rules, allow through
        
        path = (urlparse(url).path or '').lower()
        headline_lower = (headline or '').lower()
        
        # Check if in quality sections
        quality_sections = indicators.get("sections", [])
        in_quality_section = any(section in path for section in quality_sections)
        
        # Check for quality words in headline
        quality_words = indicators.get("quality_words", [])
        has_quality_words = any(word in headline_lower for word in quality_words)
        
        # Check for words to avoid in headlines
        avoid_headlines = indicators.get("avoid_headlines", [])
        has_avoid_words = any(word in headline_lower for word in avoid_headlines)
        
        # Scoring: need quality section OR quality words, but avoid bad headlines
        if has_avoid_words:
            return False
        
        return in_quality_section or has_quality_words
    except Exception:
        return True


def matches_any(patterns: List[str], path: str) -> bool:
    for pat in patterns:
        if re.search(pat, path):
            return True
    return False


def is_article_url(url: str, allowed_domain: str) -> bool:
    try:
        url = normalize_url(url)
        parsed = urlparse(url)
        host = (parsed.hostname or '').lower()
        path = (parsed.path or '').lower()
        if not host or not path or path == "/":
            return False
        if any(host == s or host.endswith("." + s) for s in DISALLOWED_HOST_SUFFIXES):
            return False
        # Must be same domain (or subdomain) as allowed_domain
        allowed = allowed_domain.lower()
        if not (host == allowed or host.endswith("." + allowed)):
            return False
        if any(k in path for k in EXCLUDED_PATH_KEYWORDS):
            return False
        
        # Get domain key for site-specific rules
        domain_key = domain_key_for_host(host)
        
        # Check site-specific exclusions first (these override everything)
        if is_excluded_by_site_pattern(url, domain_key):
            return False
        
        # Section pages: very shallow paths without article patterns
        segments = [seg for seg in path.split('/') if seg]
        if len(segments) <= 2 and not any(h in path for h in ("/article", "/story", "/news/")):
            return False
        
        # Site-specific pattern match
        patterns = SITE_ARTICLE_PATTERNS.get(domain_key)
        if patterns and matches_any(patterns, path):
            return True
        # Generic fallbacks
        if re.search(r"/\d{4}/\d{2}/\d{2}/", path):
            return True
        if re.search(r"-\d{4,}$", path):  # slug-1234567
            return True
        if path.endswith(".cms") or path.endswith(".html") or path.endswith(".htm"):
            # require at least 3 segments and some hyphens to avoid sections
            return len(segments) >= 3 and any('-' in s for s in segments)
        return False
    except Exception:
        return False


def is_desired_topic(url: str, headline: str = "") -> bool:
    """Return True if URL/headline matches focus topics and avoids excluded ones."""
    try:
        u = url.lower()
        h = (headline or "").lower()

        # Hard excludes first - these override everything
        if any(tok in u or tok in h for tok in EXCLUDED_TOPIC_KEYWORDS):
            return False

        # Check for substantive headline indicators
        substantive_words = {
            "investigation", "probe", "analysis", "report", "survey", "study",
            "controversy", "scandal", "corruption", "scam", "expose",
            "policy", "reform", "legislation", "amendment", "regulation",
            "crisis", "emergency", "urgent", "critical", "breaking",
            "impact", "effect", "consequence", "result", "outcome",
            "challenge", "problem", "issue", "concern", "threat",
            "solution", "initiative", "program", "scheme", "project",
            "development", "progress", "achievement", "success", "failure",
            "decision", "announcement", "statement", "declaration",
        }
        
        # Boost score for substantive content
        substantive_score = sum(1 for word in substantive_words if word in h)
        
        # Require presence of at least one allowed token in URL path or headline
        allowed_matches = sum(1 for tok in ALLOWED_TOPIC_KEYWORDS if tok in u or tok in h)
        
        if allowed_matches > 0:
            # If we have substantive words, accept with lower threshold
            if substantive_score >= 1:
                return True
            # Otherwise require stronger match
            elif allowed_matches >= 2:
                return True

        # Also look for common section slugs indicating focus areas
        focus_slugs = (
            "/politics/", "/world/", "/international/", "/global/", "/nation/",
            "/culture/", "/society/", "/technology/", "/tech/", "/science/",
            "/gender/", "/women/", "/jobs/", "/employment/", "/labour/", "/labor/",
            "/economy/", "/business/", "/finance/", "/market/", "/industry/",
            "/environment/", "/climate/", "/energy/", "/health/", "/education/",
            "/law/", "/legal/", "/court/", "/security/", "/defense/", "/policy/",
        )
        if any(slug in u for slug in focus_slugs):
            # Even with slug match, avoid if headline suggests fluff
            fluff_indicators = {"tips", "hack", "secrets", "tricks", "guide to"}
            if not any(indicator in h for indicator in fluff_indicators):
                return True

        return False
    except Exception:
        return False


def collect_links_for_site(page, site_url: str, allowed_domain: str) -> List[Dict[str, str]]:
    # Wait for anchors; many sites render quickly but add a short wait for safety
    try:
        page.wait_for_selector('a', timeout=WAIT_SELECTOR_TIMEOUT)
    except Exception:
        logging.warning(f"No anchors detected on {site_url}. Dumping HTML.")
        with open(f"debug_{allowed_domain}.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        return []

    # Wait a bit longer for dynamic content to load
    page.wait_for_timeout(3000)  # 3 seconds for dynamic content

    # Get domain-specific selectors for better anchor detection
    domain_key = domain_key_for_host(allowed_domain)
    site_selectors = SITE_SELECTORS.get(domain_key, ['a'])
    
    print(f"\n=== Debug for {site_url} ===")
    print(f"Domain key: {domain_key}")
    print(f"Using selectors: {site_selectors}")
    
    # Debug: Check total anchor count first
    total_anchors = page.evaluate('() => document.querySelectorAll("a").length')
    print(f"Total anchors on page: {total_anchors}")
    
    # Debug: Test each selector individually
    for selector in site_selectors:
        try:
            # Properly escape the selector for JavaScript
            escaped_selector = selector.replace('"', '\\"').replace("'", "\\'")
            count = page.evaluate(f'() => document.querySelectorAll("{escaped_selector}").length')
            print(f"  Selector '{selector}': {count} elements")
        except Exception as e:
            print(f"  Selector '{selector}': ERROR - {e}")
    
    anchors: List[Dict[str, str]] = page.evaluate('''(params) => {
        const { base, selectors } = params;
        const toAbs = (href) => {
            try { return new URL(href, base).toString(); } catch { return null; }
        };
        
        let nodes = [];
        let debug_info = [];
        
        // Try site-specific selectors first
        for (const selector of selectors) {
            try {
                const selectorNodes = Array.from(document.querySelectorAll(selector));
                debug_info.push(`${selector}: ${selectorNodes.length} nodes`);
                nodes = nodes.concat(selectorNodes);
            } catch (e) {
                debug_info.push(`${selector}: ERROR - ${e.message}`);
            }
        }
        
        console.log('Selector results:', debug_info);
        
        // If no site-specific selectors worked, fall back to generic
        if (nodes.length === 0) {
            nodes = Array.from(document.querySelectorAll('a'));
            console.log('Fallback to generic "a" selector:', nodes.length);
        }
        
        const out = [];
        const seen = new Set();
        
        for (const a of nodes) {
            const href = a.getAttribute('href');
            if (!href) continue;
            if (href.startsWith('javascript:') || href.startsWith('#')) continue;
            
            const abs = toAbs(href);
            if (!abs || seen.has(abs)) continue;
            seen.add(abs);
            
            const text = (a.getAttribute('aria-label') || a.textContent || '').trim();
            out.push({ url: abs, text });
        }
        
        console.log('Final anchor results:', out.length);
        return out;
    }''', {"base": site_url, "selectors": site_selectors})

    # Debug: Sample some URLs to understand patterns - show more for analysis
    if len(anchors) > 0:
        print(f"=== Sample URLs for {domain_key} (first 15) ===")
        for i, anchor in enumerate(anchors[:15]):
            url = anchor['url']
            text = anchor['text'][:60] + "..." if len(anchor['text']) > 60 else anchor['text']
            print(f"  {i+1}: {url}")
            print(f"      Text: {text}")
        print("=== End sample ===")

    filtered = []
    seen = set()
    scored_candidates = []  # Store candidates with scores for sorting
    
    for a in anchors:
        url = normalize_url(a['url'])
        if url in seen:
            continue
        seen.add(url)
        headline = a.get('text') or ''
        
        # First check if it's a valid article URL structure
        if not is_article_url(url, allowed_domain):
            continue
            
        # Then check topic relevance
        if not is_desired_topic(url, headline):
            continue
            
        # Check site-specific quality indicators
        domain_key = domain_key_for_host(urlparse(url).hostname or '')
        if not matches_site_quality_indicators(url, headline, domain_key):
            continue
            
        # Calculate importance score based on URL tags
        importance_score = get_url_importance_score(url)
        
        scored_candidates.append({
            'url': url, 
            'headline': headline, 
            'score': importance_score
        })
    
    # Sort by importance score (descending) and take the best ones
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Convert back to the expected format, prioritizing high-scoring URLs
    for candidate in scored_candidates:
        filtered.append({
            'url': candidate['url'], 
            'headline': candidate['headline']
        })
            
    return filtered


def is_quality_content(content: str, headline: str = "") -> bool:
    """Assess if extracted content represents substantial journalism."""
    if not content or len(content) < MIN_CONTENT_LENGTH:
        return False
    
    content_lower = content.lower()
    headline_lower = (headline or "").lower()
    
    # Quality indicators
    quality_indicators = {
        "according to", "sources said", "official", "government", "ministry",
        "data shows", "report", "study", "survey", "research", "analysis",
        "investigation", "probe", "findings", "evidence", "document",
        "statement", "announcement", "press conference", "interview",
        "expert", "analyst", "economist", "researcher", "professor",
        "impact", "effect", "consequence", "result", "statistics",
        "policy", "regulation", "law", "legislation", "amendment",
        "budget", "allocation", "fund", "investment", "expenditure",
        "crisis", "emergency", "situation", "development", "progress",
    }
    
    # Count quality indicators in content
    quality_score = sum(1 for indicator in quality_indicators if indicator in content_lower)
    
    # Fluff indicators that suggest listicles, tips, or shallow content
    fluff_indicators = {
        "here are", "here's how", "tips to", "ways to", "secrets of",
        "tricks", "hacks", "guide", "tutorial", "step by step",
        "you should", "you must", "you can", "you need",
        "amazing", "incredible", "unbelievable", "shocking facts",
        "top 10", "best ways", "easiest way", "simple steps",
    }
    
    fluff_score = sum(1 for indicator in fluff_indicators if indicator in content_lower or indicator in headline_lower)
    
    # Require meaningful quality content and minimal fluff
    return quality_score >= 3 and fluff_score <= 1


def scrape_article(browser, url: str) -> Tuple[str, str]:
    """Returns (final_url, content) or (final_url, '') on failure."""
    page = browser.new_page()
    try:
        page.goto(url, timeout=NAV_TIMEOUT, wait_until='domcontentloaded')
        final_url = page.url
        html = page.content()
        content = trafilatura.extract(html) or ''
        return final_url, content
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
        return url, ''
    finally:
        page.close()


# --------------- Main Loop ----------------

def main():
    collection = get_db_collection()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        while True:
            existing = get_existing_urls(collection)
            logging.info("Starting Indian news sites scraping cycle...")

            for name, site_url in NEWS_SITES:
                allowed_domain = (urlparse(site_url).hostname or '').lower()
                if not allowed_domain:
                    continue

                logging.info(f"\n[{name}] Visiting {site_url}")
                page = browser.new_page()
                all_candidates = []  # Collect from homepage + section pages
                
                try:
                    # 1. Process homepage
                    page.goto(site_url, timeout=NAV_TIMEOUT)
                    page.wait_for_load_state('domcontentloaded', timeout=WAIT_SELECTOR_TIMEOUT)
                    time.sleep(2)

                    homepage_candidates = collect_links_for_site(page, site_url, allowed_domain)
                    all_candidates.extend(homepage_candidates)
                    logging.info(f"[{name}] Homepage candidates: {len(homepage_candidates)}")

                    # 2. INTRUSIVE: Also process section pages for sites that need it
                    domain_key = domain_key_for_host(allowed_domain)
                    if domain_key in SITE_SECTION_URLS:
                        logging.info(f"[{name}] Processing {len(SITE_SECTION_URLS[domain_key])} section pages...")
                        for section_url in SITE_SECTION_URLS[domain_key]:
                            try:
                                page.goto(section_url, timeout=NAV_TIMEOUT)
                                page.wait_for_load_state('domcontentloaded', timeout=WAIT_SELECTOR_TIMEOUT)
                                time.sleep(1)
                                
                                section_candidates = collect_links_for_site(page, section_url, allowed_domain)
                                all_candidates.extend(section_candidates)
                                logging.info(f"[{name}] Section {section_url.split('/')[-2]} added {len(section_candidates)} candidates")
                                
                            except Exception as e:
                                logging.warning(f"[{name}] Failed to process section {section_url}: {e}")
                                continue

                    candidates = all_candidates
                    logging.info(f"[{name}] Total candidates after domain filter: {len(candidates)}")

                    # Dedupe against DB and cap per site
                    to_process = []
                    for c in candidates:
                        if len(to_process) >= PER_SITE_LIMIT:
                            break
                        # We'll resolve final URL after navigation, but skip clear dupes now
                        if c['url'] not in existing:
                            to_process.append(c)

                    logging.info(f"[{name}] Selected {len(to_process)} for scraping (cap {PER_SITE_LIMIT}).")

                    added = 0
                    for c in to_process:
                        initial_url = c['url']
                        headline = c['headline']
                        importance_score = get_url_importance_score(initial_url)
                        logging.info(f"[{name}] Scraping (score:{importance_score}): {headline[:80]} -> {initial_url}")
                        final_url, content = scrape_article(browser, initial_url)

                        # Final URL must still be a valid article on the allowed domain
                        if not is_article_url(final_url, allowed_domain):
                            logging.info(f"[{name}] Skipped: Final URL not recognized as article on {allowed_domain}")
                            continue

                        # Enforce topic focus on final URL/headline too
                        if not is_desired_topic(final_url, headline):
                            logging.info(f"[{name}] Skipped: Outside desired topics")
                            continue
                            
                        # Final site-specific quality check after redirect
                        final_domain_key = domain_key_for_host(urlparse(final_url).hostname or '')
                        if not matches_site_quality_indicators(final_url, headline, final_domain_key):
                            logging.info(f"[{name}] Skipped: Fails site-specific quality check")
                            continue

                        if final_url in existing:
                            logging.info(f"[{name}] Skipped: Already in DB")
                            continue

                        # Quality check: ensure content is substantial journalism
                        if not is_quality_content(content, headline):
                            logging.info(f"[{name}] Skipped: Content quality insufficient")
                            continue

                        if content and len(content) >= MIN_CONTENT_LENGTH:
                            collection.update_one(
                                { 'url': final_url },
                                { '$setOnInsert': {
                                    'text': content,
                                    'source': name,
                                    'headline': headline,
                                    'site': allowed_domain,
                                } },
                                upsert=True
                            )
                            existing.add(final_url)
                            added += 1
                            logging.info(f"[{name}] Saved.")
                        else:
                            logging.info(f"[{name}] Skipped: Content too short/failed.")

                    logging.info(f"[{name}] Added {added} new articles this cycle.")

                except Exception as e:
                    logging.error(f"[{name}] Error during site scrape: {e}")
                    all_candidates = []  # Reset to empty if there's an error
                finally:
                    try:
                        page.close()
                    except:
                        pass

            # Process problematic sites with modular rules
            if MODULAR_RULES_AVAILABLE:
                logging.info("\n" + "="*60)
                logging.info("PROCESSING PROBLEMATIC SITES WITH MODULAR RULES")
                logging.info("="*60)
                
                # Convert NEWS_SITES to format expected by modular processing
                problematic_sites = []
                for name, site_url in NEWS_SITES:
                    allowed_domain = (urlparse(site_url).hostname or '').lower()
                    if allowed_domain:
                        problematic_sites.append((name, site_url, allowed_domain))
                
                process_problematic_sites_with_modular_rules(browser, problematic_sites, existing, collection)

            logging.info(f"Cycle complete. Sleeping for {SLEEP_TIME_SECONDS/60:.1f} minutes...")
            try:
                time.sleep(SLEEP_TIME_SECONDS)
            except KeyboardInterrupt:
                logging.info("Interrupted by user. Exiting.")
                break

        browser.close()


def collect_links_with_modular_rules(page, site_url: str, domain_key: str) -> List[Dict[str, str]]:
    """
    Enhanced link collection using modular rules for problematic sites
    """
    if not MODULAR_RULES_AVAILABLE:
        logging.warning("Modular rules not available, falling back to standard collection")
        return []
    
    all_links = []
    
    # Get content-specific selectors for this domain
    selectors = get_content_selectors(domain_key)
    
    logging.info(f"Using modular selectors for {domain_key}: {selectors}")
    
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
                        full_url = urljoin(site_url, href)
                        
                        all_links.append({
                            'url': full_url,
                            'text': text,
                            'selector': selector
                        })
                        
                except Exception:
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
    
    # Apply modular filtering
    logging.info(f"=== ANALYZING LINKS FOR {domain_key.upper()} ===")
    
    # Step 1: Remove external domains
    internal_links = []
    for link in unique_links:
        link_domain = get_domain_key(link['url'])
        if domain_key in link_domain or link_domain in domain_key:
            internal_links.append(link)
    
    logging.info(f"Step 1 - Internal links: {len(internal_links)}/{len(unique_links)}")
    
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
    logging.info(f"=== SAMPLE VALID ARTICLES (first 10) ===")
    for i, link in enumerate(article_links[:10]):
        logging.info(f"  {i+1}. {link['url']}")
        logging.info(f"      Text: {link['text'][:80]}...")
        logging.info(f"      Selector: {link['selector']}")
    
    # Convert to format expected by main scraper
    formatted_links = []
    for link in article_links:
        formatted_links.append({
            'url': link['url'],
            'text': link['text']
        })
    
    return formatted_links


def process_problematic_sites_with_modular_rules(browser, problematic_sites: List[Tuple[str, str, str]], existing: set, collection):
    """
    Process problematic sites using modular rules and section targeting
    """
    if not MODULAR_RULES_AVAILABLE:
        logging.warning("Modular rules not available, skipping problematic sites processing")
        return
    
    # Sites that benefit from modular rules
    MODULAR_SITES = {
        'ndtv.com', 'timesofindia.indiatimes.com', 
        'economictimes.indiatimes.com', 'news18.com'
    }
    
    for name, site_url, allowed_domain in problematic_sites:
        domain_key = get_domain_key(site_url)
        
        if domain_key not in MODULAR_SITES:
            continue
            
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING {name.upper()} WITH MODULAR RULES ({domain_key})")
        logging.info(f"{'='*60}")
        
        page = browser.new_page()
        all_candidates = []
        
        try:
            # List of URLs to visit (homepage + priority sections)
            urls_to_visit = [site_url] + get_priority_sections(domain_key)
            
            for i, url in enumerate(urls_to_visit):
                page_type = "Homepage" if i == 0 else f"Section {i}"
                logging.info(f"\n--- {page_type}: {url} ---")
                
                try:
                    # Navigate to page
                    page.goto(url, timeout=60000)
                    page.wait_for_load_state('domcontentloaded', timeout=15000)
                    time.sleep(3)  # Wait for dynamic content
                    
                    # Extract links using modular rules
                    filtered_links = collect_links_with_modular_rules(page, url, domain_key)
                    
                    all_candidates.extend(filtered_links)
                    logging.info(f"{page_type} contributed {len(filtered_links)} candidates")
                    
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")
            
            # Final deduplication
            seen = set()
            unique_candidates = []
            for candidate in all_candidates:
                if candidate['url'] not in seen:
                    seen.add(candidate['url'])
                    unique_candidates.append(candidate)
            
            logging.info(f"\n{name} MODULAR SUMMARY:")
            logging.info(f"  Total unique candidates: {len(unique_candidates)}")
            
            # Filter against database
            new_candidates = [c for c in unique_candidates if c['url'] not in existing]
            logging.info(f"  New candidates (not in DB): {len(new_candidates)}")
            
            # Process the best candidates (limit for performance)
            to_process = new_candidates[:25]  # Limit to top 25
            
            added = 0
            for candidate in to_process:
                initial_url = candidate['url']
                headline = candidate['text']
                
                logging.info(f"[{name}] Modular scraping: {headline[:60]}... -> {initial_url}")
                final_url, content = scrape_article(browser, initial_url)
                
                if final_url and content and len(content) > 500:
                    try:
                        collection.update_one(
                            {'url': final_url},
                            { '$set': {
                                'url': final_url,
                                'content': content,
                                'collected_at': datetime.now(),
                                'headline': headline,
                                'site': allowed_domain,
                            } },
                            upsert=True
                        )
                        existing.add(final_url)
                        added += 1
                        logging.info(f"[{name}] Modular saved.")
                    except Exception as e:
                        logging.error(f"[{name}] Database error: {e}")
                else:
                    logging.info(f"[{name}] Modular skipped: Content insufficient")
            
            logging.info(f"[{name}] Added {added} new articles via modular rules.")
            
        except Exception as e:
            logging.error(f"[{name}] Modular processing error: {e}")
        finally:
            try:
                page.close()
            except:
                pass


if __name__ == "__main__":
    main()

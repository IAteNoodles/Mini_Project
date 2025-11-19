"""
Modular site-specific rules for Indian news scraping
Based on actual site structure analysis
"""

import re
from urllib.parse import urlparse

# ==================== SITE PATTERNS ====================

# Article URL patterns that we want to capture
SITE_ARTICLE_PATTERNS = {
    "ndtv.com": [
        # Real NDTV article patterns observed in browser
        r"/india-news/[^/]+-\d{7,}",      # /india-news/article-title-1234567
        r"/world-news/[^/]+-\d{7,}",      # /world-news/article-title-1234567
        r"/business/[^/]+-\d{7,}",        # /business/article-title-1234567
        r"/cities/[^/]+/[^/]+-\d{7,}",    # /cities/delhi/article-title-1234567
        r"/opinion/[^/]+-\d{7,}",         # /opinion/article-title-1234567
        r"/offbeat/[^/]+-\d{7,}",         # /offbeat/article-title-1234567
        r"/feature/[^/]+-\d{7,}",         # /feature/article-title-1234567
    ],
    "timesofindia.indiatimes.com": [
        # Times of India patterns
        r"/articleshow/\d+\.cms$",        # /articleshow/123456.cms
        r"/city/[^/]+/[^/]+/\d+\.cms",   # /city/delhi/article/123456.cms
        r"/india/[^/]+/\d+\.cms",        # /india/article/123456.cms
        r"/world/[^/]+/\d+\.cms",        # /world/article/123456.cms
        r"/business/[^/]+/\d+\.cms",     # /business/article/123456.cms
        r"/blogs/[^/]+/\d+\.cms",        # /blogs/author/123456.cms
    ],
    "economictimes.indiatimes.com": [
        # Economic Times patterns
        r"/news/[^/]+/[^/]+/\d+\.cms",   # /news/economy/policy/123456.cms
        r"/industry/[^/]+/\d+\.cms",     # /industry/banking/123456.cms
        r"/markets/[^/]+/\d+\.cms",      # /markets/stocks/123456.cms
        r"/politics-nation/\d+\.cms",    # /politics-nation/123456.cms
        r"/et-explains/\d+\.cms",        # /et-explains/123456.cms
        r"/jobs/\d+\.cms",               # /jobs/123456.cms
    ],
    "news18.com": [
        # News18 patterns  
        r"/india/[^/]+-\d+\.html",       # /india/article-title-123456.html
        r"/world/[^/]+-\d+\.html",       # /world/article-title-123456.html
        r"/business/[^/]+/[^/]+-\d+\.html", # /business/economy/article-123456.html
        r"/politics/[^/]+-\d+\.html",    # /politics/article-title-123456.html
        r"/explainers/[^/]+-\d+\.html",  # /explainers/article-title-123456.html
    ],
}

# ==================== DOMAIN FILTERS ====================

# Links that should be excluded (external domains, fluff content)
DOMAIN_EXCLUSIONS = {
    "ndtv.com": [
        # External domains
        r"ndtvprofit\.com",
        r"sports\.ndtv\.com", 
        r"food\.ndtv\.com",
        r"doctor\.ndtv\.com",
        r"gadgets360\.com",
        r"ndtvgames\.com",
        r"ndtvshopping\.com",
        r"rajasthan\.ndtv\.in",
        r"hindi\.ndtv\.com",
        # Fluff content within domain
        r"/entertainment",
        r"/movies",
        r"/lifestyle", 
        r"/cricket",
        r"/sports",
        r"/food",
        r"/auto",
        r"/trends",
        r"/photos",
        r"/videos",
        r"/webstories",
        r"/apps",
    ],
    "timesofindia.indiatimes.com": [
        # External domains
        r"hindi\.indiatimes\.com",
        r"navbharattimes\.indiatimes\.com", 
        r"maharashtratimes\.com",
        r"vijaykarnataka\.com",
        r"eisamay\.com",
        # Fluff content
        r"/entertainment",
        r"/sports", 
        r"/lifestyle",
        r"/travel",
        r"/food",
        r"/auto",
        r"/trending",
        r"/astrology",
        r"/movies",
        r"/tv",
        r"/web-series",
        r"/photos",
        r"/videos",
    ],
    "economictimes.indiatimes.com": [
        # Keep focused on economics/business/politics
        # Exclude entertainment/lifestyle
        r"/panache",
        r"/magazines",
        r"/slideshows", 
        r"/videos",
        r"/photos",
    ],
    "news18.com": [
        # External language sites
        r"hindi\.news18\.com",
        r"bengali\.news18\.com",
        r"news18marathi\.com",
        r"gujarati\.news18\.com",
        r"kannada\.news18\.com",
        r"tamil\.news18\.com",
        r"malayalam\.news18\.com",
        r"telugu\.news18\.com",
        r"punjab\.news18\.com",
        r"urdu\.news18\.com",
        r"assam\.news18\.com",
        r"odia\.news18\.com",
        # Fluff content
        r"/entertainment",
        r"/sports",
        r"/lifestyle", 
        r"/movies",
        r"/cricket",
        r"/photogallery",
        r"/videos",
        r"/buzz",
    ],
}

# ==================== CONTENT SELECTORS ====================

# CSS selectors for finding actual article links (not navigation)
CONTENT_SELECTORS = {
    "ndtv.com": [
        ".story_content a",      # Main story content
        ".ins_storybody a",      # Inside story body
        ".story-list a",         # Story listing
        ".lead-story a",         # Lead stories
        "h2 a",                  # Headlines
        "h3 a",                  # Sub-headlines
        ".main-story a",         # Main story links
        "article a",             # Article elements
    ],
    "timesofindia.indiatimes.com": [
        ".storylist a",          # Story listings
        ".story-content a",      # Story content
        ".top-newslist a",       # Top news
        ".newscont a",           # News content
        "h2 a",                  # Headlines
        "h3 a",                  # Sub-headlines
        "article a",             # Article elements
        ".lead-story a",         # Lead stories
    ],
    "economictimes.indiatimes.com": [
        ".eachStory a",          # Each story
        ".story-panel a",        # Story panels
        ".newslist a",           # News listings
        "h2 a",                  # Headlines
        "h3 a",                  # Sub-headlines
        "article a",             # Article elements
        ".lead-story a",         # Lead stories
    ],
    "news18.com": [
        ".story-listing a",      # Story listings
        ".story-card a",         # Story cards
        ".news-item a",          # News items
        "h2 a",                  # Headlines
        "h3 a",                  # Sub-headlines
        "article a",             # Article elements
        ".lead-story a",         # Lead stories
    ],
}

# ==================== PRIORITY SECTIONS ====================

# Section URLs to visit for sites where homepage has limited content
PRIORITY_SECTIONS = {
    "ndtv.com": [
        "https://www.ndtv.com/india-news",
        "https://www.ndtv.com/world-news", 
        "https://www.ndtv.com/business",
        "https://www.ndtv.com/cities",
        "https://www.ndtv.com/opinion",
        "https://www.ndtv.com/offbeat",
    ],
    "timesofindia.indiatimes.com": [
        "https://timesofindia.indiatimes.com/india",
        "https://timesofindia.indiatimes.com/world",
        "https://timesofindia.indiatimes.com/business",
        "https://timesofindia.indiatimes.com/city",
        "https://timesofindia.indiatimes.com/blogs",
    ],
    "economictimes.indiatimes.com": [
        "https://economictimes.indiatimes.com/news",
        "https://economictimes.indiatimes.com/industry", 
        "https://economictimes.indiatimes.com/markets",
        "https://economictimes.indiatimes.com/politics-nation",
        "https://economictimes.indiatimes.com/jobs",
    ],
    "news18.com": [
        "https://www.news18.com/india",
        "https://www.news18.com/world",
        "https://www.news18.com/business",
        "https://www.news18.com/politics",
        "https://www.news18.com/explainers",
    ],
}

# ==================== UTILITY FUNCTIONS ====================

def get_domain_key(url):
    """Extract domain key from URL"""
    parsed = urlparse(url)
    domain = parsed.hostname or ''
    domain = domain.lower()
    
    # Handle special cases
    if 'timesofindia' in domain:
        return 'timesofindia.indiatimes.com'
    elif 'economictimes' in domain:
        return 'economictimes.indiatimes.com'
    elif 'ndtv.com' in domain:
        return 'ndtv.com'
    elif 'news18.com' in domain:
        return 'news18.com'
    
    return domain

def is_valid_article_url(url, domain_key):
    """Check if URL matches valid article patterns"""
    if domain_key not in SITE_ARTICLE_PATTERNS:
        return False
        
    patterns = SITE_ARTICLE_PATTERNS[domain_key]
    for pattern in patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

def is_excluded_url(url, domain_key):
    """Check if URL should be excluded"""
    if domain_key not in DOMAIN_EXCLUSIONS:
        return False
        
    exclusions = DOMAIN_EXCLUSIONS[domain_key]
    for exclusion in exclusions:
        if re.search(exclusion, url, re.IGNORECASE):
            return True
    return False

def get_content_selectors(domain_key):
    """Get CSS selectors for content areas"""
    return CONTENT_SELECTORS.get(domain_key, ['a'])

def get_priority_sections(domain_key):
    """Get priority section URLs to visit"""
    return PRIORITY_SECTIONS.get(domain_key, [])

def filter_links(links, domain_key):
    """Apply all filtering rules to a list of links"""
    filtered = []
    
    for link in links:
        url = link.get('url', '')
        
        # Skip if excluded
        if is_excluded_url(url, domain_key):
            continue
            
        # Only include if matches article patterns
        if is_valid_article_url(url, domain_key):
            filtered.append(link)
    
    return filtered

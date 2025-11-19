import os
import time
import logging
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import trafilatura
from pymongo import MongoClient

# --- Configuration ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
SLEEP_TIME_SECONDS = 1800  # 30 minutes
MIN_CONTENT_LENGTH = 200
GOOGLE_NEWS_URL = "https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en"
TOPICS_TO_SCRAPE = ["India", "China", "US", "World", "Trade", "Tech", "Politics", "Business", "Entertainment", "Sports", "Science", "Health"]
MAX_ARTICLES_PER_CYCLE = 40

# Domains we should avoid treating as article sources
DISALLOWED_HOST_SUFFIXES = (
    "google.com",
    "gstatic.com",
    "youtube.com",
    "twitter.com",
    "t.co",
    "facebook.com",
    "instagram.com",
    "support.google.com",
    "policies.google.com",
    "play.google.com",
)


def is_google_news_article_path(path: str) -> bool:
    return "/articles/" in (path or "")


def is_disallowed_host(host: str) -> bool:
    if not host:
        return True
    host = host.lower()
    return any(host == s or host.endswith("." + s) for s in DISALLOWED_HOST_SUFFIXES)


def is_probable_news_source(host: str, path: str) -> bool:
    # Allow google news article redirect pages specifically, but not other google.* pages
    if host and (host == "news.google.com" or host.endswith(".google.com")):
        return is_google_news_article_path(path)
    return not is_disallowed_host(host)


def collect_candidate_links(page, base_url: str):
    """Collect all anchor links from the main content and return absolute URLs with text.
    Returns list of dict: {url, text}
    """
    try:
        page.wait_for_selector('main a', timeout=10000)
    except Exception:
        logging.warning("No anchors found under <main>. Dumping page to debug_page.html")
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(page.content())
        return []

    # Grab anchors in one shot from the page context
    anchors = page.evaluate('''() => {
        const nodes = Array.from(document.querySelectorAll('main a'));
        return nodes.map(a => ({
            href: a.getAttribute('href'),
            text: (a.getAttribute('aria-label') || a.textContent || '').trim()
        })).filter(x => x.href);
    }''')

    seen = set()
    results = []
    for a in anchors:
        # Build absolute URLs; account for JS side by using URL()
        try:
            abs_url = a['href'] if a['href'].startswith('http') else str(urljoin(base_url, a['href']))
        except Exception:
            # Fallback using Python join
            abs_url = urljoin(base_url, a['href'])
        if abs_url in seen:
            continue
        seen.add(abs_url)
        results.append({ 'url': abs_url, 'text': a['text'] })
    return results

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_collection():
    """Establishes a connection to MongoDB and returns the articles collection."""
    if not MONGO_URI or MONGO_URI == "YOUR_MONGO_CONNECTION_STRING_HERE":
        raise Exception("Error: MONGO_URI not found or not set in .env file.")
    client = MongoClient(MONGO_URI)
    db = client["Prisma"]
    logging.info("Successfully connected to MongoDB.")
    return db.articles

def get_existing_urls(collection):
    """Reads the database and returns a set of already scraped URLs."""
    logging.info("Fetching existing URLs from the database...")
    urls = {item['url'] for item in collection.find({}, {'url': 1})}
    logging.info(f"Found {len(urls)} existing URLs.")
    return urls

def get_full_url(base_url, href):
    """Constructs an absolute URL from a base and a relative href."""
    if not href:
        return None
    if href.startswith('http'):
        return href
    return urljoin(base_url, href)

def main():
    """Main loop to continuously scrape Google News."""
    try:
        collection = get_db_collection()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            
            while True:
                existing_urls = get_existing_urls(collection)
                logging.info("--- Starting new Google News scraping cycle. ---")
                
                page = browser.new_page()
                try:
                    logging.info(f"Navigating to Google News: {GOOGLE_NEWS_URL}")
                    page.goto(GOOGLE_NEWS_URL, timeout=90000)
                    page.wait_for_selector('article', timeout=60000)
                    time.sleep(3)

                    logging.info("Extracting candidate links from page...")
                    candidates = collect_candidate_links(page, page.url)
                    logging.info(f"Collected {len(candidates)} candidate links. Filtering...")

                    # Filter by probable news sources and topic keywords in link text (best-effort)
                    filtered = []
                    for c in candidates:
                        try:
                            parsed = urlparse(c['url'])
                            host, path = (parsed.hostname or '').lower(), parsed.path or ''
                            if not is_probable_news_source(host, path):
                                continue
                            # Optional: use topic keywords in link text to reduce noise
                            if c['text'] and any(t.lower() in c['text'].lower() for t in TOPICS_TO_SCRAPE):
                                filtered.append({'url': c['url'], 'headline': c['text']})
                            else:
                                # If text is empty or no keyword match, still consider google news article redirects
                                if host.endswith('google.com') and is_google_news_article_path(path):
                                    filtered.append({'url': c['url'], 'headline': c['text'] or '(no headline)'})
                        except Exception:
                            continue

                    # Dedupe and cap the number per cycle
                    dedup = []
                    seen_urls = set()
                    for item in filtered:
                        if item['url'] not in seen_urls:
                            seen_urls.add(item['url'])
                            dedup.append(item)
                    articles_to_scrape = dedup[:MAX_ARTICLES_PER_CYCLE]

                    logging.info(f"After filtering, {len(articles_to_scrape)} links selected for scraping.")

                    if not articles_to_scrape:
                        # Also dump page for troubleshooting
                        with open("debug_page.html", "w", encoding="utf-8") as f:
                            f.write(page.content())
                        logging.info("No articles selected. Saved page HTML to debug_page.html for inspection.")
                    else:
                        new_articles_found = 0
                        for article_info in articles_to_scrape:
                            initial_url = article_info['url']
                            logging.info(f"Scraping candidate: '{article_info['headline']}' -> {initial_url}")

                            article_page = browser.new_page()
                            try:
                                article_page.goto(initial_url, timeout=90000, wait_until='domcontentloaded')
                                final_url = article_page.url

                                parsed_final = urlparse(final_url)
                                if not is_probable_news_source(parsed_final.hostname or '', parsed_final.path or ''):
                                    logging.info("  -> Skipped: Final URL not a probable news source.")
                                    continue

                                # Skip duplicates already in DB
                                if final_url in existing_urls:
                                    logging.info("  -> Skipped: Final URL already in database.")
                                    continue

                                html = article_page.content()
                                content = trafilatura.extract(html)

                                if content and len(content) > MIN_CONTENT_LENGTH:
                                    collection.update_one(
                                        {'url': final_url},
                                        {'$setOnInsert': {
                                            'text': content,
                                            'source': 'google_news',
                                            'headline': article_info['headline']
                                        }},
                                        upsert=True
                                    )
                                    existing_urls.add(final_url)
                                    new_articles_found += 1
                                    logging.info("  -> SUCCESS: Scraped and saved to MongoDB.")
                                else:
                                    logging.info("  -> Skipped: Content too short or extraction failed.")
                            except Exception as e:
                                logging.error(f"  -> Error scraping article: {e}")
                            finally:
                                article_page.close()
                        logging.info(f"Scraping finished. Added {new_articles_found} new articles.")

                except Exception as e:
                    logging.critical(f"An error occurred during the scraping cycle: {e}")
                finally:
                    page.close()

                logging.info(f"Cycle finished. Waiting for {SLEEP_TIME_SECONDS / 60:.1f} minutes...")
                time.sleep(SLEEP_TIME_SECONDS)

            browser.close()

    except Exception as e:
        logging.critical(f"A critical error occurred in main loop: {e}")

if __name__ == "__main__":
    main()

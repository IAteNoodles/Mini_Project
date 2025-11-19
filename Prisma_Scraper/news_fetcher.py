import os
import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import trafilatura
from pymongo import MongoClient
import time
from multiprocessing import Pool, cpu_count
from functools import partial
import threading
from concurrent.futures import ThreadPoolExecutor

# Import configuration
try:
    from config_parallel import (
        NUM_CORES, MAX_BROWSER_WINDOWS, MIN_CONTENT_LENGTH, 
        SLEEP_TIME_SECONDS, BROWSER_TIMEOUT, PAGE_WAIT_TIME,
        BROWSER_ARGS, USER_AGENT, auto_configure, validate_configuration,
        get_filtered_topics
    )
    print("Loaded parallel configuration")
    # Auto-configure based on system specs
    auto_config = auto_configure()
    
    # Validate configuration and show warnings
    warnings = validate_configuration()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  {warning}")
except ImportError:
    print("config_parallel.py not found, using default configuration")
    NUM_CORES = 16
    MAX_BROWSER_WINDOWS = 4
    MIN_CONTENT_LENGTH = 200
    SLEEP_TIME_SECONDS = 900
    BROWSER_TIMEOUT = 60000
    PAGE_WAIT_TIME = 2000
    BROWSER_ARGS = ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    
    def get_filtered_topics(topics):
        return topics

load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
TOPICS = [
    # Politics (highly prone to bias)
    "Indian Politics", "US Politics", "Global Politics", "Electoral Politics", "Democracy",
    "Authoritarianism", "Political Corruption", "Dynastic Politics (India)", "Kashmir Conflict",
    "India-China Relations", "India-Pakistan Relations", "Modi Government", "Indian Parliament",
    "State Elections India", "Coalition Politics", "Indian Opposition", "Central-State Relations",
    
    # Indian Critical Issues (often with biased coverage)
    "Farmers Protests", "Land Acquisition", "Reservation Politics", "Article 370",
    "CAA NRC", "Ram Mandir", "Uniform Civil Code", "Anti-Corruption Movement", 
    "Demonetization", "GST Implementation", "Covid Response India", "Digital India",
    
    # Religion and Religious Issues (often biased coverage)
    "Hinduism", "Islam", "Christianity", "Religious Conflict", "Secularism", 
    "Hindutva", "Religious Conversions", "Communal Violence", "Cow Slaughter Ban",
    "Religious Extremism", "Blasphemy", "Religious Freedom",
    
    # Ideology and Cultural Identity (prone to partisan bias)
    "Nationalism", "Populism", "Liberalism", "Conservatism", "Socialism", 
    "Saffronization", "Cultural Identity", "Traditional Values", "Westernization",
    
    # Minority and Racial Issues (controversial coverage)
    "Minority Rights", "Dalit Rights", "Tribal Rights", "Caste Discrimination",
    "Islamophobia", "Anti-Semitism", "Racism", "Systemic Racism", "Affirmative Action", 
    "Reservation Policy (India)", "Immigration", "Refugee Crisis",
    
    # Language and Regional Issues
    "Language Politics (India)", "Hindi Imposition", "Regionalism (India)", 
    "Linguistic Identity", "State Reorganization", "North-South Divide",
    
    # Trade and Economic Issues (often politically charged)
    "Globalization", "Protectionism", "Trade Wars", "Economic Nationalism",
    "Foreign Investment", "Free Trade", "Economic Sanctions", "Labor Rights",
    
    # Media and information
    "Media Bias", "Fake News", "Social Media", "Press Freedom", "Internet Shutdowns", "Censorship", "Deepfakes", "Misinformation",

    # Censorship, surveillance, dissent
    "Surveillance", "Mass Surveillance", "Pegasus Spyware", "Privacy", "State Repression", "Crackdown on Dissent", "Criminalizing Protest", "Internet Blackouts",

    # Misrepresentation and framing
    "Protester Labeling", "Terrorism Labeling", "Protester Criminalization", "Framing of Protests", "Dehumanization", "Disinformation", "Mob Lynching", "Vigilantism",

    # Recent hot / controversial topics (India-focused and global)
    "Adani-Hindenburg", "Rafale Controversy", "Karnataka Hijab", "Love Jihad", "Gyanvapi", "Delhi Riots", "CAA NRC", "Article 370", "Farmers Protests",
    "COVID-19 Misinformation", "Vaccine Mandates", "Black Lives Matter", "Israel-Gaza", "Russia-Ukraine War", "China-Taiwan", "Climate Protests", "MeToo",

    # Platform moderation and online speech
    "Content Moderation", "Platform Censorship", "Section 69A", "Intermediary Liability", "Digital Rights", "Net Neutrality"
]
MIN_CONTENT_LENGTH = 200
SLEEP_TIME_SECONDS = 900  # 15 minutes

def get_db_collection():
    """Establishes a connection to MongoDB and returns the articles collection."""
    if not MONGO_URI or MONGO_URI == "YOUR_MONGO_CONNECTION_STRING_HERE":
        raise Exception("Error: MONGO_URI not found or not set in .env file.")
    client = MongoClient(MONGO_URI)
    db = client.get_database("Prisma")
    return db.articles

def get_existing_urls(collection):
    """Reads the database and returns a set of already scraped URLs."""
    print("Fetching existing URLs from the database...")
    urls = {item['url'] for item in collection.find({}, {'url': 1})}
    print(f"Found {len(urls)} existing URLs.")
    return urls

def scrape_single_article(article_data, topic, existing_urls):
    """
    Scrape a single article using a dedicated browser instance.
    """
    article_url = article_data['url']
    if article_url in existing_urls:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=BROWSER_ARGS
            )
            context = browser.new_context(
                user_agent=USER_AGENT
            )
            page = context.new_page()
            
            # Set timeout and navigate
            page.set_default_timeout(BROWSER_TIMEOUT)
            page.goto(article_url, wait_until='domcontentloaded')
            
            # Wait a bit for dynamic content
            page.wait_for_timeout(PAGE_WAIT_TIME)
            
            html = page.content()
            context.close()
            browser.close()

            content = trafilatura.extract(html)

            if content and len(content) > MIN_CONTENT_LENGTH:
                return {
                    'url': article_url,
                    'text': content,
                    'topic': topic,
                    'title': article_data.get('title', ''),
                    'scraped_at': time.time()
                }
            else:
                print(f"  -> Content too short for: {article_data.get('title', 'Unknown')}")
                return None

    except Exception as e:
        print(f"  -> Could not process article {article_data.get('title', 'Unknown')}: {e}")
        return None

def save_articles_batch(articles, collection):
    """
    Save a batch of articles to MongoDB using proper bulk operations.
    """
    if not articles:
        return 0
    
    try:
        from pymongo import UpdateOne
        
        operations = []
        for article in articles:
            # Clean up the article data to ensure it's JSON serializable
            clean_article = {
                'url': str(article['url']),
                'text': str(article['text']),
                'topic': str(article['topic']),
                'title': str(article.get('title', '')),
                'scraped_at': float(article.get('scraped_at', 0))
            }
            
            operations.append(
                UpdateOne(
                    {'url': clean_article['url']},
                    {'$setOnInsert': clean_article},
                    upsert=True
                )
            )
        
        if operations:
            result = collection.bulk_write(operations, ordered=False)
            return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"Error saving batch to MongoDB: {e}")
        # Try saving articles one by one as fallback
        saved_count = 0
        for article in articles:
            try:
                collection.update_one(
                    {'url': article['url']},
                    {'$setOnInsert': article},
                    upsert=True
                )
                saved_count += 1
            except Exception as single_error:
                print(f"Error saving single article {article.get('title', 'Unknown')}: {single_error}")
        return saved_count
    
    return 0

def scrape_and_save_parallel(topic, existing_urls):
    """
    Fetches news for a topic and scrapes content using parallel processing.
    """
    print(f"\n--- Searching for new articles on topic: {topic} ---")
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("Error: NEWS_API_KEY not found.")
        return []

    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "ok":
            articles = data.get("articles", [])
            
            # Filter out articles that already exist
            new_articles = [article for article in articles if article['url'] not in existing_urls]
            
            if not new_articles:
                print("No new articles found for this topic in this cycle.")
                return []
            
            print(f"Found {len(new_articles)} new articles to process for topic: {topic}")
            
            # Use ThreadPoolExecutor for parallel article scraping within this topic
            scraped_articles = []
            with ThreadPoolExecutor(max_workers=MAX_BROWSER_WINDOWS) as executor:
                # Submit all article scraping tasks
                future_to_article = {
                    executor.submit(scrape_single_article, article_data, topic, existing_urls): article_data
                    for article_data in new_articles
                }
                
                # Collect results as they complete
                for future in future_to_article:
                    result = future.result()
                    if result:
                        scraped_articles.append(result)
                        print(f"  -> Successfully scraped: {result['title']}")
            
            return scraped_articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news API for topic {topic}: {e}")
        return []

def process_topic_batch(topics_batch, existing_urls):
    """
    Process a batch of topics and return all scraped articles.
    Each process gets its own MongoDB connection.
    """
    # Create a new MongoDB connection for this process
    try:
        collection = get_db_collection()
    except Exception as e:
        print(f"Error connecting to MongoDB in process: {e}")
        return []
    
    all_scraped_articles = []
    for topic in topics_batch:
        scraped_articles = scrape_and_save_parallel(topic, existing_urls)
        all_scraped_articles.extend(scraped_articles)
        
        # Save articles immediately in this process to avoid memory buildup
        if scraped_articles:
            saved_count = save_articles_batch(scraped_articles, collection)
            print(f"Process saved {saved_count} articles for topic: {topic}")
    
    return len(all_scraped_articles)  # Return count instead of articles to save memory

def main():
    """Main loop to continuously scrape for news using parallel processing."""
    try:
        # Test MongoDB connection first
        collection = get_db_collection()
        print("MongoDB connection established successfully.")
        
        # Filter topics based on configuration
        filtered_topics = get_filtered_topics(TOPICS)
        print(f"Using {len(filtered_topics)} topics for scraping")
        
        cycle_count = 0
        while True:
            cycle_count += 1
            print(f"\n{'='*60}")
            print(f"🚀 Starting scraping cycle #{cycle_count}")
            print(f"⚙️  Configuration: {NUM_CORES} processes × {MAX_BROWSER_WINDOWS} browser windows")
            print(f"{'='*60}")
            
            cycle_start_time = time.time()
            existing_urls = get_existing_urls(collection)
            
            # Split topics into batches for parallel processing
            topics_per_process = len(filtered_topics) // NUM_CORES + (1 if len(filtered_topics) % NUM_CORES else 0)
            topic_batches = [
                filtered_topics[i:i + topics_per_process] 
                for i in range(0, len(filtered_topics), topics_per_process)
            ]
            
            print(f"📊 Processing {len(filtered_topics)} topics across {len(topic_batches)} batches")
            print(f"📋 Topics per process: {topics_per_process}")
            
            # Use multiprocessing to handle topic batches in parallel
            total_articles_processed = 0
            try:
                with Pool(processes=NUM_CORES) as pool:
                    # Create partial function with existing_urls
                    process_func = partial(process_topic_batch, existing_urls=existing_urls)
                    
                    # Process topic batches in parallel
                    print(f"🔄 Starting {NUM_CORES} parallel processes...")
                    results = pool.map(process_func, topic_batches)
                    
                    # Sum up the counts
                    total_articles_processed = sum(results)
            
            except Exception as e:
                print(f"❌ Error during parallel processing: {e}")
                import traceback
                traceback.print_exc()
            
            cycle_duration = time.time() - cycle_start_time
            
            print(f"\n{'='*60}")
            print(f"📊 Cycle #{cycle_count} Summary:")
            print(f"⏱️  Duration: {cycle_duration:.1f} seconds ({cycle_duration/60:.1f} minutes)")
            print(f"📰 Articles processed: {total_articles_processed}")
            if cycle_duration > 0:
                print(f"⚡ Throughput: {total_articles_processed/cycle_duration*60:.1f} articles/minute")
            print(f"🎯 Average per process: {total_articles_processed/NUM_CORES:.1f} articles")
            print(f"{'='*60}")
            
            print(f"\n😴 Waiting {SLEEP_TIME_SECONDS/60:.0f} minutes until next cycle...")
            print(f"⏰ Next cycle will start at: {time.strftime('%H:%M:%S', time.localtime(time.time() + SLEEP_TIME_SECONDS))}")
            
            time.sleep(SLEEP_TIME_SECONDS)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Received interrupt signal. Shutting down gracefully...")
        print("👋 Thank you for using the parallel news scraper!")
    except Exception as e:
        print(f"💥 A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Try running the test script first: python test_parallel_scraper.py")


if __name__ == "__main__":
    # This is required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()

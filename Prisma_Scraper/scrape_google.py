# -*- coding: utf-8 -*-
"""
Flask Application for Fetching News Topics and Related Articles.

This application:
1.  Fetches general news headlines.
2.  Summarizes them into key topics.
3.  Caches the generated topics using Redis.
4.  For the top 3 topics, launches background tasks.
5.  Background tasks fetch 5 related article links for each topic using Google News RSS.
6.  Resolves the final URLs of these articles using Selenium (headless Firefox).
7.  Posts the final URLs to a specified endpoint (RESULTS_POST_ENDPOINT).
8.  Uses a Semaphore to limit concurrent Selenium instances.
9.  Provides extremely verbose logging for every step.
"""

import flask
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, InvalidSessionIdException, TimeoutException
import time
import threading
import requests
import json
import logging
import os
import redis
import hashlib # Added for hashing topics
import summarise


from flask import request, jsonify
from flask_cors import CORS

# --- Configuration ---
logging.info("--- Starting Configuration ---")

# Define the endpoint where background tasks will post their results
RESULTS_POST_ENDPOINT = "http://192.168.137.113:6000/receive-json" # CHANGE IF NEEDED
logging.info(f"Results will be posted to: {RESULTS_POST_ENDPOINT}")

# Specify the path to your downloaded geckodriver executable
GECKODRIVER_PATH = r"C:\Tools\geckodriver.exe" # CHANGE IF NEEDED
logging.info(f"GeckoDriver path set to: {GECKODRIVER_PATH}")

# Specify the path to your Firefox browser executable
FIREFOX_BINARY_PATH = r"C:\Program Files\Mozilla Firefox\firefox.exe" # CHANGE IF NEEDED
logging.info(f"Firefox binary path set to: {FIREFOX_BINARY_PATH}")

# --- Hardcoded Constraints ---
NUM_INITIAL_ARTICLES = 15 # Number of articles to fetch for generating topics
MAX_TOPICS_TO_PROCESS = 3 # Max number of topics to process in background
NUM_ARTICLES_PER_TOPIC = 5 # Number of articles to fetch per topic
logging.info(f"Hardcoded Constraints: Initial Articles={NUM_INITIAL_ARTICLES}, Max Topics={MAX_TOPICS_TO_PROCESS}, Articles per Topic={NUM_ARTICLES_PER_TOPIC}")

# --- Redis Configuration ---
logging.info("--- Configuring Redis ---")
# Get Redis details from environment variables
# Make sure these env vars are set before running the app
Redis_Key = os.getenv("Redis_Key", "YOUR_DEFAULT_PASSWORD_IF_NEEDED") # Provide a default or ensure it's set
Redis_Host = os.getenv("Redis_Host", "localhost") # Default to localhost if not set
REDIS_PORT = 6379 # Default Redis port
logging.info(f"Attempting Redis connection using Host: {Redis_Host}, Port: {REDIS_PORT}")

# Cache expiration time in seconds (e.g., 5 minutes)
CACHE_TIMEOUT_SECONDS = 300
TOPICS_CACHE_KEY = 'latest_topics_v2' # Use a distinct key
logging.info(f"Redis Cache Timeout: {CACHE_TIMEOUT_SECONDS} seconds. Cache Key: {TOPICS_CACHE_KEY}")

# --- Redis Connection ---
redis_client = None
try:
    logging.info(f"Attempting to connect to Redis at {Redis_Host}:{REDIS_PORT}...")
    # Using StrictRedis for potentially more standard behavior if issues arise with basic Redis class
    redis_client = redis.Redis(
    host='redis-12173.c93.us-east-1-3.ec2.redns.redis-cloud.com',
    port=12173,
    decode_responses=True,
    username="default",
    password="VDxmDM4EMGcOHiBqdGcUn1GZKjWm1G15",)

    # Ping Redis to check the connection
    redis_client.ping()
    logging.info(f"Successfully connected and pinged Redis server at {Redis_Host}:{REDIS_PORT}.")
except redis.exceptions.AuthenticationError as e:
    logging.error(f"Redis authentication failed for {Redis_Host}:{REDIS_PORT}. Check Redis_Key. Caching disabled. Error: {e}")
    redis_client = None
except redis.exceptions.TimeoutError as e:
    logging.error(f"Redis connection timed out for {Redis_Host}:{REDIS_PORT}. Caching disabled. Error: {e}")
    redis_client = None
except redis.exceptions.ConnectionError as e:
    logging.error(f"Failed to connect to Redis at {Redis_Host}:{REDIS_PORT}. Caching will be disabled. Error: {e}", exc_info=False)
    redis_client = None # Set client to None if connection fails
except Exception as e:
    logging.error(f"An unexpected error occurred while connecting to Redis: {e}. Caching will be disabled.", exc_info=True)
    redis_client = None # Set client to None if connection fails

# --- Concurrency Limiter ---
# Limit the number of concurrent Selenium instances
MAX_CONCURRENT_SELENIUM = 3 # Reduced limit for potentially lower resource usage
selenium_semaphore = threading.Semaphore(MAX_CONCURRENT_SELENIUM)
logging.info(f"Semaphore initialized to allow {MAX_CONCURRENT_SELENIUM} concurrent Selenium tasks.")

# Check if paths exist (optional but recommended)
logging.info("--- Checking File Paths ---")
if not os.path.exists(GECKODRIVER_PATH):
    logging.error(f"GeckoDriver NOT FOUND at specified path: {GECKODRIVER_PATH}. Selenium tasks will fail.")
else:
    logging.info(f"GeckoDriver found at: {GECKODRIVER_PATH}")

if not os.path.exists(FIREFOX_BINARY_PATH):
    logging.error(f"Firefox binary NOT FOUND at specified path: {FIREFOX_BINARY_PATH}. Selenium tasks will fail.")
else:
    logging.info(f"Firefox binary found at: {FIREFOX_BINARY_PATH}")


# Configure logging - MORE VERBOSE FORMAT
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for even more detail if needed
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

logging.info("--- Initializing Flask App ---")
# --- Flask App Initialization ---
app = flask.Flask(__name__)
CORS(app) # Enable CORS for all routes
logging.info("Flask app initialized and CORS enabled.")

# --- Helper: Get Topic Hash ---
def get_topic_hash(topic_title: str) -> str:
    """Generates an MD5 hash for a given topic title string."""
    logging.debug(f"Generating MD5 hash for topic: '{topic_title}'")
    hash_object = hashlib.md5(topic_title.encode('utf-8')) # Ensure consistent encoding
    hex_dig = hash_object.hexdigest()
    logging.debug(f"Generated hash '{hex_dig}' for topic: '{topic_title}'")
    return hex_dig

# --- News Fetching Functions ---

def get_initial_news_articles():
    """
    Fetches a fixed number of news articles (NUM_INITIAL_ARTICLES)
    from Google News RSS for topic generation. Logs extensively.

    Returns:
        list: A list of dictionaries, each containing 'title', 'link', 'summary'.
              Returns an empty list on error.
    """
    logging.info(f"--- Entering get_initial_news_articles (Fetching {NUM_INITIAL_ARTICLES} articles) ---")
    rss_url = 'https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en' # India-specific feed
    logging.info(f"Using Google News RSS URL: {rss_url}")

    articles = []
    try:
        # Parse the RSS feed from Google News
        logging.debug(f"Attempting to parse RSS feed from: {rss_url}")
        feed = feedparser.parse(rss_url)
        logging.info(f"Successfully parsed RSS feed. Found {len(feed.entries)} entries.")

        if not feed.entries:
            logging.warning("RSS feed is empty. No articles to process.")
            return []

        # Extract the top N news articles
        logging.info(f"Extracting top {NUM_INITIAL_ARTICLES} articles from the feed.")
        for i, entry in enumerate(feed.entries[:NUM_INITIAL_ARTICLES]):
            logging.debug(f"Processing article {i+1}/{NUM_INITIAL_ARTICLES}: '{entry.get('title', 'N/A')}'")
            # Parse the summary and remove any HTML tags using BeautifulSoup
            summary = entry.summary if 'summary' in entry else 'No summary available'
            soup = BeautifulSoup(summary, 'html.parser')
            summary_text = soup.get_text().strip() # Remove extra whitespace

            article_data = {
                'title': entry.get('title', 'No Title Provided'), # Use .get for safety
                'link': entry.get('link', '#'),
                'summary': summary_text
            }
            articles.append(article_data)
            logging.debug(f"Added article: {article_data['title']}")

        logging.info(f"Successfully extracted {len(articles)} initial articles.")

    except Exception as e:
        logging.error(f"Error fetching or parsing initial news from {rss_url}: {e}", exc_info=True)
        articles = [] # Ensure empty list is returned on error

    logging.info(f"--- Exiting get_initial_news_articles (Returning {len(articles)} articles) ---")
    return articles


def get_resolved_news_urls_for_topic(topic_title: str):
    """
    Fetches a fixed number (NUM_ARTICLES_PER_TOPIC) of news article links
    for a given topic from Google News RSS and resolves their final redirected
    URLs using Selenium. Logs extensively.

    Args:
        topic_title (str): The topic to search for.

    Returns:
        list: A list of final redirected article URLs. Returns an empty list
              if errors occur.
    """
    logging.info(f"--- Entering get_resolved_news_urls_for_topic for topic: '{topic_title}' (Fetching {NUM_ARTICLES_PER_TOPIC} URLs) ---")
    resolved_urls = []
    driver = None # Initialize driver to None

    # --- Acquire Semaphore ---
    logging.debug(f"Waiting to acquire semaphore for topic: '{topic_title}'")
    acquired = selenium_semaphore.acquire(timeout=20) # Add a timeout to semaphore acquisition
    if not acquired:
        logging.error(f"TIMEOUT acquiring semaphore for topic: '{topic_title}'. Aborting task.")
        return [] # Cannot proceed without semaphore

    logging.info(f"Semaphore acquired for topic: '{topic_title}'. Proceeding with Selenium task.")
    try:
        # 1. Construct Google News RSS Search URL
        encoded_topic = quote(topic_title)
        # Using India-specific Google News feed
        search_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-IN&gl=IN&ceid=IN:en"
        logging.info(f"Fetching RSS feed for topic '{topic_title}' from URL: {search_url}")

        # 2. Fetch initial links from RSS
        try:
            feed = feedparser.parse(search_url)
            logging.info(f"Parsed RSS feed for '{topic_title}'. Found {len(feed.entries)} entries.")
            initial_links = [entry.link for entry in feed.entries[:NUM_ARTICLES_PER_TOPIC]]
            logging.info(f"Extracted {len(initial_links)} initial links for topic '{topic_title}'.")
        except Exception as e:
            logging.error(f"Failed to fetch or parse RSS feed for topic '{topic_title}' from {search_url}: {e}", exc_info=True)
            return [] # Cannot proceed if RSS fetch fails

        if not initial_links:
            logging.warning(f"No articles found via RSS for topic: '{topic_title}' from {search_url}. Returning empty list.")
            return [] # Exit early if no links

        # 3. --- Selenium Driver Setup ---
        logging.info(f"Initializing Selenium Chrome driver for topic: '{topic_title}'...")
        try:
            driver = get_selenium_driver()
            logging.info(f"Selenium Chrome driver initialized successfully for topic: '{topic_title}'.")
        except (WebDriverException, FileNotFoundError) as e:
            logging.error(f"CRITICAL: Failed to initialize Selenium Chrome driver for topic '{topic_title}'. Check paths/permissions. Error: {e}", exc_info=True)
            return []  # Cannot proceed without a driver

        # 4. --- Process Links ---
        logging.info(f"Starting to resolve {len(initial_links)} URLs for topic '{topic_title}' using Selenium.")
        for i, link in enumerate(initial_links):
            logging.debug(f"Processing URL {i+1}/{len(initial_links)}: {link}")
            try:
                driver.get(link)
                logging.debug(f"Called driver.get('{link}'). Waiting for page load/redirects...")
                # Using a fixed sleep is less reliable than explicit waits, but simpler.
                # Increase sleep if redirects take longer.
                time.sleep(4) # Allow time for JS redirects
                final_url = driver.current_url
                logging.debug(f"Original URL: {link} -> Resolved URL: {final_url}")

                # Validate the final URL before adding
                if final_url and isinstance(final_url, str) and final_url.startswith('http'):
                    resolved_urls.append(final_url)
                    logging.debug(f"Added final URL: {final_url}")
                else:
                    logging.warning(f"Could not resolve or invalid final URL ('{final_url}') obtained for original link: {link}")
                    # Optionally add the original link if resolution fails?
                    # resolved_urls.append(link)
                    # logging.debug(f"Adding original link as fallback: {link}")

            # Catch specific Selenium errors during URL processing
            except TimeoutException:
                logging.error(f"TIMEOUT loading URL {link} for topic '{topic_title}'. Skipping this URL.")
                continue # Skip to the next link
            except InvalidSessionIdException as e:
                logging.error(f"Invalid Selenium session ID for topic '{topic_title}' while processing {link}. Driver may be dead. Stopping URL processing for this topic. Error: {e}", exc_info=False)
                break # Exit the for loop for this topic - driver is likely unusable
            except WebDriverException as e:
                logging.error(f"WebDriverException processing URL {link} for topic '{topic_title}'. Skipping this URL. Error: {e}", exc_info=False)
                continue # Skip to the next link
            except Exception as e:
                # Catch any other unexpected errors for this specific URL
                logging.error(f"Unexpected error processing URL {link} for topic '{topic_title}': {e}", exc_info=True)
                continue # Skip to the next link

        logging.info(f"Finished resolving URLs for topic '{topic_title}'. Found {len(resolved_urls)} final URLs.")

    except Exception as e:
        # Log error for the overall function logic (outside driver setup/link processing)
        logging.error(f"Error during execution of get_resolved_news_urls_for_topic for '{topic_title}': {e}", exc_info=True)
        resolved_urls = [] # Ensure empty list on major error
    finally:
        # --- Ensure Driver is Quit ---
        if driver:
            logging.info(f"Attempting to quit Selenium driver for topic: '{topic_title}'")
            try:
                driver.quit()
                logging.info(f"Successfully quit Selenium driver for topic: '{topic_title}'.")
            except Exception as e:
                logging.error(f"Error quitting Selenium driver for topic '{topic_title}': {e}", exc_info=False) # Less verbose if quitting fails

        # --- Release Semaphore ---
        if acquired: # Only release if it was acquired
             selenium_semaphore.release()
             logging.info(f"Semaphore released for topic: '{topic_title}'.")
        else:
            logging.debug(f"Semaphore was not acquired for topic '{topic_title}', no release needed.")


    logging.info(f"--- Exiting get_resolved_news_urls_for_topic for topic: '{topic_title}' (Returning {len(resolved_urls)} URLs) ---")
    return resolved_urls

# --- Background Task Function ---
def process_topic_in_background(topic_title: str):
    """
    Runs in a background thread. Fetches and resolves news URLs for a topic
    using get_resolved_news_urls_for_topic, then POSTs the results to
    RESULTS_POST_ENDPOINT. Logs extensively.

    Args:
        topic_title (str): The topic to process.
    """
    thread_name = threading.current_thread().name
    logging.info(f"--- [{thread_name}] Background task started for topic: '{topic_title}' ---")

    try:
        # 1. Fetch and resolve URLs (Semaphore handled within the function)
        logging.info(f"[{thread_name}] Calling get_resolved_news_urls_for_topic for topic '{topic_title}'...")
        final_urls = get_resolved_news_urls_for_topic(topic_title) # Uses hardcoded NUM_ARTICLES_PER_TOPIC
        logging.info(f"[{thread_name}] Received {len(final_urls)} resolved URLs for topic '{topic_title}'.")

        if not final_urls:
            logging.warning(f"[{thread_name}] No final URLs found or generated for topic '{topic_title}'. Nothing to post.")
            logging.info(f"--- [{thread_name}] Background task finished early for topic: '{topic_title}' ---")
            return # Exit if no URLs were found/resolved

        # 2. Prepare the JSON payload
        payload = {
            "topic_title": topic_title,
            "topic_hash": get_topic_hash(topic_title), # Include hash for potential matching
            "article_urls": final_urls
        }
        logging.info(f"[{thread_name}] Prepared JSON payload for topic '{topic_title}' with {len(final_urls)} URLs.")
        logging.debug(f"[{thread_name}] Payload: {json.dumps(payload, indent=2)}") # Log payload content at debug level

        # 3. POST the results to the target endpoint
        headers = {'Content-Type': 'application/json'}
        post_timeout = 60 # Timeout for the POST request
        logging.info(f"[{thread_name}] Attempting to POST results for topic '{topic_title}' to {RESULTS_POST_ENDPOINT} (Timeout: {post_timeout}s)")
        try:
            response = requests.post(RESULTS_POST_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=post_timeout)
            logging.info(f"[{thread_name}] POST request sent for topic '{topic_title}'. Status Code: {response.status_code}")
            # Check if the request was successful (status code 2xx)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            logging.info(f"[{thread_name}] Successfully posted results for topic '{topic_title}' to {RESULTS_POST_ENDPOINT}.")

        except requests.exceptions.Timeout:
            logging.error(f"[{thread_name}] TIMEOUT posting results for topic '{topic_title}' to {RESULTS_POST_ENDPOINT}.", exc_info=False)
        except requests.exceptions.ConnectionError:
            logging.error(f"[{thread_name}] Connection Error posting results for topic '{topic_title}' to {RESULTS_POST_ENDPOINT}. Is the receiving server running and accessible?", exc_info=False)
        except requests.exceptions.RequestException as e:
            # Log other request errors (like HTTP errors)
            logging.error(f"[{thread_name}] Failed to post results for topic '{topic_title}' to {RESULTS_POST_ENDPOINT}. Status Code: {e.response.status_code if e.response else 'N/A'}. Response: {e.response.text if e.response else 'N/A'}", exc_info=False)

    except Exception as e:
        # Catch any unexpected errors within the thread logic itself
        logging.error(f"[{thread_name}] Unexpected error in background task logic for topic '{topic_title}': {e}", exc_info=True)

    logging.info(f"--- [{thread_name}] Background task finished for topic: '{topic_title}' ---")


# --- API Endpoints ---

@app.route("/api/get_topics", methods=["GET"])
def get_topics():
    """
    GET Endpoint: Fetches initial news, summarizes into topics, caches results,
    starts background processing for the top MAX_TOPICS_TO_PROCESS topics,
    and returns the list of [hash, topic_title] pairs. Logs extensively.

    Returns:
        JSON: A list containing [hash, topic_title] for the generated topics,
              or an error message.
    """
    logging.info("====== [/api/get_topics] Endpoint Called ======")

    # --- Check Cache ---
    if redis_client:
        logging.info("Redis client available. Attempting to retrieve topics from cache.")
        try:
            cached_data = redis_client.get(TOPICS_CACHE_KEY)
            if cached_data:
                logging.info(f"CACHE HIT for key '{TOPICS_CACHE_KEY}'. Returning cached data.")
                # Deserialize the cached JSON string
                cached_topics_list = json.loads(cached_data)
                # Basic validation of cached data format
                if isinstance(cached_topics_list, list) and all(isinstance(item, list) and len(item) == 2 for item in cached_topics_list):
                    logging.info(f"Cached data format looks valid. Returning {len(cached_topics_list)} topics from cache.")
                    logging.info("====== [/api/get_topics] Request Finished (Returned Cached Data) ======")
                    return jsonify(cached_topics_list), 200
                else:
                    logging.warning("Cached data format is unexpected. Proceeding to generate fresh data.")
                    # Optionally delete the invalid cache entry
                    try:
                        redis_client.delete(TOPICS_CACHE_KEY)
                        logging.info(f"Deleted invalid cache entry for key '{TOPICS_CACHE_KEY}'.")
                    except redis.exceptions.RedisError as del_e:
                        logging.warning(f"Could not delete invalid cache key '{TOPICS_CACHE_KEY}': {del_e}")

            else:
                logging.info(f"CACHE MISS for key '{TOPICS_CACHE_KEY}'.")

        except (redis.exceptions.RedisError, json.JSONDecodeError) as e:
            logging.error(f"Error accessing or deserializing Redis cache key '{TOPICS_CACHE_KEY}'. Proceeding without cache. Error: {e}", exc_info=False)
        except Exception as e:
            logging.error(f"Unexpected error during cache retrieval: {e}. Proceeding without cache.", exc_info=True)
    else:
        logging.warning("Redis client is not connected. Cannot use cache.")

    # --- Cache Miss or Redis Down: Generate New Topics ---
    logging.info("Proceeding to generate fresh topics.")
    try:
        # 1. Get initial articles
        logging.info(f"Calling get_initial_news_articles() to fetch {NUM_INITIAL_ARTICLES} articles.")
        articles_data = get_initial_news_articles() # Uses hardcoded number
        if not articles_data:
            logging.error("Failed to get initial articles. Cannot generate topics.")
            return jsonify({"error": "Failed to fetch initial news articles required for topic generation"}), 500
        logging.info(f"Received {len(articles_data)} initial articles for summarization.")

        # 2. Summarise into topics (using the external 'summarise' module)
        logging.info("Calling summarise.summarise_into_topics()...")
        try:
            # Ensure summarise module/function is available
            if 'summarise' not in globals() or not hasattr(summarise, 'summarise_into_topics'):
                 logging.error("Summarise module or function not available. Cannot generate topics.")
                 raise RuntimeError("Summarization module/function not loaded.")

            raw_data = summarise.summarise_into_topics_and_domain(articles_data)

            raw_topics  = [item[0] for item in raw_data] # Extract only the topic titles
            raw_domains = [item[1] for item in raw_data] # Domains are not used in this version

            logging.info(f"Summarization complete. Received {len(raw_topics)} raw topics: {raw_topics}")
        except Exception as sum_e:
             logging.error(f"Error during topic summarization using 'summarise' module: {sum_e}", exc_info=True)
             return jsonify({"error": "Topic summarization failed."}), 500


        if not isinstance(raw_topics, list):
            logging.error(f"Summarization did not return a list. Got type: {type(raw_topics)}. Value: {raw_topics}")
            return jsonify({"error": "Topic summarization returned unexpected format"}), 500

        # Filter out empty/invalid topics and prepare [hash, topic] pairs
        hashed_topics = []
        for i, topic in enumerate(raw_topics):
            if isinstance(topic, str) and topic.strip():
                topic_clean = topic.strip()
                topic_hash = get_topic_hash(topic_clean)
                hashed_topics.append([topic_hash, topic_clean])
                logging.debug(f"Generated valid topic pair {i+1}: [{topic_hash}, '{topic_clean}']")
            else:
                logging.warning(f"Skipping invalid/empty topic item at index {i}: {topic}")

        logging.info(f"Generated {len(hashed_topics)} valid [hash, topic] pairs.")

        if not hashed_topics:
            logging.warning("No valid topics were generated after filtering. Returning empty list.")
            # Still cache the empty result? Maybe not.
            return jsonify([]), 200 # Return empty list, not an error

        # --- Select Top Topics ---
        topics_to_process = hashed_topics[:MAX_TOPICS_TO_PROCESS]
        logging.info(f"Selected top {len(topics_to_process)} topics for background processing based on MAX_TOPICS_TO_PROCESS={MAX_TOPICS_TO_PROCESS}.")
        logging.debug(f"Topics selected for processing: {topics_to_process}")

        # --- Store *all* valid generated topics (hashed_topics) in Cache ---
        if redis_client:
            logging.info(f"Attempting to store all {len(hashed_topics)} generated topics in Redis cache.")
            try:
                cache_data = json.dumps(hashed_topics) # Cache all generated topics
                redis_client.set(TOPICS_CACHE_KEY, cache_data, ex=CACHE_TIMEOUT_SECONDS)
                logging.info(f"Successfully stored topics in Redis cache key '{TOPICS_CACHE_KEY}' with {CACHE_TIMEOUT_SECONDS}s expiry.")
            except redis.exceptions.RedisError as e:
                logging.error(f"Error storing generated topics in Redis cache: {e}", exc_info=False)
            except Exception as e:
                logging.error(f"Unexpected error during cache storing: {e}", exc_info=True)
        else:
             logging.warning("Redis client not available. Skipping caching.")

        # 3. Start background tasks ONLY for the selected top topics
        logging.info(f"Starting background processing threads for {len(topics_to_process)} selected topics.")
        threads = []
        for topic_hash, topic_title in topics_to_process: # Iterate through the *selected* list
            logging.info(f"Creating background thread for topic: '{topic_title}' (Hash: {topic_hash})")
            thread = threading.Thread(
                target=process_topic_in_background,
                args=(topic_title,), # Pass only the title
                name=f"TopicThread-{topic_hash[:8]}" # Use part of hash for thread name
            )
            threads.append(thread)
            thread.start()
            logging.info(f"Background thread '{thread.name}' started for topic '{topic_title}'.")

        # 4. Return the list of *all* generated topics (with hashes) immediately
        logging.info(f"Returning list of all {len(hashed_topics)} generated topics to the client.")
        logging.debug(f"Data being returned: {hashed_topics}")
        logging.info("====== [/api/get_topics] Request Finished (Generated New Data) ======")
        # Important: Return all generated topics, even if only processing a subset
        return jsonify(hashed_topics), 200

    except Exception as e:
        logging.error(f"CRITICAL ERROR in /api/get_topics main logic: {e}", exc_info=True)
        logging.info("====== [/api/get_topics] Request Finished (Internal Server Error) ======")
        return jsonify({"error": "An internal server error occurred during topic generation"}), 500


@app.route("/api/receive_topic_news", methods=["POST"])
def receive_topic_news():
    """
    POST Endpoint: Receives the processed news URLs (and topic info)
    from background tasks. Logs received data. Add storage logic here.
    """
    logging.info("====== [/api/receive_topic_news] Endpoint Called (POST Request) ======")

    if not flask.request.is_json:
        logging.warning("Received non-JSON request to /api/receive_topic_news. Rejecting.")
        return jsonify({"error": "Request must be JSON"}), 400

    logging.debug("Request content type is JSON. Attempting to parse...")
    data = flask.request.get_json()
    logging.debug(f"Received raw JSON data: {data}")

    # Validate received data structure
    topic_title = data.get("topic_title")
    topic_hash = data.get("topic_hash") # Expecting hash now
    article_urls = data.get("article_urls")

    if not topic_title or not isinstance(topic_title, str):
        logging.warning(f"Received invalid or missing 'topic_title' in data: {data}")
        return jsonify({"error": "Missing or invalid 'topic_title' (must be a string)"}), 400
    if not topic_hash or not isinstance(topic_hash, str):
        logging.warning(f"Received invalid or missing 'topic_hash' in data: {data}")
        return jsonify({"error": "Missing or invalid 'topic_hash' (must be a string)"}), 400
    if not isinstance(article_urls, list): # Allow empty list, but must be a list
        logging.warning(f"Received invalid 'article_urls' (must be a list) in data: {data}")
        return jsonify({"error": "Missing or invalid 'article_urls' (must be a list)"}), 400

    logging.info(f"Successfully received and validated news data for Topic: '{topic_title}' (Hash: {topic_hash}), Found URLs: {len(article_urls)}")
    # Optionally log the URLs if needed for debugging (can be long)
    # logging.debug(f"Received URLs: {article_urls}")

    # --- Add your logic here to process/store the received data ---
    # Example: Store in another Redis key per topic hash, save to DB, etc.
    logging.info(f"Placeholder: Processing/storing data for topic '{topic_title}' (Hash: {topic_hash})...")
    # Example: Store URLs in Redis list keyed by topic hash
    # if redis_client:
    #     try:
    #         redis_key = f"topic_urls:{topic_hash}"
    #         # Clear previous entries and add new ones (or just append)
    #         redis_client.delete(redis_key)
    #         if article_urls:
    #             redis_client.rpush(redis_key, *article_urls)
    #         redis_client.expire(redis_key, CACHE_TIMEOUT_SECONDS * 2) # Keep URLs longer?
    #         logging.info(f"Stored {len(article_urls)} URLs in Redis key '{redis_key}'")
    #     except redis.exceptions.RedisError as e:
    #         logging.error(f"Failed to store URLs in Redis for hash {topic_hash}: {e}")
    # --- End of storage logic ---

    logging.info("====== [/api/receive_topic_news] Request Finished Successfully ======")
    return jsonify({"message": "Data received successfully"}), 200


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure environment variables are checked
    if "Redis_Key" not in os.environ:
        logging.warning("Environment variable 'Redis_Key' is not set. Using default/empty password for Redis connection.")
    if "Redis_Host" not in os.environ:
        logging.warning("Environment variable 'Redis_Host' is not set. Using 'localhost' for Redis connection.")

    host = '0.0.0.0' 
    port = 5001
    logging.info(f"Starting Flask server on {host}:{port} with threaded=True, debug=True, use_reloader=False")
    try:
        app.run(host=host, port=port, threaded=True, debug=True, use_reloader=False)
    except Exception as e:
        logging.critical(f"Failed to start Flask application: {e}", exc_info=True)
    finally:
        logging.info("Flask Application Has Been Shut Down")

# --- Updated Selenium Configuration ---
def get_selenium_driver():
    """Initialize and return a headless Selenium Chrome WebDriver."""
    options = Options()

    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")  # Standard window size
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")

    # Specify the path to your ChromeDriver executable
    CHROMEDRIVER_PATH = r"C:\Tools\chromedriver.exe"  # Update this path as needed

    # Initialize the Chrome WebDriver
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(30)  # Set page load timeout
    driver.implicitly_wait(5)  # Set implicit wait time
    return driver
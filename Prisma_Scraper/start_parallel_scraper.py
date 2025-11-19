#!/usr/bin/env python3
"""
Startup script for the Parallel News Scraper.
This script will guide you through setup and help you get started.
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print a nice banner."""
    banner = """
PARALLEL NEWS SCRAPER
=====================
* Ultra-fast news scraping with 16-core parallel processing
* Scrapes multiple news sources simultaneously  
* Up to 10x faster than sequential scraping
* Bias analysis and sentiment classification ready
    """
    print(banner)

def check_requirements():
    """Check if all required packages are installed."""
    print("Checking requirements...")
    
    # Map package names to their import names
    requirements = {
        'requests': 'requests',
        'python-dotenv': 'dotenv', 
        'playwright': 'playwright',
        'trafilatura': 'trafilatura',
        'pymongo': 'pymongo',
        'psutil': 'psutil'
    }
    
    missing = []
    for package_name, import_name in requirements.items():
        try:
            __import__(import_name)
            print(f"OK {package_name}")
        except ImportError:
            print(f"MISSING {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("All packages installed!")
    return True

def check_environment():
    """Check environment variables."""
    print("\nChecking environment configuration...")
    
    env_file = ".env"
    if not os.path.exists(env_file):
        print("MISSING .env file not found")
        print("Creating sample .env file...")
        
        sample_env = """# MongoDB Connection String
MONGO_URI=YOUR_MONGO_CONNECTION_STRING_HERE

# News API Key (get from https://newsapi.org/)
NEWS_API_KEY=YOUR_NEWS_API_KEY_HERE
"""
        with open(env_file, 'w') as f:
            f.write(sample_env)
        
        print("OK Created .env file. Please edit it with your credentials.")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    mongo_uri = os.getenv("MONGO_URI")
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not mongo_uri or mongo_uri == "YOUR_MONGO_CONNECTION_STRING_HERE":
        print("MISSING MONGO_URI not configured")
        return False
    else:
        print("OK MONGO_URI configured")
    
    if not news_api_key or news_api_key == "YOUR_NEWS_API_KEY_HERE":
        print("MISSING NEWS_API_KEY not configured")
        return False
    else:
        print("OK NEWS_API_KEY configured")
    
    return True

def install_playwright():
    """Install Playwright browsers."""
    print("\nInstalling Playwright browsers...")
    
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      check=True, capture_output=True)
        print("OK Playwright browsers installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR Failed to install Playwright browsers: {e}")
        return False
    except FileNotFoundError:
        print("ERROR Playwright not found. Install it first: pip install playwright")
        return False

def run_tests():
    """Run the test suite."""
    print("\nRunning tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_parallel_scraper.py"], 
                               capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except FileNotFoundError:
        print("X test_parallel_scraper.py not found")
        return False

def show_performance_info():
    """Show performance information."""
    print("\nPerformance Information:")
    
    try:
        result = subprocess.run([sys.executable, "performance_monitor.py"], 
                               capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("X performance_monitor.py not found")

def main():
    """Main setup function."""
    print_banner()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\nPlease install missing packages and run this script again.")
        return False
    
    # Step 2: Check environment
    if not check_environment():
        print("\nPlease configure your .env file and run this script again.")
        return False
    
    # Step 3: Install Playwright
    if not install_playwright():
        print("\nPlaywright installation failed.")
        return False
    
    # Step 4: Run tests
    if not run_tests():
        print("\nTests failed. Please check the configuration.")
        return False
    
    # Step 5: Show performance info
    show_performance_info()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("\nReady to start scraping!")
    print("Commands:")
    print("  python news_fetcher.py           # Start the parallel scraper")
    print("  python test_parallel_scraper.py  # Run tests")
    print("  python performance_monitor.py    # Check performance estimates")
    print("  python config_parallel.py        # View/adjust configuration")
    
    print("\nPro Tips:")
    print("  • Monitor system resources during first run")
    print("  • Adjust NUM_CORES in config_parallel.py if needed")
    print("  • Check logs for any errors or warnings")
    print("  • The scraper will run continuously - use Ctrl+C to stop")
    
    # Ask if user wants to start scraping
    print("\n" + "="*50)
    response = input("Start scraping now? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\nStarting parallel news scraper...")
        print("Press Ctrl+C to stop scraping\n")
        
        try:
            os.system(f"{sys.executable} news_fetcher.py")
        except KeyboardInterrupt:
            print("\nScraping stopped by user.")
    else:
        print("\nSetup complete. Run 'python news_fetcher.py' when ready!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)

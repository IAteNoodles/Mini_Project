#!/usr/bin/env python3
"""
Database management utility for the news scraper
"""

import os
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_local_database():
    """Clear the local JSON database"""
    db_file = "local_articles_backup.json"
    if os.path.exists(db_file):
        # Backup the current file
        backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.rename(db_file, backup_file)
        logging.info(f"✅ Backed up existing database to {backup_file}")
    
    # Create empty database
    empty_db = {
        "articles": [],
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_articles": 0
        }
    }
    
    with open(db_file, 'w', encoding='utf-8') as f:
        json.dump(empty_db, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✅ Created fresh local database: {db_file}")

def show_database_stats():
    """Show database statistics"""
    try:
        from fixed_indian_scraper import DatabaseManager
        
        db_manager = DatabaseManager()
        existing_urls = db_manager.get_existing_urls()
        
        logging.info(f"📊 Database Statistics:")
        logging.info(f"   - Total articles: {len(existing_urls)}")
        logging.info(f"   - Using MongoDB: {db_manager.use_mongodb}")
        
        if hasattr(db_manager, 'local_db'):
            local_count = db_manager.local_db.count_articles()
            logging.info(f"   - Local articles: {local_count}")
        
        # Show sample URLs
        if existing_urls:
            logging.info(f"📄 Sample URLs (first 5):")
            for i, url in enumerate(list(existing_urls)[:5]):
                logging.info(f"   {i+1}. {url}")
        
    except Exception as e:
        logging.error(f"❌ Error reading database: {e}")

def clear_mongodb():
    """Clear MongoDB collection (if connected)"""
    try:
        from fixed_indian_scraper import DatabaseManager
        
        db_manager = DatabaseManager()
        if db_manager.use_mongodb and db_manager.collection is not None:
            count = db_manager.collection.count_documents({})
            result = db_manager.collection.delete_many({})
            logging.info(f"✅ Cleared {result.deleted_count} articles from MongoDB")
        else:
            logging.info("❌ MongoDB not connected")
            
    except Exception as e:
        logging.error(f"❌ Error clearing MongoDB: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("📋 Database Management Utility")
        print("Usage:")
        print("  python db_utils.py stats          - Show database statistics")
        print("  python db_utils.py clear-local    - Clear local JSON database")
        print("  python db_utils.py clear-mongo    - Clear MongoDB collection")
        print("  python db_utils.py clear-all      - Clear both databases")
        return
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        show_database_stats()
    elif command == "clear-local":
        clear_local_database()
    elif command == "clear-mongo":
        clear_mongodb()
    elif command == "clear-all":
        clear_local_database()
        clear_mongodb()
    else:
        logging.error(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()

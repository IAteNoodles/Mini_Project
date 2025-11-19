#!/usr/bin/env python3
"""
Local MongoDB setup for development when Atlas is blocked
"""

import os
import json
from datetime import datetime

class LocalJSONDatabase:
    """Simple JSON-based local database as MongoDB alternative"""
    
    def __init__(self, db_file="local_articles.json"):
        self.db_file = db_file
        self.data = self._load_data()
    
    def _load_data(self):
        """Load existing data or create new file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"articles": []}
        return {"articles": []}
    
    def _save_data(self):
        """Save data to JSON file"""
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def find_urls(self):
        """Get all existing URLs"""
        return {article['url'] for article in self.data['articles']}
    
    def insert_article(self, url, headline, content, site):
        """Insert new article"""
        article = {
            'url': url,
            'headline': headline,
            'content': content,
            'site': site,
            'scraped_at': datetime.now().isoformat()
        }
        self.data['articles'].append(article)
        self._save_data()
        return True
    
    def count_articles(self):
        """Count total articles"""
        return len(self.data['articles'])
    
    def get_recent_articles(self, limit=10):
        """Get recent articles"""
        return self.data['articles'][-limit:]

# Test the local database
if __name__ == "__main__":
    print("🗃️ Setting up Local JSON Database...")
    
    db = LocalJSONDatabase()
    count = db.count_articles()
    
    print(f"✅ Local database ready!")
    print(f"   File: {os.path.abspath(db.db_file)}")
    print(f"   Current articles: {count}")
    
    # Test insert
    db.insert_article(
        url="https://example.com/test",
        headline="Test Article",
        content="This is a test article",
        site="test.com"
    )
    
    print(f"✅ Test article added. Total: {db.count_articles()}")
    print(f"📄 Recent articles:")
    for article in db.get_recent_articles(3):
        print(f"   - {article['headline']} ({article['site']})")

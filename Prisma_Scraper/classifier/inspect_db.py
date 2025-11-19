#!/usr/bin/env python3
"""
Quick database inspector to check article content
"""
import os
import sys
from pathlib import Path
import pymongo

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def load_environment():
    """Load environment variables from .env file"""
    env_file = parent_dir / ".env"
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"\'')
                    env_vars[key.strip()] = value
                    os.environ[key.strip()] = value
    
    return env_vars

def main():
    print("🔍 Database Article Inspector")
    print("=" * 40)
    
    # Load environment and connect
    env_vars = load_environment()
    mongodb_url = env_vars.get("MONGODB_URL")
    
    client = pymongo.MongoClient(mongodb_url)
    db = client["Prisma"]
    collection = db["articles"]
    
    print("📊 Checking article content...")
    
    # Check total count
    total = collection.count_documents({})
    processed = collection.count_documents({"processed": True})
    print(f"Total articles: {total}")
    print(f"Processed: {processed}")
    
    # Check articles with content
    with_content = collection.count_documents({"article": {"$exists": True, "$ne": "", "$ne": None}})
    without_content = collection.count_documents({"$or": [{"article": {"$exists": False}}, {"article": ""}, {"article": None}]})
    
    print(f"With content: {with_content}")
    print(f"Without content: {without_content}")
    
    # Sample some articles with content
    print("\n🔍 Sample articles with content:")
    articles_with_content = list(collection.find(
        {
            "article": {"$exists": True, "$ne": "", "$ne": None},
            "processed": {"$ne": True}
        },
        {"_id": 1, "url": 1, "article": 1}
    ).limit(5))
    
    for i, article in enumerate(articles_with_content, 1):
        content = article.get("article", "")
        url = article.get("url", "Unknown")
        print(f"\n{i}. URL: {url[:80]}...")
        print(f"   Content length: {len(content)} characters")
        print(f"   Content preview: {content[:100]}...")
    
    # Check field names
    print(f"\n📋 Sample document structure:")
    sample = collection.find_one()
    if sample:
        keys = list(sample.keys())
        print(f"Available fields: {keys}")
        
        # Check if content is in different field
        for key in keys:
            if 'content' in key.lower() or 'text' in key.lower() or 'body' in key.lower():
                value = sample.get(key, "")
                if isinstance(value, str) and len(value) > 50:
                    print(f"   {key}: {len(value)} chars - {value[:100]}...")

if __name__ == "__main__":
    main()

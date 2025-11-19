#!/usr/bin/env python3
"""
Test MongoDB Atlas connection and check processed articles
"""
import os
import sys
from pathlib import Path

# Load environment variables
env_file = Path('.env')
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                os.environ[key.strip()] = value

from pymongo import MongoClient

def test_connection():
    # Use the MongoDB URL from environment
    mongodb_url = os.getenv('MONGODB_URL')
    print(f'Connecting to: {mongodb_url[:50]}...')
    
    try:
        client = MongoClient(mongodb_url)
        db = client['Prisma']
        collection = db['processed_Articles']
        
        # Test connection
        client.admin.command('ismaster')
        print('✅ Connected to MongoDB Atlas successfully!')
        
        # Get the most recent processed article
        recent = list(collection.find().sort('_id', -1).limit(1))
        if recent:
            doc = recent[0]
            print(f'📄 Most recent article:')
            print(f'   Article ID: {doc["_id"]}')
            print(f'   Summary length: {len(doc.get("summary", ""))} characters')
            print(f'   Summary word count: {len(doc.get("summary", "").split())} words')
            print(f'   Confidence: {doc.get("confidence_score", "N/A")}')
            print(f'   Model used: {doc.get("model_used", "N/A")}')
            print(f'   Summary preview: {doc.get("summary", "N/A")[:200]}...')
            print(f'   Bias detected: {doc.get("bias_mode", "N/A")}')
        else:
            print('No processed articles found')
            
        # Get collection stats
        total_processed = collection.count_documents({})
        print(f'📊 Total processed articles: {total_processed}')
        
        # Get stats from original collection too
        original_collection = db['articles']
        total_original = original_collection.count_documents({})
        processed_original = original_collection.count_documents({"processed": True})
        
        print(f'📰 Original collection stats:')
        print(f'   Total articles: {total_original}')
        print(f'   Processed: {processed_original}')
        print(f'   Unprocessed: {total_original - processed_original}')
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    test_connection()

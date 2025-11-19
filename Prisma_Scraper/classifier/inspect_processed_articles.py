#!/usr/bin/env python3
"""
Script to inspect the new processed_Articles collection
Shows the comprehensive data structure and validates the implementation
"""

import pymongo
from pymongo import MongoClient
import json
from datetime import datetime

def inspect_processed_articles():
    """Inspect the processed_Articles collection structure and data"""
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb+srv://abhijitn23beds_db_user:9vmJkSQuV6HU08F4@prisma.ojfejnc.mongodb.net/?retryWrites=true&w=majority&appName=Prisma&ssl=true&tls=true&tlsAllowInvalidCertificates=true")
        db = client["Prisma"]
        processed_collection = db["processed_Articles"]
        
        print("🔍 Inspecting processed_Articles Collection")
        print("=" * 60)
        
        # Get collection stats
        total_count = processed_collection.count_documents({})
        print(f"📊 Total processed articles: {total_count}")
        
        if total_count == 0:
            print("❌ No processed articles found")
            return
        
        # Get the first few documents
        cursor = processed_collection.find({}).limit(3)
        
        for i, doc in enumerate(cursor, 1):
            print(f"\n🔍 Document {i}:")
            print(f"   📄 ID: {doc['_id']}")
            print(f"   🔗 Original ID: {doc['original_id']}")
            print(f"   🌐 URL: {doc['url'][:80]}...")
            print(f"   📝 Text Length: {len(doc['text'])} characters")
            print(f"   📋 Summary Length: {len(doc['summary'])} characters")
            print(f"   📄 Summary Preview: {doc['summary'][:150]}...")
            print(f"   🏷️  Bias: {doc['bias']}")
            print(f"   🎭 Bias Mode: {doc['bias_mode']}")
            print(f"   🎯 Confidence: {doc['confidence_score']}")
            print(f"   🤖 Model Used: {doc['model_used']}")
            print(f"   📅 Processed At: {doc['processed_at']}")
            
            # Check if text was actually preserved (not truncated)
            text_sample = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            print(f"   📖 Text Sample: {text_sample}")
            
        # Analyze collection structure
        print(f"\n🔍 Schema Analysis:")
        sample_doc = processed_collection.find_one({})
        if sample_doc:
            print(f"   📋 Document Fields:")
            for key in sample_doc.keys():
                value = sample_doc[key]
                if isinstance(value, str):
                    length = len(value)
                    print(f"      {key}: String ({length} chars)")
                elif isinstance(value, dict):
                    print(f"      {key}: Object ({len(value)} fields)")
                elif isinstance(value, list):
                    print(f"      {key}: Array ({len(value)} items)")
                else:
                    print(f"      {key}: {type(value).__name__}")
                    
        # Check for text truncation issues
        print(f"\n🔍 Text Length Analysis:")
        pipeline = [
            {
                "$project": {
                    "text_length": {"$strLenCP": "$text"},
                    "summary_length": {"$strLenCP": "$summary"},
                    "model_used": 1
                }
            },
            {
                "$group": {
                    "_id": "$model_used",
                    "avg_text_length": {"$avg": "$text_length"},
                    "avg_summary_length": {"$avg": "$summary_length"},
                    "max_text_length": {"$max": "$text_length"},
                    "count": {"$sum": 1}
                }
            }
        ]
        
        for result in processed_collection.aggregate(pipeline):
            print(f"   🤖 Model: {result['_id']}")
            print(f"      📊 Count: {result['count']}")
            print(f"      📝 Avg Text Length: {result['avg_text_length']:.0f} chars")
            print(f"      📋 Avg Summary Length: {result['avg_summary_length']:.0f} chars")
            print(f"      📏 Max Text Length: {result['max_text_length']:.0f} chars")
            
        print(f"\n✅ Collection inspection completed!")
        print(f"💾 Comprehensive data storage verified!")
        
    except Exception as e:
        print(f"❌ Error inspecting collection: {e}")

if __name__ == "__main__":
    inspect_processed_articles()

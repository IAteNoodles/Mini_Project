#!/usr/bin/env python3
"""
Verification script to ensure complete article text is preserved in processed_Articles collection
"""

import pymongo
from pymongo import MongoClient

def verify_complete_article_storage():
    """Verify that complete article text is being saved correctly"""
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["Prisma"]
        
        original_collection = db["articles"]
        processed_collection = db["processed_Articles"]
        
        print("🔍 Verifying Complete Article Text Storage")
        print("=" * 60)
        
        # Get some processed articles
        processed_articles = list(processed_collection.find({}).limit(5))
        
        if not processed_articles:
            print("❌ No processed articles found")
            return
        
        print(f"📊 Found {len(processed_articles)} processed articles to verify")
        
        verification_passed = 0
        verification_failed = 0
        
        for i, processed_article in enumerate(processed_articles, 1):
            print(f"\n🔍 Verifying Article {i}:")
            
            # Get original article
            original_id = processed_article.get('original_id')
            original_article = original_collection.find_one({"_id": original_id})
            
            if not original_article:
                print(f"   ❌ Original article not found")
                verification_failed += 1
                continue
            
            # Compare text lengths
            original_text = original_article.get('text', '')
            processed_text = processed_article.get('text', '')
            stored_length = processed_article.get('article_length', 0)
            
            print(f"   📏 Original text length: {len(original_text)} characters")
            print(f"   📏 Processed text length: {len(processed_text)} characters")
            print(f"   📏 Stored length field: {stored_length} characters")
            
            # Verify text integrity
            if len(original_text) == len(processed_text) == stored_length:
                print(f"   ✅ Text length verification PASSED")
                
                # Sample text comparison
                if original_text[:100] == processed_text[:100]:
                    print(f"   ✅ Text content verification PASSED")
                    verification_passed += 1
                else:
                    print(f"   ❌ Text content mismatch detected")
                    verification_failed += 1
            else:
                print(f"   ❌ Text length mismatch detected")
                print(f"      Expected: {len(original_text)}")
                print(f"      Got: {len(processed_text)}")
                print(f"      Stored: {stored_length}")
                verification_failed += 1
            
            # Show URL for reference
            url = processed_article.get('url', 'Unknown')
            print(f"   🔗 URL: {url[:80]}...")
        
        print(f"\n📊 Verification Results:")
        print(f"   ✅ Passed: {verification_passed}")
        print(f"   ❌ Failed: {verification_failed}")
        print(f"   📈 Success Rate: {(verification_passed/(verification_passed+verification_failed)*100):.1f}%")
        
        if verification_passed == len(processed_articles):
            print(f"\n🎉 All articles verified successfully!")
            print(f"💾 Complete article text is properly preserved in processed_Articles collection")
        else:
            print(f"\n⚠️  Some verification issues detected")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    verify_complete_article_storage()

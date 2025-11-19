#!/usr/bin/env python3
"""
Lightweight MongoDB test and article analysis without heavy ML dependencies
"""
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def load_environment():
    """Load environment variables manually"""
    env_file = current_dir / ".env"
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"\'')
                    env_vars[key.strip()] = value
                    os.environ[key.strip()] = value
    
    return env_vars

class SimpleArticleInput:
    """Simple article input representation"""
    def __init__(self, url: str, article: str):
        self.url = url
        self.article = article

class SimpleBiasClassification:
    """Simple bias classification representation"""
    def __init__(self, political: int = 0, gender: int = 0, cultural: int = 0, ideology: int = 0):
        self.political = political
        self.gender = gender
        self.cultural = cultural
        self.ideology = ideology
    
    def to_dict(self):
        return {
            "political": self.political,
            "gender": self.gender,
            "cultural": self.cultural,
            "ideology": self.ideology
        }

class SimpleArticleOutput:
    """Simple article output representation"""
    def __init__(self, url: str, article: str, summary: str, bias: SimpleBiasClassification, confidence_score: float = 0.0):
        self.url = url
        self.article = article
        self.summary = summary
        self.bias = bias
        self.confidence_score = confidence_score
        self.processed_at = datetime.now()
    
    def to_dict(self):
        return {
            "url": self.url,
            "article": self.article,
            "summary": self.summary,
            "bias": self.bias.to_dict(),
            "confidence_score": self.confidence_score,
            "processed_at": self.processed_at.isoformat(),
            "processed": True
        }

class SimpleBiasClassifier:
    """Regex-based bias classifier"""
    
    def __init__(self):
        self.political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign|trump|biden)\b',
        ]
        
        self.gender_patterns = [
            r'\b(men|women|male|female|masculine|feminine|gender|sex|sexist|feminist)\b',
        ]
        
        self.cultural_patterns = [
            r'\b(culture|cultural|ethnic|race|racial|religion|religious|immigrant|foreigner|native|traditional|western|eastern|muslim|christian|jewish|hindu|buddhist)\b',
        ]
        
        self.ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin|angle)\b',
        ]
    
    def classify(self, text: str) -> SimpleBiasClassification:
        """Classify text for bias using regex patterns"""
        text_lower = text.lower()
        
        political = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.political_patterns) else 0
        
        gender = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                         for pattern in self.gender_patterns) else 0
        
        cultural = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.cultural_patterns) else 0
        
        ideology = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.ideology_patterns) else 0
        
        return SimpleBiasClassification(
            political=political,
            gender=gender,
            cultural=cultural,
            ideology=ideology
        )

class SimpleMongoClient:
    """Simple MongoDB client without heavy dependencies"""
    
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            import pymongo
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            self.client.admin.command('ismaster')
            return True
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return False
    
    def get_articles(self, limit: int = 10, skip: int = 0, filter_query: Dict = None) -> List[Dict]:
        """Get articles from MongoDB"""
        try:
            if self.collection is None:
                raise Exception("Not connected to MongoDB")
            
            query = filter_query or {}
            cursor = self.collection.find(query).skip(skip).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"Error getting articles: {e}")
            return []
    
    def update_article(self, article_id: str, update_data: Dict) -> bool:
        """Update an article in MongoDB"""
        try:
            if self.collection is None:
                raise Exception("Not connected to MongoDB")
            
            from bson import ObjectId
            result = self.collection.update_one(
                {"_id": ObjectId(article_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating article: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            if self.collection is None:
                raise Exception("Not connected to MongoDB")
            
            total_count = self.collection.count_documents({})
            processed_count = self.collection.count_documents({"processed": True})
            unprocessed_count = total_count - processed_count
            
            return {
                "total_articles": total_count,
                "processed": processed_count,
                "unprocessed": unprocessed_count
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

class SimpleProcessor:
    """Simple article processor"""
    
    def __init__(self, mongo_client: SimpleMongoClient):
        self.mongo_client = mongo_client
        self.bias_classifier = SimpleBiasClassifier()
    
    def create_summary(self, text: str, max_length: int = 200) -> str:
        """Create a simple summary by taking first few sentences"""
        sentences = text.split('.')
        summary = ""
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            if len(summary) + len(sentence) < max_length:
                summary += sentence.strip() + ". "
            else:
                break
        
        return summary.strip()
    
    def analyze_article(self, article_data: Dict) -> SimpleArticleOutput:
        """Analyze an article using simple methods"""
        
        article_input = SimpleArticleInput(
            url=article_data.get("url", ""),
            article=article_data.get("article", "")
        )
        
        # Create summary
        summary = self.create_summary(article_input.article)
        
        # Classify bias
        bias = self.bias_classifier.classify(article_input.article)
        
        # Calculate confidence (simple heuristic)
        confidence = 0.6 if len(article_input.article) > 100 else 0.3
        
        return SimpleArticleOutput(
            url=article_input.url,
            article=article_input.article,
            summary=summary,
            bias=bias,
            confidence_score=confidence
        )
    
    def process_articles(self, limit: int = 10) -> Dict[str, Any]:
        """Process unprocessed articles"""
        
        print(f"🔄 Processing up to {limit} articles...")
        
        # Get unprocessed articles
        articles = self.mongo_client.get_articles(
            limit=limit,
            filter_query={"processed": {"$ne": True}}
        )
        
        if not articles:
            return {"message": "No unprocessed articles found", "processed": 0}
        
        processed_count = 0
        successful_count = 0
        
        for article in articles:
            try:
                # Analyze article
                result = self.analyze_article(article)
                
                # Update in database
                update_data = {
                    "summary": result.summary,
                    "bias": result.bias.to_dict(),
                    "confidence_score": result.confidence_score,
                    "processed_at": result.processed_at.isoformat(),
                    "processed": True
                }
                
                article_id = str(article["_id"])
                if self.mongo_client.update_article(article_id, update_data):
                    successful_count += 1
                    print(f"✅ Processed: {result.url[:50]}...")
                else:
                    print(f"❌ Failed to update: {result.url[:50]}...")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing article: {e}")
        
        return {
            "processed": processed_count,
            "successful": successful_count,
            "failed": processed_count - successful_count
        }

def main():
    """Main function"""
    print("🚀 Simple Prisma Scraper Test")
    print("=" * 50)
    
    # Load environment
    print("⚙️  Loading environment...")
    env_vars = load_environment()
    
    mongodb_url = env_vars.get("MONGODB_URL")
    if not mongodb_url:
        print("❌ MONGODB_URL not found in environment")
        return
    
    database_name = env_vars.get("MONGODB_DATABASE", "Prisma")
    collection_name = env_vars.get("MONGODB_COLLECTION", "articles")
    
    print(f"📊 Database: {database_name}")
    print(f"📋 Collection: {collection_name}")
    
    # Connect to MongoDB
    print("\n🔗 Connecting to MongoDB...")
    mongo_client = SimpleMongoClient(mongodb_url, database_name, collection_name)
    
    if not mongo_client.connect():
        print("❌ Failed to connect to MongoDB")
        return
    
    print("✅ Connected to MongoDB successfully!")
    
    # Get statistics
    print("\n📊 Getting collection statistics...")
    stats = mongo_client.get_stats()
    if stats:
        print(f"📰 Total articles: {stats['total_articles']}")
        print(f"✅ Processed: {stats['processed']}")
        print(f"⏳ Unprocessed: {stats['unprocessed']}")
    
    # Test article processing
    if stats.get('unprocessed', 0) > 0:
        print(f"\n🧪 Testing article processing...")
        processor = SimpleProcessor(mongo_client)
        
        # Process a small sample
        result = processor.process_articles(limit=3)
        
        print(f"\n🎯 Processing Results:")
        print(f"📝 Processed: {result['processed']}")
        print(f"✅ Successful: {result['successful']}")
        print(f"❌ Failed: {result['failed']}")
        
        # Get updated stats
        new_stats = mongo_client.get_stats()
        if new_stats:
            print(f"\n📊 Updated Statistics:")
            print(f"📰 Total articles: {new_stats['total_articles']}")
            print(f"✅ Processed: {new_stats['processed']}")
            print(f"⏳ Unprocessed: {new_stats['unprocessed']}")
    
    else:
        print("\n💡 No unprocessed articles found. All articles have been processed!")
    
    print("\n🏁 Test completed!")
    print("\n📝 Next Steps:")
    print("1. This lightweight version successfully connects to MongoDB")
    print("2. It can process articles using regex-based bias detection")
    print("3. For advanced LLM analysis, resolve the PyTorch/Python compatibility issues")
    print("4. Consider using Python 3.11 or 3.12 for better package compatibility")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Article Summarizer using OpenAI API (fallback for local model issues)
"""
import os
import sys
from pathlib import Path
import pymongo
from datetime import datetime
import re
from typing import Dict, List, Optional
import json

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

class OpenAIArticleSummarizer:
    """Article summarization using OpenAI API"""
    
    def __init__(self):
        self.mongo_client = None
        self.collection = None
        self.openai_client = None
        
        print("🚀 Initializing OpenAI Article Summarizer")
        
    def setup_openai(self):
        """Setup OpenAI client"""
        try:
            # Check if OpenAI is available
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  OPENAI_API_KEY not found. You can:")
                print("1. Add OPENAI_API_KEY to your .env file")
                print("2. Set it as environment variable")
                print("3. Use the regex-based summarizer instead")
                return False
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized")
            return True
            
        except ImportError:
            print("❌ OpenAI library not installed. Install with: pip install openai")
            return False
        except Exception as e:
            print(f"❌ Error setting up OpenAI: {e}")
            return False
    
    def connect_to_mongodb(self):
        """Connect to MongoDB database"""
        try:
            env_vars = load_environment()
            mongodb_url = env_vars.get("MONGODB_URL")
            database_name = env_vars.get("MONGODB_DATABASE", "Prisma")
            collection_name = env_vars.get("MONGODB_COLLECTION", "articles")
            
            if not mongodb_url:
                raise Exception("MONGODB_URL not found in environment")
            
            print(f"🔗 Connecting to MongoDB...")
            print(f"📊 Database: {database_name}")
            print(f"📋 Collection: {collection_name}")
            
            self.mongo_client = pymongo.MongoClient(mongodb_url)
            self.db = self.mongo_client[database_name]
            self.collection = self.db[collection_name]
            
            # Test connection
            self.mongo_client.admin.command('ismaster')
            print("✅ Connected to MongoDB successfully!")
            return True
            
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            return False
    
    def generate_summary_openai(self, article_text: str, url: str = "") -> str:
        """Generate summary using OpenAI API"""
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            
            # Truncate if too long (OpenAI has token limits)
            if len(article_text) > 3000:
                article_text = article_text[:3000] + "..."
            
            prompt = f"""You are an expert journalist. Create a comprehensive, detailed summary of this news article.

INSTRUCTIONS:
- Capture ALL key information, facts, quotes, and context
- Include important details like names, dates, numbers, locations
- Maintain journalistic objectivity
- Make the summary 4-6 sentences long
- Focus on Who, What, When, Where, Why, and How

ARTICLE:
{article_text}

SUMMARY:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert journalist who creates comprehensive, accurate news summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary if summary else "Failed to generate summary"
            
        except Exception as e:
            print(f"❌ Error generating OpenAI summary: {e}")
            return f"OpenAI Error: {str(e)}"
    
    def generate_summary_regex(self, article_text: str, url: str = "") -> str:
        """Generate simple summary using extractive method"""
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', article_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return "No content available for summarization"
            
            # Take first 3 sentences as summary (simple extractive approach)
            summary_sentences = sentences[:3]
            summary = '. '.join(summary_sentences)
            
            # Ensure it ends with a period
            if summary and not summary.endswith('.'):
                summary += '.'
            
            return summary if len(summary) > 10 else "Summary generation failed"
            
        except Exception as e:
            return f"Regex summary error: {e}"
    
    def classify_bias(self, article_text: str) -> Dict[str, int]:
        """Classify bias using regex patterns"""
        text_lower = article_text.lower()
        
        political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign|trump|biden)\b',
        ]
        
        gender_patterns = [
            r'\b(men|women|male|female|masculine|feminine|gender|sex|sexist|feminist|he|she|his|her)\b',
        ]
        
        cultural_patterns = [
            r'\b(culture|cultural|ethnic|race|racial|religion|religious|immigrant|foreigner|native|traditional|muslim|christian|jewish|hindu|buddhist)\b',
        ]
        
        ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin)\b',
        ]
        
        political = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in political_patterns) else 0
        gender = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in gender_patterns) else 0
        cultural = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in cultural_patterns) else 0
        ideology = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in ideology_patterns) else 0
        
        return {
            "political": political,
            "gender": gender,
            "cultural": cultural,
            "ideology": ideology
        }
    
    def process_articles(self, limit: int = 10, use_openai: bool = True) -> Dict:
        """Process unprocessed articles from MongoDB"""
        if not self.collection:
            return {"error": "Not connected to MongoDB"}
        
        print(f"🔄 Processing up to {limit} articles...")
        print(f"🤖 Using: {'OpenAI API' if use_openai and self.openai_client else 'Regex-based summarization'}")
        
        # Get unprocessed articles
        articles = list(self.collection.find(
            {"processed": {"$ne": True}},
            {"_id": 1, "url": 1, "article": 1}
        ).limit(limit))
        
        if not articles:
            return {"message": "No unprocessed articles found", "processed": 0}
        
        processed_count = 0
        successful_count = 0
        failed_count = 0
        
        for i, article in enumerate(articles, 1):
            try:
                print(f"📝 Processing article {i}/{len(articles)}: {article.get('url', 'Unknown URL')[:60]}...")
                
                article_text = article.get("article", "")
                url = article.get("url", "")
                
                if not article_text:
                    print(f"⚠️  Skipping article with no content")
                    continue
                
                # Generate summary
                if use_openai and self.openai_client:
                    summary = self.generate_summary_openai(article_text, url)
                    method_used = "openai_gpt3.5"
                else:
                    summary = self.generate_summary_regex(article_text, url)
                    method_used = "regex_extractive"
                
                # Classify bias
                bias = self.classify_bias(article_text)
                
                # Calculate confidence score
                if "Error:" in summary or "Failed" in summary:
                    confidence = 0.2
                elif use_openai and self.openai_client and len(summary) > 50:
                    confidence = 0.85
                elif len(summary) > 50:
                    confidence = 0.6
                else:
                    confidence = 0.3
                
                # Update article in database
                update_data = {
                    "summary": summary,
                    "bias": bias,
                    "confidence_score": confidence,
                    "processed_at": datetime.now().isoformat(),
                    "processed": True,
                    "model_used": method_used
                }
                
                result = self.collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    successful_count += 1
                    print(f"✅ Successfully processed and updated")
                    print(f"📄 Summary: {summary[:100]}...")
                    print(f"🏷️  Bias detected: {[k for k, v in bias.items() if v == 1]}")
                else:
                    failed_count += 1
                    print(f"❌ Failed to update database")
                
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Error processing article: {e}")
        
        return {
            "processed": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": f"{(successful_count/processed_count*100):.1f}%" if processed_count > 0 else "0%"
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the article collection"""
        if not self.collection:
            return {"error": "Not connected to MongoDB"}
        
        try:
            total_count = self.collection.count_documents({})
            processed_count = self.collection.count_documents({"processed": True})
            unprocessed_count = total_count - processed_count
            
            # Get bias statistics for processed articles
            bias_stats = {}
            if processed_count > 0:
                pipeline = [
                    {"$match": {"processed": True, "bias": {"$exists": True}}},
                    {"$group": {
                        "_id": None,
                        "political_count": {"$sum": "$bias.political"},
                        "gender_count": {"$sum": "$bias.gender"},
                        "cultural_count": {"$sum": "$bias.cultural"},
                        "ideology_count": {"$sum": "$bias.ideology"}
                    }}
                ]
                
                result = list(self.collection.aggregate(pipeline))
                if result:
                    bias_data = result[0]
                    bias_stats = {
                        "political": bias_data.get("political_count", 0),
                        "gender": bias_data.get("gender_count", 0),
                        "cultural": bias_data.get("cultural_count", 0),
                        "ideology": bias_data.get("ideology_count", 0)
                    }
            
            return {
                "total_articles": total_count,
                "processed": processed_count,
                "unprocessed": unprocessed_count,
                "processed_percentage": f"{(processed_count/total_count*100):.1f}%" if total_count > 0 else "0%",
                "bias_statistics": bias_stats
            }
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}

def main():
    """Main function to run article summarization"""
    print("🚀 Advanced Article Summarizer")
    print("=" * 50)
    
    # Initialize summarizer
    summarizer = OpenAIArticleSummarizer()
    
    # Connect to MongoDB
    if not summarizer.connect_to_mongodb():
        print("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Get initial statistics
    stats = summarizer.get_collection_stats()
    if "error" not in stats:
        print(f"\n📊 Collection Statistics:")
        print(f"📰 Total articles: {stats['total_articles']}")
        print(f"✅ Processed: {stats['processed']} ({stats['processed_percentage']})")
        print(f"⏳ Unprocessed: {stats['unprocessed']}")
        
        if stats.get('bias_statistics'):
            print(f"\n🏷️  Bias Statistics (Processed Articles):")
            for bias_type, count in stats['bias_statistics'].items():
                print(f"   {bias_type.capitalize()}: {count} articles")
    
    # Setup OpenAI (optional)
    use_openai = summarizer.setup_openai()
    
    if not use_openai:
        print("\n🔄 Proceeding with regex-based summarization...")
    
    # Process articles
    print(f"\n🔄 Starting article processing...")
    limit = int(input("Enter number of articles to process (default 5): ") or "5")
    
    result = summarizer.process_articles(limit=limit, use_openai=use_openai)
    
    print(f"\n🎯 Processing Results:")
    print(f"📝 Processed: {result.get('processed', 0)}")
    print(f"✅ Successful: {result.get('successful', 0)}")
    print(f"❌ Failed: {result.get('failed', 0)}")
    print(f"📊 Success Rate: {result.get('success_rate', '0%')}")
    
    # Get updated statistics
    updated_stats = summarizer.get_collection_stats()
    if "error" not in updated_stats:
        print(f"\n📊 Updated Statistics:")
        print(f"📰 Total articles: {updated_stats['total_articles']}")
        print(f"✅ Processed: {updated_stats['processed']} ({updated_stats['processed_percentage']})")
        print(f"⏳ Unprocessed: {updated_stats['unprocessed']}")
        
        if updated_stats.get('bias_statistics'):
            print(f"\n🏷️  Updated Bias Statistics:")
            for bias_type, count in updated_stats['bias_statistics'].items():
                print(f"   {bias_type.capitalize()}: {count} articles")
    
    print(f"\n🏁 Article summarization completed!")
    print(f"\n💡 Tips:")
    print(f"- Add OPENAI_API_KEY to .env for better summaries")
    print(f"- Current method provides good regex-based summaries")
    print(f"- All summaries include bias classification")

if __name__ == "__main__":
    main()

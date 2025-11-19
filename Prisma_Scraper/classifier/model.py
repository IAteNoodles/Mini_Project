from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
from pathlib import Path
import pymongo
from datetime import datetime
import re
from typing import Dict, List, Optional

# Add parent directory to path to access config
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

class ArticleSummarizer:
    """Article summarization using Mistral 7B model"""
    
    def __init__(self, model_name: str = "unsloth/mistral-7b-v0.2-bnb-4bit"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.mongo_client = None
        
        print(f"🚀 Initializing ArticleSummarizer with {model_name}")
        print(f"🔧 Device: {self.device}")
        
    def load_model(self):
        """Load the Mistral model and tokenizer"""
        try:
            print(f"📦 Loading tokenizer and model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
    
    def create_summarization_prompt(self, article_text: str, url: str = "") -> str:
        """Create a prompt for article summarization"""
        
        # Truncate article if too long (keep first 2000 characters)
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."
        
        prompt = f"""You are an expert journalist tasked with creating comprehensive summaries of news articles.

INSTRUCTIONS:
- Create a detailed, informative summary that captures ALL key points
- Include important facts, figures, quotes, and context
- Maintain journalistic objectivity and accuracy
- The summary should be 3-5 sentences long
- Focus on Who, What, When, Where, Why, and How

ARTICLE TO SUMMARIZE:
{article_text}

COMPREHENSIVE SUMMARY:"""
        
        return prompt
    
    def generate_summary(self, article_text: str, url: str = "", max_new_tokens: int = 200) -> str:
        """Generate summary for an article"""
        try:
            if not self.model or not self.tokenizer:
                raise Exception("Model not loaded")
            
            # Create summarization prompt
            prompt = self.create_summarization_prompt(article_text, url)
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode generated text
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the summary part (after the prompt)
            summary_start = full_response.find("COMPREHENSIVE SUMMARY:") + len("COMPREHENSIVE SUMMARY:")
            summary = full_response[summary_start:].strip()
            
            # Clean up the summary
            summary = self.clean_summary(summary)
            
            return summary if summary else "Summary generation failed"
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Error: {str(e)}"
    
    def clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary"""
        # Remove any remaining prompt text
        summary = re.sub(r'^.*?COMPREHENSIVE SUMMARY:\s*', '', summary, flags=re.IGNORECASE)
        
        # Remove incomplete sentences at the end
        sentences = summary.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            sentences = sentences[:-1]
        
        # Join sentences and clean up
        summary = '. '.join(sentences).strip()
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def classify_bias(self, article_text: str) -> Dict[str, int]:
        """Simple regex-based bias classification"""
        text_lower = article_text.lower()
        
        political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign)\b',
        ]
        
        gender_patterns = [
            r'\b(men|women|male|female|masculine|feminine|gender|sex|sexist|feminist)\b',
        ]
        
        cultural_patterns = [
            r'\b(culture|cultural|ethnic|race|racial|religion|religious|immigrant|foreigner|native|traditional)\b',
        ]
        
        ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted)\b',
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
    
    def process_articles(self, limit: int = 10) -> Dict:
        """Process unprocessed articles from MongoDB"""
        if not self.collection:
            return {"error": "Not connected to MongoDB"}
        
        print(f"🔄 Processing up to {limit} articles...")
        
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
                summary = self.generate_summary(article_text, url)
                
                # Classify bias
                bias = self.classify_bias(article_text)
                
                # Calculate confidence score
                confidence = 0.8 if len(summary) > 50 and "Error:" not in summary else 0.3
                
                # Update article in database
                update_data = {
                    "summary": summary,
                    "bias": bias,
                    "confidence_score": confidence,
                    "processed_at": datetime.now().isoformat(),
                    "processed": True,
                    "model_used": self.model_name
                }
                
                result = self.collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    successful_count += 1
                    print(f"✅ Successfully processed and updated")
                    print(f"📄 Summary: {summary[:100]}...")
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
            
            return {
                "total_articles": total_count,
                "processed": processed_count,
                "unprocessed": unprocessed_count,
                "processed_percentage": f"{(processed_count/total_count*100):.1f}%" if total_count > 0 else "0%"
            }
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}

def main():
    """Main function to run article summarization"""
    print("🚀 Article Summarizer with Mistral 7B")
    print("=" * 50)
    
    # Initialize summarizer
    summarizer = ArticleSummarizer()
    
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
    
    # Load the model
    if not summarizer.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Process articles
    print(f"\n🔄 Starting article processing...")
    result = summarizer.process_articles(limit=5)  # Process 5 articles as a test
    
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
    
    print(f"\n🏁 Article summarization completed!")

if __name__ == "__main__":
    main()

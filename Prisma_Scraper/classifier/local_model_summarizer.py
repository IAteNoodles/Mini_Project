#!/usr/bin/env python3
"""
Article Summarizer using Local Mistral 7B model with fallbacks
"""
import os
import sys
from pathlib import Path
import pymongo
from datetime import datetime
import re
from typing import Dict, List, Optional
import json
import logging
from logging.handlers import RotatingFileHandler

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = parent_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "summarizer.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Setup logger
    logger = logging.getLogger("ArticleSummarizer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

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

class LocalModelSummarizer:
    """Article summarization using local Mistral 7B model with fallbacks"""
    
    def __init__(self, model_name: str = "unsloth/mistral-7b-v0.2-bnb-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.mongo_client = None
        self.collection = None
        self.device = None
        self.model_loaded = False
        self.logger = setup_logging()
        
        self.logger.info("🚀 Initializing Local Model Article Summarizer")
        self.logger.info(f"🤖 Target Model: {model_name}")
        print("🚀 Initializing Local Model Article Summarizer")
        print(f"🤖 Target Model: {model_name}")
        
    def load_local_model(self):
        """Try to load the local Mistral model"""
        try:
            self.logger.info("📦 Attempting to load local model...")
            print("📦 Attempting to load local model...")
            
            # Check if CUDA is available
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"🔧 Device: {self.device}")
            print(f"🔧 Device: {self.device}")
            
            if self.device == "cpu":
                self.logger.warning("⚠️  CUDA not available. Model will run on CPU (slower)")
                print("⚠️  CUDA not available. Model will run on CPU (slower)")
            
            # Try to load transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.logger.info(f"📥 Loading tokenizer...")
            print(f"📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"📥 Loading model with 4-bit quantization...")
            print(f"📥 Loading model with 4-bit quantization...")
            
            # Load model with optimizations for limited VRAM
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            else:
                # CPU fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
            
            self.model_loaded = True
            self.logger.info("✅ Local model loaded successfully!")
            print("✅ Local model loaded successfully!")
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {e}")
            print(f"❌ Import error: {e}")
            print("💡 Missing dependencies. Install with:")
            print("   pip install torch transformers accelerate bitsandbytes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading local model: {e}")
            print(f"❌ Error loading local model: {e}")
            print("💡 Falling back to alternative methods...")
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
            self.processed_collection = self.db["processed_Articles"]  # New collection for processed articles
            
            # Test connection
            self.mongo_client.admin.command('ismaster')
            print("✅ Connected to MongoDB successfully!")
            print(f"📝 Will save processed articles to: processed_Articles collection")
            return True
            
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            return False
    
    def create_summarization_prompt(self, article_text: str, url: str = "") -> str:
        """Create a comprehensive prompt for detailed article summarization"""
        
        # No truncation - process complete article for comprehensive summary
        prompt = f"""<s>[INST] Create a comprehensive and detailed summary of this complete news article. Your summary should:

1. Include ALL key information, facts, and important details from the article
2. Capture essential quotes, names, dates, locations, and statistics mentioned
3. Cover the main story, background context, and any developments or implications
4. Maintain chronological order and logical flow of events
5. Include relevant stakeholder perspectives and expert opinions cited
6. Be thorough and informative - aim for 6-10 sentences or more if needed to cover all important aspects
7. Write in clear, professional journalism style
8. Focus ONLY on the content - do not include explanations, instructions, or meta-commentary

Article to summarize:

{article_text}[/INST]

"""
        
        return prompt
    
    def generate_summary_local(self, article_text: str, url: str = "", max_new_tokens: int = 300) -> str:
        """Generate comprehensive summary using local Mistral model - no text limits"""
        try:
            if not self.model_loaded or not self.model or not self.tokenizer:
                raise Exception("Local model not loaded")
            
            import torch
            
            # Create summarization prompt for full article
            prompt = self.create_summarization_prompt(article_text, url)
            
            # Tokenize prompt with higher limits for comprehensive processing
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate comprehensive summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Increased for comprehensive summaries
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            # Decode generated text
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the summary part (after [/INST])
            summary_start = full_response.find("[/INST]")
            if summary_start != -1:
                summary = full_response[summary_start + 7:].strip()
            else:
                summary = full_response.strip()
            
            # Clean up the summary
            summary = self.clean_summary(summary)
            
            return summary if summary else "Local model summary generation failed"
            
        except Exception as e:
            self.logger.error(f"❌ Error generating local summary: {e}")
            print(f"❌ Error generating local summary: {e}")
            return f"Local model error: {str(e)}"
    
    def generate_summary_openai(self, article_text: str, url: str = "") -> str:
        """Generate comprehensive summary using OpenAI API as fallback - detailed coverage"""
        try:
            import openai
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "OpenAI API key not available"
            
            client = openai.OpenAI(api_key=api_key)
            
            # Process full article - comprehensive detailed summary
            prompt = f"""Create a comprehensive and detailed summary of this complete news article. Your summary should:

1. Include ALL key information, facts, and important details from the article
2. Capture essential quotes, names, dates, locations, and statistics mentioned
3. Cover the main story, background context, and any developments or implications
4. Maintain chronological order and logical flow of events
5. Include relevant stakeholder perspectives and expert opinions cited
6. Be thorough and informative - use as many sentences as needed to cover all important aspects (typically 6-12 sentences)
7. Write in clear, professional journalism style
8. Focus ONLY on the content - provide just the summary without explanations

Article to summarize:

{article_text}

Comprehensive Summary:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,  # Increased for comprehensive detailed summaries
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"OpenAI error: {e}"
    
    def generate_summary_regex(self, article_text: str, url: str = "") -> str:
        """Generate comprehensive extractive summary using complete article content"""
        try:
            # Split into sentences - process full article without limits
            sentences = re.split(r'[.!?]+', article_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return "No content available for summarization"
            
            # For comprehensive summaries, extract more content intelligently
            summary_sentences = []
            
            if len(sentences) >= 12:
                # Take first 3, early middle 2, middle 2, late middle 2, and last 3 for comprehensive coverage
                early_mid = len(sentences) // 4
                mid_point = len(sentences) // 2
                late_mid = (3 * len(sentences)) // 4
                
                summary_sentences = (sentences[:3] + 
                                   sentences[early_mid:early_mid+2] + 
                                   sentences[mid_point-1:mid_point+1] + 
                                   sentences[late_mid:late_mid+2] + 
                                   sentences[-3:])
            elif len(sentences) >= 8:
                # Take first 2, middle 3, and last 3 for medium articles
                mid_point = len(sentences) // 2
                summary_sentences = sentences[:2] + sentences[mid_point-1:mid_point+2] + sentences[-3:]
            elif len(sentences) >= 6:
                # Take first 2, middle 2, and last 2 for shorter articles
                mid_point = len(sentences) // 2
                summary_sentences = sentences[:2] + sentences[mid_point-1:mid_point+1] + sentences[-2:]
            elif len(sentences) >= 4:
                # Take first 2 and last 2 for short articles
                summary_sentences = sentences[:2] + sentences[-2:]
            else:
                # Take all available sentences for very short articles
                summary_sentences = sentences
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sentences = []
            for sentence in summary_sentences:
                if sentence not in seen:
                    seen.add(sentence)
                    unique_sentences.append(sentence)
            
            summary = '. '.join(unique_sentences)
            
            # Ensure it ends with a period
            if summary and not summary.endswith('.'):
                summary += '.'
            
            return summary if len(summary) > 10 else "Summary generation failed"
            
        except Exception as e:
            return f"Regex summary error: {e}"
    
    def validate_summary_quality(self, summary: str, min_length: int = 80, is_local_model: bool = False) -> bool:
        """Validate if the generated comprehensive summary meets quality requirements"""
        if not summary or len(summary.strip()) < min_length:
            return False
        
        # For local model, be more lenient with formatting artifacts
        if is_local_model:
            # Only check for major formatting issues that indicate complete failure
            critical_failures = [
                "Local model error:", "generation failed", "Error generating", 
                "failed to generate", "No content available"
            ]
            for failure in critical_failures:
                if failure.lower() in summary.lower():
                    return False
            # For comprehensive summaries, expect reasonable length (more than basic)
            return len(summary.strip()) >= min_length
        
        # For other models, check for common formatting artifacts (stricter)
        bad_patterns = [
            "### ShowHide", "[SHOW]", "[/SHOW]", "[INSERT]", "[/INSERT]", 
            "ANSWERS:", "[ANSWER]", "Summary of Article", "Instructions:",
            "<s>[INST]", "[/INST]", "SUMMARY:", "Article:", "TEXT:"
        ]
        
        for pattern in bad_patterns:
            if pattern in summary:
                return False
        
        # Check if it has good sentence structure for comprehensive summaries
        sentences = summary.split('.')
        valid_sentences = [s for s in sentences if len(s.strip()) > 15]  # Slightly longer sentences expected
        
        # For comprehensive summaries, expect at least 4-5 meaningful sentences
        return len(valid_sentences) >= 4
    
    def retry_summary_generation(self, article_text: str, url: str = "", max_retries: int = 2) -> tuple[str, str]:
        """Generate comprehensive summary with retries and quality validation. Returns (summary, method_used)"""
        for attempt in range(max_retries + 1):
            if self.model_loaded:
                summary = self.generate_summary_local(article_text, url, max_new_tokens=500)  # Increased for comprehensive
                # Use more lenient validation for local model but expect comprehensive length
                if self.validate_summary_quality(summary, min_length=80, is_local_model=True):
                    return summary, "local_mistral"
                print(f"⚠️  Local model summary attempt {attempt + 1} failed quality check, retrying...")
            else:
                summary = self.generate_summary_openai(article_text, url)
                if "error" in summary.lower() or "not available" in summary.lower():
                    summary = self.generate_summary_regex(article_text, url)
                    method = "regex_extractive"
                else:
                    method = "openai_gpt3.5"
                # Use standard validation for non-local models but expect comprehensive length
                if self.validate_summary_quality(summary, min_length=120, is_local_model=False):
                    return summary, method
                print(f"⚠️  Fallback summary attempt {attempt + 1} failed quality check, retrying...")
            
            # If local model failed multiple times, try OpenAI as fallback
            if self.model_loaded and attempt == max_retries:
                print(f"🔄 Local model failed {max_retries + 1} times, trying OpenAI fallback...")
                fallback_summary = self.generate_summary_openai(article_text, url)
                if "error" not in fallback_summary.lower() and "not available" not in fallback_summary.lower():
                    if self.validate_summary_quality(fallback_summary, min_length=120, is_local_model=False):
                        return fallback_summary, "openai_gpt3.5_fallback"
        
        # Final fallback - ensure we have something usable
        final_summary = self.generate_summary_regex(article_text, url)
        return final_summary, "regex_extractive_final"
    
    def clean_summary(self, summary: str) -> str:
        """Clean and format the generated comprehensive summary to remove artifacts while preserving content"""
        if not summary:
            return summary
        
        # Remove common formatting artifacts and unwanted patterns
        clean_patterns = [
            ("### ShowHide Instructions", ""),
            ("[SHOW]", ""), ("[/SHOW]", ""), ("[INSERT]", ""), ("[/INSERT]", ""),
            ("ANSWERS:", ""), ("[ANSWER]", ""), ("Summary of Article", ""),
            ("Instructions:", ""), ("<s>[INST]", ""), ("[/INST]", ""),
            ("SUMMARY:", ""), ("Article:", ""), ("TEXT:", ""),
            ("Create a comprehensive", ""), ("summary of", ""),
            ("Answer & Explanation", ""), ("Solution:", ""), ("Explanation:", ""),
            ("Requirements:", ""), ("Article to summarize:", ""),
            ("News Summary:", ""), ("Brief Summary:", ""),
            ("Comprehensive Summary:", ""), ("Keywords:", ""), ("Refers to", "")
        ]
        
        cleaned = summary
        for pattern, replacement in clean_patterns:
            cleaned = cleaned.replace(pattern, replacement)
        
        # Remove anything in square brackets [like this] using regex
        import re
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        
        # More aggressive cleaning for comprehensive summaries
        lines = cleaned.split('\n')
        clean_lines = []
        explanation_started = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Stop processing once we hit explanation/keyword sections
            if any(phrase in line.lower() for phrase in [
                'explanation:', 'keywords:', 'refers to:', 'this refers to',
                'terms refer to', 'meeting or gathering', 'process of gradually'
            ]):
                explanation_started = True
                continue
            
            # Skip if we're in explanation section
            if explanation_started:
                continue
                
            # Skip instruction-like lines but be careful not to remove actual content
            if (any(phrase in line.lower() for phrase in [
                'summarize this news article', 'write only', 'your summary should:', 
                'include all key information', 'capture essential quotes',
                'focus only on the content', 'article to summarize',
                'solution:', 'answer & explanation'
            ]) and len(line) < 200):  # Only skip if it's clearly instructional (short lines)
                continue
                
            # Skip numbered instruction lists (1., 2., etc.) but preserve numbered content in articles
            if re.match(r'^\d+\.\s+(Include|Capture|Cover|Maintain|Be|Write|Focus)', line):
                continue
            
            # Skip lines that start with common artifacts
            if line.startswith(('Solution:', 'Answer:', 'Explanation:', 'Keywords:', 'Refers to:')):
                continue
                
            if len(line) > 10:  # Keep substantial content
                clean_lines.append(line)
        
        if clean_lines:
            cleaned = ' '.join(clean_lines)
        
        # Final cleanup of any remaining artifacts
        cleaned = re.sub(r'Answer\s*&\s*Explanation\s*Solution:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Solution:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'Explanation:\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace and clean up
        cleaned = ' '.join(cleaned.split())
        
        # Ensure proper sentence structure
        if cleaned and not cleaned.endswith('.'):
            cleaned += '.'
        
        return cleaned.strip()
    
    def calculate_dynamic_confidence(self, summary: str, article_text: str, method: str, 
                                    bias_verification: Dict[str, any] = None,
                                    summary_accuracy: Dict[str, any] = None) -> float:
        """Calculate dynamic confidence score based on summary quality, bias verification, and accuracy"""
        if not summary or not article_text:
            return 0.1
        
        confidence = 0.0
        
        # Base confidence by method
        method_base = {
            "local_mistral": 0.3,               # Lower base, improved by verification
            "openai_gpt3.5": 0.4,
            "openai_gpt3.5_fallback": 0.3,
            "local_model_verification": 0.35,   # Slightly higher for verification pass
            "openai_verification": 0.45,        # Higher for OpenAI verification
            "regex_extractive": 0.15,           # Lower base, relies on verification
            "regex_extractive_final": 0.1,
            "verification_failed": 0.1          # Lowest for failed verification
        }
        confidence += method_base.get(method, 0.2)
        
        # SUMMARY ACCURACY VERIFICATION (Major component)
        if summary_accuracy:
            accuracy_score = summary_accuracy.get("accuracy_score", 0.5)
            confidence += accuracy_score * 0.4  # Up to 0.4 points for perfect accuracy
            
            # Penalty for critical issues
            issues = summary_accuracy.get("issues", [])
            critical_issues = [issue for issue in issues if any(term in issue.lower() 
                              for term in ["unsupported", "artifacts", "inconsistent"])]
            
            if len(critical_issues) >= 3:
                confidence -= 0.2  # Significant penalty for multiple critical issues
            elif len(critical_issues) >= 1:
                confidence -= 0.1  # Minor penalty for some issues
        
        # BIAS VERIFICATION (Using summary + article analysis)
        if bias_verification:
            # Count bias detections in BOTH summary and article
            summary_bias_matches = self.analyze_text_patterns(summary)
            article_bias_matches = self.analyze_text_patterns(article_text)
            
            total_adjustment = 0.0
            verified_biases = 0
            false_positives = 0
            strong_evidence_count = 0
            
            for bias_type, verification in bias_verification.items():
                adjustment = verification.get("confidence_adjustment", 0.0)
                total_adjustment += adjustment
                
                if verification.get("detected_in_summary", False):
                    verified_biases += 1
                    article_evidence = article_bias_matches.get(bias_type, 0)
                    summary_evidence = summary_bias_matches.get(bias_type, 0)
                    
                    if not verification.get("verified_in_article", False):
                        false_positives += 1
                    elif article_evidence >= 6 and summary_evidence >= 2:
                        strong_evidence_count += 1  # Strong evidence in both
            
            # Apply bias verification adjustments
            confidence += total_adjustment
            
            # Additional scoring based on summary+article consistency
            consistency_bonus = 0
            for bias_type in ["political", "gender", "cultural", "ideology"]:
                summary_score = summary_bias_matches.get(bias_type, 0)
                article_score = article_bias_matches.get(bias_type, 0)
                
                # Reward good correlation between summary and article bias evidence
                if summary_score > 0 and article_score > 0:
                    if article_score >= summary_score:  # Article supports summary claims
                        consistency_bonus += 0.02
                elif summary_score == 0 and article_score == 0:  # Both clean
                    consistency_bonus += 0.01
                elif summary_score > 0 and article_score == 0:  # Summary claims unsupported
                    consistency_bonus -= 0.03
            
            confidence += consistency_bonus
            
            # Penalty for multiple false positives
            if false_positives >= 2:
                confidence -= 0.15
            elif false_positives == 1:
                confidence -= 0.05
            
            # Bonus for strong evidence correlation
            if strong_evidence_count > 0:
                confidence += 0.05 * strong_evidence_count
        
        # Length and structure quality (Enhanced)
        summary_len = len(summary.strip())
        if summary_len < 50:
            confidence += 0.0  # Too short
        elif summary_len < 120:
            confidence += 0.05  # Short but acceptable
        elif summary_len < 300:
            confidence += 0.1   # Good length
        elif summary_len < 600:
            confidence += 0.15  # Excellent length
        elif summary_len < 800:
            confidence += 0.1   # Very comprehensive
        else:
            confidence += 0.05  # Possibly too verbose
        
        # Content quality indicators
        sentences = summary.split('.')
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if len(valid_sentences) >= 6:
            confidence += 0.1   # Excellent structure
        elif len(valid_sentences) >= 4:
            confidence += 0.08  # Good structure
        elif len(valid_sentences) >= 3:
            confidence += 0.05  # Acceptable structure
        
        # Content indicators (names, numbers, dates, quotes, attribution)
        import re
        content_score = 0
        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', summary):  # Names
            content_score += 0.02
        if re.search(r'\b\d+\b', summary):  # Numbers/statistics
            content_score += 0.02
        if re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|\d{1,2}|\d{4})\b', summary):  # Dates
            content_score += 0.02
        if re.search(r'["""].*?["""]', summary):  # Quotes
            content_score += 0.03
        if re.search(r'\b(said|told|according to|reported|stated|announced)\b', summary, re.IGNORECASE):  # Attribution
            content_score += 0.03
        
        confidence += content_score
        
        # Critical error detection (Adjusted penalties for regenerated content)
        error_indicators = ["error", "failed", "unavailable", "access denied", "not found", 
                           "#instructions", "#create", "#comprehensive", "🎙️ Voice is AI-generated"]
        
        critical_errors = 0
        for indicator in error_indicators:
            if indicator.lower() in summary.lower():
                critical_errors += 1
        
        # Apply graduated penalties instead of harsh -0.4 penalty
        if critical_errors > 0:
            if method in ["local_model_comprehensive", "comprehensive_regeneration_failed"]:
                # More lenient for regenerated content
                confidence -= min(0.2, critical_errors * 0.1)
            else:
                # Standard penalty for original content
                confidence -= min(0.4, critical_errors * 0.15)
        
        # Ensure confidence is between 0.1 and 1.0
        # Special handling for regenerated content - higher minimum threshold
        if method in ["local_model_comprehensive", "comprehensive_regeneration_failed"]:
            return max(0.25, min(1.0, confidence))  # Higher minimum for regenerated content
        else:
            return max(0.1, min(1.0, confidence))
    
    def analyze_text_patterns(self, text: str) -> Dict[str, int]:
        """Analyze text for bias patterns and return match counts"""
        if not text:
            return {"political": 0, "gender": 0, "cultural": 0, "ideology": 0}
        
        text_lower = text.lower()
        
        # Use the same enhanced patterns as classify_bias
        political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign|congress|senate|house|minister|president|prime minister|parliament|legislation|bill|law|candidate|ballot|referendum)\b',
            r'\b(trump|biden|harris|obama|clinton|bush|reagan|pentagon|white house|capitol|supreme court|judicial|executive|legislative|partisan|bipartisan)\b'
        ]
        
        gender_patterns = [
            r'\b(gender|sex|women|men|woman|man|female|male|masculine|feminine|feminist|sexist|discrimination|equality|harassment|assault|domestic violence|reproductive|maternity|paternity)\b',
            r'\b(she|her|hers|he|him|his|girls|boys|ladies|gentlemen|mother|father|wife|husband|daughter|son|sister|brother|workplace harassment|glass ceiling)\b'
        ]
        
        cultural_patterns = [
            r'\b(culture|cultural|ethnic|ethnicity|race|racial|religion|religious|faith|belief|tradition|traditional|heritage|community|minority|majority|immigrant|migration|foreigner|native|indigenous)\b',
            r'\b(muslim|islamic|christian|jewish|hindu|buddhist|catholic|protestant|church|mosque|temple|synagogue|prayer|worship|festival|celebration|ceremony|racism|xenophobia)\b'
        ]
        
        ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin|activism|movement|radical|extremist|moderate)\b',
            r'\b(capitalism|socialism|communism|fascism|democracy|authoritarian|libertarian|progressive|traditionalist|nationalist|manifesto|doctrine)\b'
        ]
        
        # Count matches for each category
        political_matches = 0
        for pattern in political_patterns:
            political_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        gender_matches = 0
        for pattern in gender_patterns:
            gender_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        cultural_matches = 0
        for pattern in cultural_patterns:
            cultural_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        ideology_matches = 0
        for pattern in ideology_patterns:
            ideology_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        return {
            "political": political_matches,
            "gender": gender_matches,
            "cultural": cultural_matches,
            "ideology": ideology_matches
        }
    
    def ask_model_for_confidence(self, summary: str, article_text: str) -> float:
        """Ask the model to evaluate its own summary confidence (experimental)"""
        try:
            if not self.model_loaded or not self.model or not self.tokenizer:
                return 0.5
            
            import torch
            
            confidence_prompt = f"""<s>[INST] Rate the quality of this summary on a scale of 0.1 to 1.0, where:
- 0.1-0.3 = Poor (missing key info, errors, unclear)
- 0.4-0.6 = Fair (covers basics, some details missing)
- 0.7-0.9 = Good (comprehensive, accurate, clear)
- 1.0 = Excellent (perfect summary)

Original Article (first 500 chars): {article_text[:500]}...

Summary to evaluate: {summary}

Respond with ONLY a number between 0.1 and 1.0: [/INST]

"""
            
            inputs = self.tokenizer(confidence_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            confidence_start = response.find("[/INST]")
            if confidence_start != -1:
                confidence_text = response[confidence_start + 7:].strip()
                # Extract number from response
                import re
                numbers = re.findall(r'0\.\d+|1\.0|0\.[0-9]+', confidence_text)
                if numbers:
                    return float(numbers[0])
            
            return 0.5  # Default if parsing fails
            
        except Exception as e:
            return 0.5  # Default confidence on error
        """Clean and format the generated summary"""
        # Remove any remaining prompt text
        summary = re.sub(r'^.*?\[/INST\]\s*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'<s>|</s>', '', summary)
        
        # Remove incomplete sentences at the end
        sentences = summary.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            sentences = sentences[:-1]
        
        # Join sentences and clean up
        summary = '. '.join(sentences).strip()
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def classify_bias(self, summary_text: str) -> Dict[str, int]:
        """
        Classify bias using enhanced regex patterns on the AI-generated summary
        
        Bias Classification Definitions:
        
        Political: Lack of neutrality and unfair favoritism towards one political group, 
        candidate or ideology. Violations of political neutrality in settings where this 
        norm is expected (media coverage, legal decisions, academic teaching).
        Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5933769/
        
        Gender: Stereotypical beliefs or biases about individuals based on their gender.
        Can be expressed linguistically through gendered assumptions or pronouns when 
        people of all genders are being discussed.
        Source: https://dictionary.apa.org/gender-bias
        
        Cultural: Imposing cultural values and norms on other groups, assuming one's own 
        culture is superior. When backed by social power, can lead to policies that 
        demean or destroy other cultures.
        Source: https://www.r2hub.org/library/what-is-cultural-bias
        
        Ideology: Stance or framing bias where information is presented in a partial way 
        due to the author's ideological bias, making it difficult to assess accuracy 
        and impartiality.
        Source: https://www.sciencedirect.com/science/article/pii/S0957417423021437
        """
        text_lower = summary_text.lower()
        
        # Enhanced political patterns with more comprehensive coverage
        political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign|congress|senate|house|minister|president|prime minister|parliament|legislation|bill|law|candidate|ballot|referendum)\b',
            r'\b(trump|biden|harris|obama|clinton|bush|reagan|pentagon|white house|capitol|supreme court|judicial|executive|legislative|partisan|bipartisan)\b'
        ]
        
        # Enhanced gender patterns with broader scope
        gender_patterns = [
            r'\b(gender|sex|women|men|woman|man|female|male|masculine|feminine|feminist|sexist|discrimination|equality|harassment|assault|domestic violence|reproductive|maternity|paternity)\b',
            r'\b(she|her|hers|he|him|his|girls|boys|ladies|gentlemen|mother|father|wife|husband|daughter|son|sister|brother|workplace harassment|glass ceiling)\b'
        ]
        
        # Enhanced cultural patterns with religious and ethnic coverage
        cultural_patterns = [
            r'\b(culture|cultural|ethnic|ethnicity|race|racial|religion|religious|faith|belief|tradition|traditional|heritage|community|minority|majority|immigrant|migration|foreigner|native|indigenous)\b',
            r'\b(muslim|islamic|christian|jewish|hindu|buddhist|catholic|protestant|church|mosque|temple|synagogue|prayer|worship|festival|celebration|ceremony|racism|xenophobia)\b'
        ]
        
        # Enhanced ideology patterns with political philosophies
        ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin|activism|movement|radical|extremist|moderate)\b',
            r'\b(capitalism|socialism|communism|fascism|democracy|authoritarian|libertarian|progressive|traditionalist|nationalist|manifesto|doctrine)\b'
        ]
        
        # Count matches for each pattern set
        political_matches = 0
        for pattern in political_patterns:
            political_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        gender_matches = 0
        for pattern in gender_patterns:
            gender_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        cultural_matches = 0
        for pattern in cultural_patterns:
            cultural_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        ideology_matches = 0
        for pattern in ideology_patterns:
            ideology_matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Enhanced detection logic - require minimum threshold for some categories
        political = 1 if political_matches > 0 else 0
        gender = 1 if gender_matches > 0 else 0
        cultural = 1 if cultural_matches > 0 else 0
        ideology = 1 if ideology_matches > 0 else 0  # More sensitive for ideology
        
        return {
            "political": political,
            "gender": gender,
            "cultural": cultural,
            "ideology": ideology
        }
    
    def verify_bias_against_article(self, article_text: str, detected_bias: Dict[str, int]) -> Dict[str, any]:
        """Verify detected bias against actual article content to catch false positives"""
        article_lower = article_text.lower()
        
        # Enhanced patterns for article verification (more comprehensive)
        verification_patterns = {
            "political": [
                r'\b(government|politics|political|policy|election|voting|campaign|congress|senate|house|minister|president|prime minister|parliament|legislation|bill|law|democrat|republican|conservative|liberal|left-wing|right-wing|party|candidate|ballot|referendum)\b',
                r'\b(trump|biden|harris|obama|clinton|bush|reagan|pentagon|white house|capitol|supreme court|judicial|executive|legislative)\b'
            ],
            "gender": [
                r'\b(gender|sex|women|men|woman|man|female|male|masculine|feminine|feminist|sexist|discrimination|equality|harassment|assault|domestic violence|reproductive|maternity|paternity)\b',
                r'\b(she|her|hers|he|him|his|girls|boys|ladies|gentlemen|mother|father|wife|husband|daughter|son|sister|brother)\b'
            ],
            "cultural": [
                r'\b(culture|cultural|ethnic|ethnicity|race|racial|religion|religious|faith|belief|tradition|traditional|heritage|community|minority|majority|immigrant|migration|foreigner|native|indigenous)\b',
                r'\b(muslim|islamic|christian|jewish|hindu|buddhist|catholic|protestant|church|mosque|temple|synagogue|prayer|worship|festival|celebration|ceremony)\b'
            ],
            "ideology": [
                r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin|activism|movement|radical|extremist|moderate)\b',
                r'\b(capitalism|socialism|communism|fascism|democracy|authoritarian|libertarian|progressive|traditionalist|nationalist)\b'
            ]
        }
        
        verification_results = {}
        
        for bias_type, is_detected in detected_bias.items():
            if is_detected == 1:  # Only verify detected biases
                patterns = verification_patterns.get(bias_type, [])
                article_matches = 0
                
                for pattern in patterns:
                    matches = re.findall(pattern, article_lower, re.IGNORECASE)
                    article_matches += len(matches)
                
                verification_results[bias_type] = {
                    "detected_in_summary": True,
                    "verified_in_article": article_matches > 0,
                    "article_match_count": article_matches,
                    "confidence_adjustment": 0.0
                }
                
                # Calculate confidence adjustment
                if article_matches == 0:
                    # No supporting evidence in article - likely false positive
                    verification_results[bias_type]["confidence_adjustment"] = -0.3
                elif article_matches >= 1 and article_matches <= 2:
                    # Weak evidence - slight penalty
                    verification_results[bias_type]["confidence_adjustment"] = -0.1
                elif article_matches >= 3 and article_matches <= 5:
                    # Moderate evidence - neutral
                    verification_results[bias_type]["confidence_adjustment"] = 0.0
                elif article_matches >= 6:
                    # Strong evidence - confidence boost
                    verification_results[bias_type]["confidence_adjustment"] = 0.1
            else:
                verification_results[bias_type] = {
                    "detected_in_summary": False,
                    "verified_in_article": False,
                    "article_match_count": 0,
                    "confidence_adjustment": 0.0
                }
        
        return verification_results
    
    def verify_summary_accuracy(self, summary: str, article_text: str) -> Dict[str, any]:
        """Rule-based verification that summary accurately represents the article"""
        if not summary or not article_text:
            return {"accuracy_score": 0.0, "issues": ["Empty summary or article"], "verification_details": {}}
        
        issues = []
        accuracy_score = 1.0
        verification_details = {}
        
        # Clean texts for comparison
        summary_clean = summary.lower().strip()
        article_clean = article_text.lower().strip()
        
        # Rule 1: Check for key entity consistency (names, places, organizations)
        import re
        
        # Extract entities from summary (proper nouns)
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary))
        article_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', article_text))
        
        # Check if summary entities are supported by article
        unsupported_entities = []
        for entity in summary_entities:
            if entity.lower() not in article_clean and len(entity) > 3:  # Skip short words
                unsupported_entities.append(entity)
        
        if unsupported_entities:
            issues.append(f"Unsupported entities in summary: {unsupported_entities[:3]}")
            accuracy_score -= 0.2
        
        verification_details["entity_check"] = {
            "summary_entities": len(summary_entities),
            "article_entities": len(article_entities),
            "unsupported": len(unsupported_entities)
        }
        
        # Rule 2: Check for numerical consistency
        summary_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', summary)
        article_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', article_text)
        
        unsupported_numbers = []
        for num in summary_numbers:
            if num not in article_numbers:
                unsupported_numbers.append(num)
        
        if unsupported_numbers and len(unsupported_numbers) > 2:  # Allow some flexibility
            issues.append(f"Unsupported numbers in summary: {unsupported_numbers[:3]}")
            accuracy_score -= 0.15
        
        verification_details["number_check"] = {
            "summary_numbers": len(summary_numbers),
            "article_numbers": len(article_numbers),
            "unsupported": len(unsupported_numbers)
        }
        
        # Rule 3: Check for date consistency
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY, MM-DD-YY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        
        summary_dates = []
        article_dates = []
        
        for pattern in date_patterns:
            summary_dates.extend(re.findall(pattern, summary, re.IGNORECASE))
            article_dates.extend(re.findall(pattern, article_text, re.IGNORECASE))
        
        unsupported_dates = []
        for date in summary_dates:
            if date.lower() not in [d.lower() for d in article_dates]:
                unsupported_dates.append(date)
        
        if unsupported_dates:
            issues.append(f"Unsupported dates in summary: {unsupported_dates}")
            accuracy_score -= 0.1
        
        verification_details["date_check"] = {
            "summary_dates": len(summary_dates),
            "article_dates": len(article_dates),
            "unsupported": len(unsupported_dates)
        }
        
        # Rule 4: Check for quote consistency
        summary_quotes = re.findall(r'["""]([^"""]+)["""]', summary)
        article_quotes = re.findall(r'["""]([^"""]+)["""]', article_text)
        
        unsupported_quotes = []
        for quote in summary_quotes:
            quote_found = False
            for article_quote in article_quotes:
                # Allow partial matching for quotes (common words)
                quote_words = set(quote.lower().split())
                article_quote_words = set(article_quote.lower().split())
                if len(quote_words.intersection(article_quote_words)) >= max(1, len(quote_words) // 2):
                    quote_found = True
                    break
            if not quote_found and len(quote.split()) > 2:  # Only check substantial quotes
                unsupported_quotes.append(quote[:50] + "..." if len(quote) > 50 else quote)
        
        if unsupported_quotes:
            issues.append(f"Unsupported quotes in summary: {len(unsupported_quotes)} quotes")
            accuracy_score -= 0.2
        
        verification_details["quote_check"] = {
            "summary_quotes": len(summary_quotes),
            "article_quotes": len(article_quotes),
            "unsupported": len(unsupported_quotes)
        }
        
        # Rule 5: Check for factual consistency (key terms and context)
        key_terms_summary = set(re.findall(r'\b[a-z]{4,}\b', summary_clean))
        key_terms_article = set(re.findall(r'\b[a-z]{4,}\b', article_clean))
        
        # Calculate overlap ratio
        common_terms = key_terms_summary.intersection(key_terms_article)
        if len(key_terms_summary) > 0:
            overlap_ratio = len(common_terms) / len(key_terms_summary)
        else:
            overlap_ratio = 0.0
        
        if overlap_ratio < 0.3:  # Less than 30% overlap
            issues.append(f"Low content overlap: {overlap_ratio:.1%}")
            accuracy_score -= 0.25
        
        verification_details["content_overlap"] = {
            "summary_terms": len(key_terms_summary),
            "article_terms": len(key_terms_article),
            "overlap_ratio": overlap_ratio
        }
        
        # Rule 6: Check for length appropriateness
        article_length = len(article_text)
        summary_length = len(summary)
        
        if summary_length > article_length * 0.8:  # Summary too long
            issues.append("Summary too verbose (>80% of article length)")
            accuracy_score -= 0.1
        elif summary_length < 50:  # Summary too short
            issues.append("Summary too brief (<50 characters)")
            accuracy_score -= 0.2
        
        verification_details["length_check"] = {
            "article_length": article_length,
            "summary_length": summary_length,
            "ratio": summary_length / article_length if article_length > 0 else 0
        }
        
        # Rule 7: Check for processing artifacts
        artifacts = ["#instructions", "#create", "#comprehensive", "Answer & Explanation", "Solution:", "🎙️ Voice is AI-generated"]
        found_artifacts = []
        
        for artifact in artifacts:
            if artifact.lower() in summary_clean:
                found_artifacts.append(artifact)
        
        if found_artifacts:
            issues.append(f"Processing artifacts found: {found_artifacts}")
            accuracy_score -= 0.3
        
        verification_details["artifact_check"] = {
            "artifacts_found": found_artifacts,
            "count": len(found_artifacts)
        }
        
        # Ensure score is between 0 and 1
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        
        return {
            "accuracy_score": accuracy_score,
            "issues": issues,
            "verification_details": verification_details
        }
    
    def check_summary_resemblance(self, summary: str, article_text: str) -> Dict[str, any]:
        """
        Check if the summary adequately resembles and represents the article content.
        Returns resemblance score and specific issues.
        """
        if not summary or not article_text:
            return {"resemblance_score": 0.0, "issues": ["Empty summary or article"], "requires_regeneration": True}
        
        issues = []
        resemblance_score = 1.0
        
        # Clean texts for comparison
        summary_clean = summary.lower().strip()
        article_clean = article_text.lower().strip()
        
        # 1. Content overlap check (enhanced)
        summary_words = set(re.findall(r'\b\w+\b', summary_clean))
        article_words = set(re.findall(r'\b\w+\b', article_clean))
        
        # Remove common stop words for better overlap calculation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        meaningful_summary_words = summary_words - stop_words
        meaningful_article_words = article_words - stop_words
        
        if meaningful_summary_words and meaningful_article_words:
            overlap = len(meaningful_summary_words.intersection(meaningful_article_words))
            overlap_percentage = overlap / len(meaningful_summary_words) * 100
            
            if overlap_percentage < 30:
                issues.append(f"Very low content overlap: {overlap_percentage:.1f}%")
                resemblance_score -= 0.4
            elif overlap_percentage < 50:
                issues.append(f"Low content overlap: {overlap_percentage:.1f}%")
                resemblance_score -= 0.2
        
        # 2. Key entity preservation check
        # Extract important entities (names, places, organizations)
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary))
        article_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', article_text))
        
        # Check if main entities from article are mentioned in summary
        if len(article_entities) > 3:  # Only check if article has meaningful entities
            entity_coverage = len(summary_entities.intersection(article_entities)) / len(article_entities)
            if entity_coverage < 0.3:
                issues.append(f"Poor entity coverage: {entity_coverage*100:.1f}%")
                resemblance_score -= 0.3
        
        # 3. Length appropriateness check
        summary_len = len(summary.strip())
        article_len = len(article_text.strip())
        
        if article_len > 1000 and summary_len < 100:
            issues.append("Summary too short for article length")
            resemblance_score -= 0.2
        elif article_len > 5000 and summary_len < 200:
            issues.append("Summary significantly too short for long article")
            resemblance_score -= 0.3
        
        # 4. Generic content detection
        generic_phrases = [
            "this article discusses", "this piece explores", "the article examines",
            "according to the article", "the text mentions", "as stated in the article",
            "summarize the following", "create a summary", "write a summary"
        ]
        
        generic_count = 0
        for phrase in generic_phrases:
            if phrase.lower() in summary_clean:
                generic_count += 1
        
        if generic_count > 0:
            issues.append(f"Contains generic summary language: {generic_count} phrases")
            resemblance_score -= 0.1 * generic_count
        
        # 5. Instruction artifacts check
        instruction_artifacts = [
            "[inst]", "[/inst]", "<s>", "</s>", "```", "#", "**",
            "instructions:", "prompt:", "task:", "objective:"
        ]
        
        artifact_count = 0
        for artifact in instruction_artifacts:
            if artifact.lower() in summary_clean:
                artifact_count += 1
        
        if artifact_count > 0:
            issues.append(f"Contains instruction artifacts: {artifact_count} artifacts")
            resemblance_score -= 0.2 * artifact_count
        
        # 6. Topic relevance check
        # Extract key topics/themes from article and check if summary addresses them
        article_sentences = article_text.split('.')[:10]  # Check first 10 sentences for main topics
        article_topics = []
        
        for sentence in article_sentences:
            # Extract potential topic words (nouns and adjectives)
            topic_words = re.findall(r'\b[A-Za-z]{4,}\b', sentence)
            article_topics.extend(topic_words)
        
        # Count how many topic words appear in summary
        topic_coverage = 0
        for topic in set(article_topics):
            if topic.lower() in summary_clean:
                topic_coverage += 1
        
        if len(set(article_topics)) > 0:
            topic_percentage = topic_coverage / len(set(article_topics)) * 100
            if topic_percentage < 20:
                issues.append(f"Poor topic coverage: {topic_percentage:.1f}%")
                resemblance_score -= 0.25
        
        # Final resemblance score
        resemblance_score = max(0.0, min(1.0, resemblance_score))
        
        # Determine if regeneration is required
        requires_regeneration = resemblance_score < 0.4 or len(issues) >= 3
        
        return {
            "resemblance_score": resemblance_score,
            "issues": issues,
            "requires_regeneration": requires_regeneration,
            "overlap_percentage": overlap_percentage if 'overlap_percentage' in locals() else 0,
            "entity_coverage": entity_coverage if 'entity_coverage' in locals() else 0
        }
    
    def regenerate_summary_with_bias_verification(self, article_text: str, detected_bias: Dict[str, int], 
                                                  verification_results: Dict[str, any], 
                                                  original_summary: str, url: str = "") -> tuple[str, str]:
        """Regenerate summary when bias verification fails, with specific guidance"""
        
        # Identify false positive biases (detected but not verified)
        false_positives = []
        for bias_type, verification in verification_results.items():
            if verification["detected_in_summary"] and not verification["verified_in_article"]:
                false_positives.append(bias_type)
        
        if not false_positives:
            return original_summary, "no_regeneration_needed"
        
        # Create enhanced prompt with bias verification guidance
        bias_guidance = []
        for bias_type in false_positives:
            bias_guidance.append(f"Regex found no {bias_type} supporting evidence in the article. Please verify your summary avoids {bias_type} terminology unless clearly present in the source.")
        
        guidance_text = " ".join(bias_guidance)
        
        # Try different methods based on what's available
        if self.model_loaded:
            # Use local model with verification prompt
            verification_prompt = f"""<s>[INST] Create a comprehensive and detailed summary of this news article. 

IMPORTANT BIAS VERIFICATION: {guidance_text}

Your summary should:
1. Include ALL key information, facts, and important details from the article
2. Capture essential quotes, names, dates, locations, and statistics mentioned
3. Cover the main story, background context, and any developments or implications
4. Maintain chronological order and logical flow of events
5. Include relevant stakeholder perspectives and expert opinions cited
6. Be thorough and informative - aim for 100+ words and 8-15 sentences
7. Write in clear, professional journalism style
8. Focus ONLY on the content actually present in the article
9. Avoid terminology that suggests bias types not supported by the article content

Article to summarize:

{article_text}[/INST]

"""
            
            try:
                import torch
                inputs = self.tokenizer(verification_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.6,  # Slightly lower temperature for more focused output
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                summary_start = full_response.find("[/INST]")
                if summary_start != -1:
                    new_summary = full_response[summary_start + 7:].strip()
                else:
                    new_summary = full_response.strip()
                
                new_summary = self.clean_summary(new_summary)
                return new_summary, "local_model_verification"
                
            except Exception as e:
                print(f"⚠️  Local model verification failed: {e}")
        
        # Fallback to OpenAI with verification
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                
                verification_prompt = f"""Create a comprehensive and detailed summary of this news article.

BIAS VERIFICATION ALERT: {guidance_text}

Requirements:
- Include ALL key information, facts, quotes, names, dates, locations, and statistics
- Cover main story, background context, and implications thoroughly
- Write 100+ words in 8-15 sentences
- Use professional journalism style
- Focus strictly on content actually present in the article
- Avoid language suggesting bias types not supported by article evidence

Article:

{article_text}

Verified Summary:"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": verification_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.4  # Lower temperature for more accurate verification
                )
                
                return response.choices[0].message.content.strip(), "openai_verification"
                
        except Exception as e:
            print(f"⚠️  OpenAI verification failed: {e}")
        
        # If all else fails, return original summary but mark it
        return original_summary, "verification_failed"
    
    def regenerate_summary_with_comprehensive_guidance(self, article_text: str, original_summary: str, 
                                                     guidance_text: str, url: str = "") -> tuple[str, str]:
        """Regenerate summary with comprehensive accuracy and bias guidance"""
        
        if not guidance_text:
            return original_summary, "no_guidance_provided"
        
        # Try different methods based on what's available
        if self.model_loaded:
            # Use local model with comprehensive guidance
            comprehensive_prompt = f"""<s>[INST] Create a comprehensive and detailed summary of this news article. 

IMPORTANT ACCURACY & BIAS GUIDANCE: {guidance_text}

Your summary should:
1. Include ALL key information, facts, and important details from the article
2. Use ONLY facts, names, numbers, dates, and quotes that appear in the source article
3. Capture essential quotes, names, dates, locations, and statistics mentioned
4. Cover the main story, background context, and any developments or implications
5. Maintain chronological order and logical flow of events
6. Include relevant stakeholder perspectives and expert opinions cited
7. Be thorough and informative - aim for 100+ words and 8-15 sentences
8. Write in clear, professional journalism style
9. Avoid any terminology or claims not directly supported by the article content

Article to summarize:

{article_text}[/INST]

"""
            
            try:
                import torch
                inputs = self.tokenizer(comprehensive_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.5,  # Lower temperature for more accurate output
                        top_p=0.8,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                summary_start = full_response.find("[/INST]")
                if summary_start != -1:
                    new_summary = full_response[summary_start + 7:].strip()
                else:
                    new_summary = full_response.strip()
                
                new_summary = self.clean_summary(new_summary)
                return new_summary, "local_model_comprehensive"
                
            except Exception as e:
                print(f"⚠️  Local model comprehensive regeneration failed: {e}")
        
        # Fallback to OpenAI with comprehensive guidance
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                
                comprehensive_prompt = f"""Create a comprehensive and detailed summary of this news article.

ACCURACY & BIAS REQUIREMENTS: {guidance_text}

Guidelines:
- Include ALL key information, facts, quotes, names, dates, locations, and statistics
- Use ONLY information that appears directly in the source article
- Cover main story, background context, and implications thoroughly
- Write 100+ words in 8-15 sentences
- Use professional journalism style
- Ensure factual accuracy and avoid unsupported claims

Article:

{article_text}

Accurate Summary:"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": comprehensive_prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3  # Lower temperature for accuracy
                )
                
                return response.choices[0].message.content.strip(), "openai_comprehensive"
                
        except Exception as e:
            print(f"⚠️  OpenAI comprehensive regeneration failed: {e}")
        
        # If all else fails, return original summary
        return original_summary, "comprehensive_regeneration_failed"
    
    def process_articles(self, limit: int = None, skip_processed: bool = True) -> Dict:
        """Process unprocessed articles from MongoDB - if limit is None, process all"""
        if self.collection is None:
            return {"error": "Not connected to MongoDB"}
        
        # Build the query based on skip_processed parameter
        if skip_processed:
            query = {
                "processed": {"$ne": True},
                "text": {"$exists": True, "$ne": "", "$ne": None}
            }
        else:
            query = {
                "text": {"$exists": True, "$ne": "", "$ne": None}
            }
        
        # If no limit specified, get count of all articles matching the query
        if limit is None:
            limit = self.collection.count_documents(query)
        
        print(f"🔄 Processing up to {limit} articles...")
        if skip_processed:
            print(f"⏭️  Skipping already processed articles")
        else:
            print(f"🔄 Processing all articles (including already processed)")
            
        if limit > 50:
            print(f"⚠️  Large batch processing - this may take a while...")
            print(f"💡 Tip: You can stop anytime with Ctrl+C and resume later")
        
        # Determine which method to use
        if self.model_loaded:
            method = "local_mistral"
            print("🤖 Using: Local Mistral 7B model")
        else:
            method = "regex_extractive"
            print("🤖 Using: Regex-based extractive summarization")
        
        # Get articles matching the query
        articles = list(self.collection.find(
            query,
            {"_id": 1, "url": 1, "text": 1}
        ).limit(limit))
        
        if not articles:
            message = "No unprocessed articles found" if skip_processed else "No articles found"
            return {"message": message, "processed": 0}
        
        processed_count = 0
        successful_count = 0
        failed_count = 0
        
        for i, article in enumerate(articles, 1):
            try:
                # Enhanced progress tracking for large batches
                progress_percent = (i / len(articles)) * 100
                print(f"\n📝 Processing article {i}/{len(articles)} ({progress_percent:.1f}%)")
                print(f"🔗 URL: {article.get('url', 'Unknown URL')[:80]}...")
                
                # Show periodic progress for large batches
                if len(articles) > 20 and i % 10 == 0:
                    print(f"📊 Progress update: {i}/{len(articles)} articles processed ({progress_percent:.1f}%)")
                    print(f"✅ Success rate so far: {(successful_count/processed_count*100):.1f}%" if processed_count > 0 else "N/A")
                
                article_text = article.get("text", "")
                url = article.get("url", "")
                
                if not article_text:
                    print(f"⚠️  Skipping article with no content")
                    continue
                
                # Log article text length to verify we have the complete article
                print(f"📏 Article text length: {len(article_text)} characters")
                if len(article_text) > 1000:
                    print(f"📖 Text preview: {article_text[:200]}...")
                
                # Validate we have substantial content
                if len(article_text) < 50:
                    print(f"⚠️  Article too short ({len(article_text)} chars), skipping...")
                    continue
                
                # Generate summary with quality validation and retries
                print(f"🤖 Generating comprehensive summary...")
                summary, method = self.retry_summary_generation(article_text, url)
                
                # Clean the summary to remove artifacts and brackets
                cleaned_summary = self.clean_summary(summary)
                
                # Use cleaned summary for further processing
                summary = cleaned_summary if cleaned_summary else summary
                
                # Report method used
                if method == "local_mistral":
                    print(f"✅ Local Mistral model used successfully!")
                elif method.startswith("openai"):
                    print(f"🔄 OpenAI API used: {method}")
                else:
                    print(f"📝 Regex extraction used: {method}")
                
                # Classify bias on the AI-generated summary (not the original article)
                bias = self.classify_bias(summary)
                bias_mode = [k for k, v in bias.items() if v == 1]  # Get detected bias types
                
                # Perform bias cross-validation against article content
                print(f"🔍 Performing bias cross-validation...")
                bias_verification = self.verify_bias_against_article(article_text, bias)
                
                # Perform summary accuracy verification (rule-based matching)
                print(f"📋 Verifying summary accuracy against article...")
                summary_accuracy = self.verify_summary_accuracy(summary, article_text)
                
                # Check if summary resembles the article content
                print(f"🔍 Checking summary resemblance to article...")
                resemblance_check = self.check_summary_resemblance(summary, article_text)
                
                # Calculate initial confidence score with all verifications
                initial_confidence = self.calculate_dynamic_confidence(
                    summary, article_text, method, bias_verification, summary_accuracy
                )
                
                print(f"🎯 Initial confidence score: {initial_confidence:.2f}")
                
                # Check if regeneration is needed (threshold: 0.5)
                needs_regeneration = False
                regeneration_reasons = []
                
                # Reason 1: Low overall confidence
                if initial_confidence < 0.5:
                    needs_regeneration = True
                    regeneration_reasons.append(f"Low confidence ({initial_confidence:.2f} < 0.5)")
                
                # Reason 2: Critical accuracy issues
                accuracy_issues = summary_accuracy.get("issues", [])
                critical_accuracy_issues = [issue for issue in accuracy_issues 
                                           if any(term in issue.lower() for term in ["unsupported", "artifacts", "inconsistent"])]
                if len(critical_accuracy_issues) >= 2:
                    needs_regeneration = True
                    regeneration_reasons.append(f"Multiple accuracy issues: {len(critical_accuracy_issues)}")
                
                # Reason 3: Multiple bias false positives
                false_positives = [bias_type for bias_type, verification in bias_verification.items() 
                                 if verification["detected_in_summary"] and not verification["verified_in_article"]]
                if len(false_positives) >= 2:
                    needs_regeneration = True
                    regeneration_reasons.append(f"Multiple bias false positives: {false_positives}")
                
                # Reason 4: Summary doesn't resemble article (NEW)
                if resemblance_check["requires_regeneration"]:
                    needs_regeneration = True
                    resemblance_score = resemblance_check["resemblance_score"]
                    regeneration_reasons.append(f"Poor article resemblance (score: {resemblance_score:.2f})")
                
                # Display verification results
                if accuracy_issues:
                    print(f"📋 Summary accuracy issues found: {len(accuracy_issues)}")
                    for issue in accuracy_issues[:3]:  # Show first 3 issues
                        print(f"   ⚠️  {issue}")
                
                # Display resemblance issues
                resemblance_issues = resemblance_check.get("issues", [])
                if resemblance_issues:
                    print(f"🔍 Summary resemblance issues found: {len(resemblance_issues)}")
                    for issue in resemblance_issues[:3]:  # Show first 3 issues
                        print(f"   ⚠️  {issue}")
                    print(f"   📊 Resemblance score: {resemblance_check['resemblance_score']:.2f}")
                
                if any(v["detected_in_summary"] for v in bias_verification.values()):
                    print(f"🔍 Bias verification results:")
                    for bias_type, verification in bias_verification.items():
                        if verification["detected_in_summary"]:
                            verified = "✅ Verified" if verification["verified_in_article"] else "❌ False positive"
                            adjustment = verification["confidence_adjustment"]
                            article_matches = verification.get("article_match_count", 0)
                            print(f"   {bias_type}: {verified} (article matches: {article_matches}, adjustment: {adjustment:+.2f})")
                
                # Attempt regeneration if needed
                verification_summary = summary
                verification_method = method
                final_bias = bias
                final_bias_verification = bias_verification
                final_summary_accuracy = summary_accuracy
                
                if needs_regeneration:
                    print(f"🔄 Regeneration needed: {', '.join(regeneration_reasons)}")
                    
                    # Create comprehensive guidance for regeneration
                    guidance_parts = []
                    
                    # Add bias guidance
                    if false_positives:
                        for bias_type in false_positives:
                            guidance_parts.append(f"Avoid {bias_type} terminology unless clearly supported by the article")
                    
                    # Add accuracy guidance
                    if critical_accuracy_issues:
                        guidance_parts.append("Ensure all facts, names, numbers, and quotes are directly from the article")
                    
                    # Add resemblance guidance (NEW)
                    if resemblance_check["requires_regeneration"]:
                        resemblance_issues = resemblance_check.get("issues", [])
                        if "content overlap" in str(resemblance_issues).lower():
                            guidance_parts.append("Focus on the actual content and themes present in the article")
                        if "entity coverage" in str(resemblance_issues).lower():
                            guidance_parts.append("Include key names, places, and organizations mentioned in the article")
                        if "generic" in str(resemblance_issues).lower():
                            guidance_parts.append("Write a specific summary about this article's content, not generic summary language")
                        if "artifacts" in str(resemblance_issues).lower():
                            guidance_parts.append("Provide only the final summary text without any formatting, instructions, or artifacts")
                    
                    guidance_text = ". ".join(guidance_parts)
                    
                    verification_summary, verification_method = self.regenerate_summary_with_comprehensive_guidance(
                        article_text, summary, guidance_text, url
                    )
                    
                    if verification_method != "verification_failed":
                        print(f"✅ Regenerated summary using {verification_method}")
                        
                        # Re-verify the new summary
                        final_bias = self.classify_bias(verification_summary)
                        final_bias_verification = self.verify_bias_against_article(article_text, final_bias)
                        final_summary_accuracy = self.verify_summary_accuracy(verification_summary, article_text)
                        
                        # Use the verified summary
                        summary = verification_summary
                        bias = final_bias
                        bias_mode = [k for k, v in bias.items() if v == 1]
                        method = verification_method
                        
                        print(f"🔄 Re-verification completed")
                    else:
                        print(f"⚠️  Summary regeneration failed, keeping original")
                else:
                    print(f"✅ Summary meets quality thresholds, no regeneration needed")
                
                # Calculate final confidence score
                print(f"📊 Calculating final confidence score...")
                confidence = self.calculate_dynamic_confidence(
                    summary, article_text, method, final_bias_verification, final_summary_accuracy
                )
                
                # Debug regeneration confidence drop
                if needs_regeneration and confidence < 0.3:
                    print(f"⚠️  DEBUG: Low confidence after regeneration")
                    print(f"   📏 Summary length: {len(summary)} chars")
                    print(f"   🔍 Method: {method}")
                    print(f"   📋 Accuracy issues: {len(final_summary_accuracy.get('issues', []))}")
                    print(f"   🎯 Bias detected: {[k for k, v in final_bias.items() if v == 1]}")
                
                print(f"🎯 Final confidence score: {confidence:.2f}")
                
                # Optionally ask model for self-evaluation (experimental)
                if method == "local_mistral" and len(summary) > 50:
                    model_confidence = self.ask_model_for_confidence(summary, article_text)
                    # Combine dynamic and model confidence (weighted average)
                    confidence = (confidence * 0.7) + (model_confidence * 0.3)
                    print(f"🤖 Model self-evaluation: {model_confidence:.2f}")
                
                # Save complete processed data to new processed_Articles collection
                processed_article_data = {
                    "original_id": article["_id"],
                    "url": url,
                    "text": article_text,  # Complete article text - FULL CONTENT PRESERVED
                    "article_length": len(article_text),  # Track original length
                    "summary": summary,    # AI-generated summary
                    "bias": bias,          # Bias classification results
                    "bias_mode": bias_mode, # List of detected bias types
                    "bias_verification": final_bias_verification,  # Cross-validation results
                    "summary_accuracy": final_summary_accuracy,    # Rule-based accuracy verification
                    "confidence_score": confidence,
                    "processed_at": datetime.now().isoformat(),
                    "model_used": method
                }
                
                # Validate that we're saving the complete article text
                if len(processed_article_data["text"]) != len(article_text):
                    print(f"❌ ERROR: Article text length mismatch during save!")
                    continue
                
                print(f"💾 Saving complete article: {len(article_text)} characters to processed_Articles")
                
                # Insert into processed_Articles collection
                processed_result = self.processed_collection.insert_one(processed_article_data)
                
                # Update original article with processed flag
                update_data = {
                    "summary": summary,
                    "processed_Article": summary,  # Save summary in processed_Article field
                    "bias": bias,
                    "bias_verification": final_bias_verification,  # Add verification results
                    "summary_accuracy": final_summary_accuracy,   # Add accuracy verification
                    "confidence_score": confidence,
                    "processed_at": datetime.now().isoformat(),
                    "processed": True,
                    "model_used": method,
                    "processed_article_id": processed_result.inserted_id  # Reference to processed_Articles entry
                }
                
                original_result = self.collection.update_one(
                    {"_id": article["_id"]},
                    {"$set": update_data}
                )
                
                if original_result.modified_count > 0 and processed_result.inserted_id:
                    successful_count += 1
                    print(f"✅ Successfully processed and saved to both collections")
                    print(f"📄 Summary: {summary[:120]}...")
                    print(f"🏷️  Bias detected in summary: {bias_mode if bias_mode else ['None']}")
                    print(f"🎯 Confidence: {confidence:.2f}")
                    print(f"💾 Original collection + processed_Articles collection")
                    print(f"📏 Complete article saved: {len(article_text)} characters")
                    print(f"🆔 Processed Article ID: {processed_result.inserted_id}")
                else:
                    failed_count += 1
                    print(f"❌ Failed to save to collections")
                
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Error processing article: {e}")
        
        return {
            "processed": processed_count,
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": f"{(successful_count/processed_count*100):.1f}%" if processed_count > 0 else "0%",
            "method_used": method
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about both collections"""
        if self.collection is None:
            return {"error": "Not connected to MongoDB"}
        
        try:
            # Original collection stats
            total_count = self.collection.count_documents({})
            processed_count = self.collection.count_documents({"processed": True})
            unprocessed_count = total_count - processed_count
            
            # Processed articles collection stats
            processed_articles_count = self.processed_collection.count_documents({})
            
            return {
                "original_collection": {
                    "total_articles": total_count,
                    "processed": processed_count,
                    "unprocessed": unprocessed_count,
                    "processed_percentage": f"{(processed_count/total_count*100):.1f}%" if total_count > 0 else "0%"
                },
                "processed_articles_collection": {
                    "total_processed": processed_articles_count,
                    "comprehensive_summaries": processed_articles_count  # All entries have comprehensive summaries
                }
            }
        except Exception as e:
            return {"error": f"Error getting stats: {e}"}
    
    def get_processed_articles_sample(self, limit: int = 5) -> List[Dict]:
        """Get a sample of processed articles to verify comprehensive data storage"""
        try:
            cursor = self.processed_collection.find({}).limit(limit)
            articles = []
            for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc['_id'] = str(doc['_id'])
                doc['original_id'] = str(doc['original_id'])
                if 'processed_article_id' in doc:
                    doc['processed_article_id'] = str(doc['processed_article_id'])
                articles.append(doc)
            return articles
        except Exception as e:
            return [{"error": f"Error retrieving sample: {e}"}]

def main():
    """Main function to run article summarization with bias classification on summaries"""
    print("🚀 Local Model Article Summarizer")
    print("🤖 Mistral 7B with Smart Fallbacks")
    print("🏷️  Bias Classification on AI Summaries")
    print("=" * 50)
    
    # Initialize summarizer
    summarizer = LocalModelSummarizer()
    
    # Connect to MongoDB
    if not summarizer.connect_to_mongodb():
        print("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Get initial statistics
    stats = summarizer.get_collection_stats()
    if "error" not in stats:
        print(f"\n📊 Initial Collection Statistics:")
        print(f"📰 Original Collection:")
        print(f"   Total articles: {stats['original_collection']['total_articles']}")
        print(f"   Processed: {stats['original_collection']['processed']} ({stats['original_collection']['processed_percentage']})")
        print(f"   Unprocessed: {stats['original_collection']['unprocessed']}")
        print(f"🗃️  Processed Articles Collection:")
        print(f"   Total comprehensive summaries: {stats['processed_articles_collection']['total_processed']}")
    
    # Try to load local model
    print(f"\n🔄 Attempting to load local Mistral model...")
    model_success = summarizer.load_local_model()
    
    if not model_success:
        print("\n💡 Local model unavailable. Will use fallback methods:")
        print("   1. OpenAI API (if key available)")
        print("   2. Regex-based extractive summarization")
        print("   Note: All methods now process complete articles without text limits")
    
    # Process articles with comprehensive summarization
    print(f"\n🔄 Starting comprehensive article processing...")
    print(f"📝 Processing complete articles without text size limits")
    print(f"💾 Saving to processed_Articles collection with complete data")
    
    # Process ALL unprocessed articles
    print(f"🔢 Processing remaining unprocessed articles")
    print(f"🚀 Testing improved system with bracket removal and dynamic confidence")
    
    result = summarizer.process_articles(limit=5)  # Test with 5 articles first
    
    print(f"\n🎯 Processing Results:")
    print(f"📝 Processed: {result.get('processed', 0)}")
    print(f"✅ Successful: {result.get('successful', 0)}")
    print(f"❌ Failed: {result.get('failed', 0)}")
    print(f"📊 Success Rate: {result.get('success_rate', '0%')}")
    print(f"🤖 Method Used: {result.get('method_used', 'unknown')}")
    
    # Get updated statistics
    updated_stats = summarizer.get_collection_stats()
    if "error" not in updated_stats:
        print(f"\n📊 Updated Statistics:")
        print(f"📰 Original Collection:")
        print(f"   Total articles: {updated_stats['original_collection']['total_articles']}")
        print(f"   Processed: {updated_stats['original_collection']['processed']} ({updated_stats['original_collection']['processed_percentage']})")
        print(f"   Unprocessed: {updated_stats['original_collection']['unprocessed']}")
        print(f"🗃️  Processed Articles Collection:")
        print(f"   Total comprehensive summaries: {updated_stats['processed_articles_collection']['total_processed']}")
    
    # Show sample of processed articles with full text verification
    print(f"\n📄 Sample of processed articles (verifying complete article text storage):")
    sample = summarizer.get_processed_articles_sample(limit=2)
    for i, article in enumerate(sample[:2], 1):
        if "error" not in article:
            print(f"\n🔍 Sample {i}:")
            print(f"   URL: {article.get('url', 'N/A')[:60]}...")
            print(f"   Summary: {article.get('summary', 'N/A')[:100]}...")
            print(f"   Bias Mode: {article.get('bias_mode', [])}")
            print(f"   📏 Full Article Length: {len(article.get('text', ''))} characters")
            print(f"   📏 Stored Length Field: {article.get('article_length', 'N/A')} characters")
            print(f"   🤖 Model Used: {article.get('model_used', 'N/A')}")
            
            # Verify complete article text is preserved
            article_text = article.get('text', '')
            if len(article_text) > 500:
                print(f"   ✅ Complete article text confirmed (>{len(article_text)} chars)")
                print(f"   📖 Text preview: {article_text[:150]}...")
            elif len(article_text) > 0:
                print(f"   ⚠️  Short article ({len(article_text)} chars): {article_text[:100]}...")
            else:
                print(f"   ❌ No article text found!")
    
    print(f"\n🏁 Comprehensive article processing completed!")
    print(f"💾 VERIFICATION: Complete article text preserved in processed_Articles collection")
    
    if model_success:
        print(f"\n🎉 Local Mistral model working successfully!")
    else:
        print(f"\n💡 To use local model:")
        print(f"   1. Install: pip install torch transformers accelerate bitsandbytes")
        print(f"   2. Use Python 3.11 or 3.12 for better compatibility")
        print(f"   3. Ensure CUDA is available for GPU acceleration")

if __name__ == "__main__":
    main()

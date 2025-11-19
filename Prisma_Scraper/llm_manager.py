"""
LLM Manager for article summarization and bias classification
"""
import re
import torch
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import instructor
from openai import OpenAI

from models import LLMResponse, BiasClassification, ArticleInput, ArticleOutput
from config import config

class LocalLLM(LLM):
    """Custom LangChain LLM wrapper for local models"""
    
    model_name: str = Field(default=config.model.model_name)
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    pipeline: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the local LLM model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config.model.load_in_8bit,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                return_full_text=False,
                max_new_tokens=config.model.max_length,
                temperature=config.model.temperature,
                do_sample=config.model.do_sample
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response using the local model"""
        try:
            response = self.pipeline(prompt)
            return response[0]['generated_text'].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    @property
    def _llm_type(self) -> str:
        return "local_llm"

class BiasClassifier:
    """Handles bias classification with fallback mechanisms"""
    
    def __init__(self):
        self.political_patterns = [
            r'\b(conservative|liberal|democrat|republican|left-wing|right-wing)\b',
            r'\b(trump|biden|election|voting|campaign)\b',
            r'\b(government|policy|politics|political)\b'
        ]
        
        self.gender_patterns = [
            r'\b(men|women|male|female|masculine|feminine)\b',
            r'\b(he|she|his|her|him|herself|himself)\b',
            r'\b(gender|sex|sexist|feminist)\b'
        ]
        
        self.cultural_patterns = [
            r'\b(culture|cultural|ethnic|race|racial|religion|religious)\b',
            r'\b(immigrant|foreigner|native|traditional|western|eastern)\b',
            r'\b(muslim|christian|jewish|hindu|buddhist)\b'
        ]
        
        self.ideology_patterns = [
            r'\b(ideology|ideological|belief|opinion|perspective|viewpoint)\b',
            r'\b(agenda|propaganda|biased|partisan|slanted)\b',
            r'\b(framing|narrative|spin|angle)\b'
        ]
    
    def classify_with_regex(self, text: str) -> BiasClassification:
        """Fallback classification using regex patterns"""
        text_lower = text.lower()
        
        political = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.political_patterns) else 0
        
        gender = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                         for pattern in self.gender_patterns) else 0
        
        cultural = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.cultural_patterns) else 0
        
        ideology = 1 if any(re.search(pattern, text_lower, re.IGNORECASE) 
                           for pattern in self.ideology_patterns) else 0
        
        return BiasClassification(
            political=political,
            gender=gender,
            cultural=cultural,
            ideology=ideology
        )

class LLMManager:
    """Main class for managing LLM operations"""
    
    def __init__(self, use_local_model: bool = True):
        """
        Initialize LLM Manager
        
        Args:
            use_local_model: Whether to use local model or OpenAI API
        """
        self.use_local_model = use_local_model
        self.llm: Optional[LLM] = None
        self.instructor_client: Optional[Any] = None
        self.bias_classifier = BiasClassifier()
        
        if use_local_model:
            self._setup_local_model()
        else:
            self._setup_openai_model()
    
    def _setup_local_model(self):
        """Setup local LLM"""
        try:
            self.llm = LocalLLM()
            logger.info("Local LLM setup completed")
        except Exception as e:
            logger.error(f"Failed to setup local LLM: {e}")
            raise
    
    def _setup_openai_model(self):
        """Setup OpenAI model with instructor"""
        try:
            client = OpenAI()  # Uses OPENAI_API_KEY env var
            self.instructor_client = instructor.patch(client)
            logger.info("OpenAI model setup completed")
        except Exception as e:
            logger.error(f"Failed to setup OpenAI model: {e}")
            raise
    
    def create_analysis_prompt(self, article: ArticleInput) -> str:
        """Create prompt for article analysis"""
        prompt = f"""
You are an expert analyst tasked with summarizing articles and detecting bias. 

ARTICLE TO ANALYZE:
URL: {article.url}
Content: {article.article}

TASK 1 - COMPREHENSIVE SUMMARY:
Provide a complete, detailed summary that covers ALL key points, facts, arguments, and details from the article. Do not miss any important information. The summary should be comprehensive and thorough.

TASK 2 - BIAS CLASSIFICATION:
Analyze the article for the following types of bias and respond with 1 (present) or 0 (not present):

1. POLITICAL BIAS: Lack of neutrality, unfair favoritism towards political groups, candidates, or ideologies. Look for partisan language, one-sided political coverage, or clear political leanings.

2. GENDER BIAS: Stereotypical beliefs about individuals based on gender, gendered language assumptions, or unequal treatment based on gender.

3. CULTURAL BIAS: Imposing cultural values on others, assuming cultural superiority, or discriminatory practices against cultural groups.

4. IDEOLOGY BIAS: Presenting information in a partial way influenced by ideological stance, making it difficult to assess accuracy and impartiality.

Respond in this exact format:
SUMMARY: [Your comprehensive summary here]
POLITICAL_BIAS: [0 or 1]
GENDER_BIAS: [0 or 1]  
CULTURAL_BIAS: [0 or 1]
IDEOLOGY_BIAS: [0 or 1]
REASONING: [Brief explanation of your classifications]
"""
        return prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_with_local_model(self, prompt: str) -> str:
        """Generate response with local model"""
        if not self.llm:
            raise RuntimeError("Local LLM not initialized")
        
        return self.llm(prompt)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _generate_with_openai(self, prompt: str) -> LLMResponse:
        """Generate structured response with OpenAI"""
        if not self.instructor_client:
            raise RuntimeError("OpenAI client not initialized")
        
        response = self.instructor_client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=LLMResponse,
            messages=[{"role": "user", "content": prompt}],
            max_retries=3
        )
        return response
    
    def parse_local_response(self, response: str) -> Optional[LLMResponse]:
        """Parse response from local model"""
        try:
            # Extract sections using regex
            summary_match = re.search(r'SUMMARY:\s*(.*?)(?=POLITICAL_BIAS:|$)', response, re.DOTALL)
            political_match = re.search(r'POLITICAL_BIAS:\s*([01])', response)
            gender_match = re.search(r'GENDER_BIAS:\s*([01])', response)
            cultural_match = re.search(r'CULTURAL_BIAS:\s*([01])', response)
            ideology_match = re.search(r'IDEOLOGY_BIAS:\s*([01])', response)
            reasoning_match = re.search(r'REASONING:\s*(.*?)$', response, re.DOTALL)
            
            if not all([summary_match, political_match, gender_match, cultural_match, ideology_match]):
                return None
            
            return LLMResponse(
                summary=summary_match.group(1).strip(),
                political_bias=int(political_match.group(1)),
                gender_bias=int(gender_match.group(1)),
                cultural_bias=int(cultural_match.group(1)),
                ideology_bias=int(ideology_match.group(1)),
                reasoning=reasoning_match.group(1).strip() if reasoning_match else ""
            )
            
        except Exception as e:
            logger.error(f"Error parsing local model response: {e}")
            return None
    
    def analyze_article(self, article: ArticleInput) -> ArticleOutput:
        """
        Analyze article for summary and bias classification
        
        Args:
            article: ArticleInput object
            
        Returns:
            ArticleOutput with analysis results
        """
        try:
            prompt = self.create_analysis_prompt(article)
            
            # Try LLM analysis first
            llm_response = None
            
            if self.use_local_model:
                try:
                    raw_response = self._generate_with_local_model(prompt)
                    llm_response = self.parse_local_response(raw_response)
                except Exception as e:
                    logger.warning(f"Local model failed: {e}")
            else:
                try:
                    llm_response = self._generate_with_openai(prompt)
                except Exception as e:
                    logger.warning(f"OpenAI model failed: {e}")
            
            # If LLM analysis succeeded, use it
            if llm_response:
                bias_classification = llm_response.to_bias_classification()
                summary = llm_response.summary
                logger.info(f"LLM analysis successful for article: {article.url}")
            else:
                # Fallback to regex classification
                logger.warning(f"LLM analysis failed, using regex fallback for: {article.url}")
                bias_classification = self.bias_classifier.classify_with_regex(article.article)
                # Generate simple summary (first 3 sentences)
                sentences = article.article.split('.')[:3]
                summary = '. '.join(sentences) + '.'
            
            return ArticleOutput(
                url=article.url,
                article=article.article,
                summary=summary,
                bias=bias_classification,
                model_version=config.model.model_name,
                confidence_score=0.8 if llm_response else 0.3
            )
            
        except Exception as e:
            logger.error(f"Error analyzing article {article.url}: {e}")
            # Return minimal result
            return ArticleOutput(
                url=article.url,
                article=article.article,
                summary="Analysis failed",
                bias=BiasClassification(political=0, gender=0, cultural=0, ideology=0),
                model_version=config.model.model_name,
                confidence_score=0.0
            )

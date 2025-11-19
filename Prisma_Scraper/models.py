"""
Data models for the LLM Article Analysis Framework
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from datetime import datetime

class BiasClassification(BaseModel):
    """Bias classification results"""
    political: int = Field(description="1 if political bias detected, 0 otherwise", ge=0, le=1)
    gender: int = Field(description="1 if gender bias detected, 0 otherwise", ge=0, le=1)
    cultural: int = Field(description="1 if cultural bias detected, 0 otherwise", ge=0, le=1)
    ideology: int = Field(description="1 if ideology bias detected, 0 otherwise", ge=0, le=1)
    
    @validator('political', 'gender', 'cultural', 'ideology')
    def validate_binary_values(cls, v):
        if v not in [0, 1]:
            raise ValueError("Bias values must be 0 or 1")
        return v

class ArticleInput(BaseModel):
    """Input article data from MongoDB"""
    id: str = Field(description="MongoDB ObjectId as string")
    url: str = Field(description="Article URL")
    article: str = Field(description="Article content")
    topic: Optional[str] = Field(description="Article topic", default=None)
    
    class Config:
        json_encoders = {
            ObjectId: str
        }

class ArticleOutput(BaseModel):
    """Output article data with analysis results"""
    url: str = Field(description="Article URL")
    article: str = Field(description="Original article content")
    summary: str = Field(description="Generated comprehensive summary")
    bias: BiasClassification = Field(description="Bias classification results")
    processed_at: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(description="Version of the model used for analysis")
    confidence_score: Optional[float] = Field(description="Overall confidence in classification", default=None)

class ProcessingResult(BaseModel):
    """Result of processing a batch of articles"""
    total_articles: int = Field(description="Total number of articles processed")
    successful: int = Field(description="Number of successfully processed articles")
    failed: int = Field(description="Number of failed articles")
    results: List[ArticleOutput] = Field(description="Processed article results")
    errors: List[Dict[str, Any]] = Field(description="Error details for failed articles")
    processing_time: float = Field(description="Total processing time in seconds")

class LLMResponse(BaseModel):
    """Structured response from LLM for bias classification"""
    summary: str = Field(description="Comprehensive article summary covering all key points and details")
    political_bias: int = Field(description="1 if political bias detected, 0 otherwise", ge=0, le=1)
    gender_bias: int = Field(description="1 if gender bias detected, 0 otherwise", ge=0, le=1)
    cultural_bias: int = Field(description="1 if cultural bias detected, 0 otherwise", ge=0, le=1)
    ideology_bias: int = Field(description="1 if ideology bias detected, 0 otherwise", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation of the classification decisions")
    
    def to_bias_classification(self) -> BiasClassification:
        """Convert to BiasClassification object"""
        return BiasClassification(
            political=self.political_bias,
            gender=self.gender_bias,
            cultural=self.cultural_bias,
            ideology=self.ideology_bias
        )

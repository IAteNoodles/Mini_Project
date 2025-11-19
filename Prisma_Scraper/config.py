"""
Configuration module for the LLM Article Analysis Framework
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class DatabaseConfig(BaseModel):
    """MongoDB database configuration"""
    connection_string: str = Field(default="mongodb://localhost:27017/")
    database_name: str = Field(default="Prisma")
    collection_name: str = Field(default="articles")
    
class ModelConfig(BaseModel):
    """LLM model configuration"""
    model_name: str = Field(default="microsoft/DialoGPT-medium")  # Lightweight for 6GB VRAM
    device: str = Field(default="cuda")
    max_length: int = Field(default=1024)
    temperature: float = Field(default=0.1)
    do_sample: bool = Field(default=True)
    load_in_8bit: bool = Field(default=True)  # For memory efficiency
    
class BiasDefinitions(BaseModel):
    """Bias classification definitions"""
    political: str = Field(
        default="Political bias refers to lack of neutrality and unfair favoritism towards one political group, candidate or ideology. It appears in media coverage, legal decisions, or academic settings where neutrality is expected."
    )
    gender: str = Field(
        default="Gender bias involves stereotypical beliefs about individuals based on their gender, expressed through language choices or assumptions about gender roles."
    )
    cultural: str = Field(
        default="Cultural bias occurs when individuals or groups impose their cultural values on others, assuming their culture is superior and potentially leading to discriminatory practices."
    )
    ideology: str = Field(
        default="Ideology bias (stance/framing bias) occurs when information is presented in a partial way, influenced by the author's ideological stance, making it difficult to assess accuracy and impartiality."
    )

class AppConfig(BaseModel):
    """Main application configuration"""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    bias_definitions: BiasDefinitions = Field(default_factory=BiasDefinitions)
    
    # Processing settings
    batch_size: int = Field(default=1)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/article_analysis.log")

# Global config instance
config = AppConfig()

# Ensure log directory exists
Path("logs").mkdir(exist_ok=True)

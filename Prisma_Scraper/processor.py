"""
Main processing pipeline for article analysis
"""
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from loguru import logger

from database import MongoDBManager
from llm_manager import LLMManager
from models import ArticleInput, ArticleOutput, ProcessingStats
from config import config

class ArticleProcessor:
    """Main processor for analyzing articles from MongoDB"""
    
    def __init__(self, use_local_model: bool = True):
        """
        Initialize the article processor
        
        Args:
            use_local_model: Whether to use local LLM or OpenAI API
        """
        self.db_manager = MongoDBManager()
        self.llm_manager = LLMManager(use_local_model=use_local_model)
        self.stats = ProcessingStats()
        
    async def process_batch(self, articles: List[ArticleInput], batch_size: int = None) -> List[ArticleOutput]:
        """
        Process a batch of articles
        
        Args:
            articles: List of ArticleInput objects
            batch_size: Size of processing batches (defaults to config value)
            
        Returns:
            List of processed ArticleOutput objects
        """
        if batch_size is None:
            batch_size = config.processing.batch_size
            
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} articles")
            
            batch_results = []
            for article in batch:
                try:
                    result = self.llm_manager.analyze_article(article)
                    batch_results.append(result)
                    self.stats.processed += 1
                    
                    if result.confidence_score > 0.5:
                        self.stats.successful += 1
                    else:
                        self.stats.failed += 1
                        
                except Exception as e:
                    logger.error(f"Error processing article {article.url}: {e}")
                    self.stats.failed += 1
                    
            results.extend(batch_results)
            
            # Optional delay between batches to prevent overload
            if config.processing.delay_between_batches > 0:
                await asyncio.sleep(config.processing.delay_between_batches)
                
        return results
    
    async def process_unprocessed_articles(
        self, 
        limit: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> ProcessingStats:
        """
        Process articles that haven't been analyzed yet
        
        Args:
            limit: Maximum number of articles to process
            filter_criteria: Additional MongoDB filter criteria
            
        Returns:
            Processing statistics
        """
        try:
            # Fetch unprocessed articles
            articles = await self.db_manager.get_unprocessed_articles(
                limit=limit,
                filter_criteria=filter_criteria
            )
            
            if not articles:
                logger.info("No unprocessed articles found")
                return self.stats
                
            logger.info(f"Found {len(articles)} unprocessed articles")
            
            # Process articles in batches
            processed_articles = await self.process_batch(articles)
            
            # Save results back to database
            if processed_articles:
                await self.db_manager.save_processed_articles(processed_articles)
                logger.info(f"Saved {len(processed_articles)} processed articles to database")
            
            # Update statistics
            self.stats.total_articles = len(articles)
            self.stats.end_time = datetime.now()
            self.stats.duration = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error in process_unprocessed_articles: {e}")
            raise
    
    async def reprocess_failed_articles(
        self,
        confidence_threshold: float = 0.3,
        limit: Optional[int] = None
    ) -> ProcessingStats:
        """
        Reprocess articles that failed or had low confidence scores
        
        Args:
            confidence_threshold: Minimum confidence score to consider for reprocessing
            limit: Maximum number of articles to reprocess
            
        Returns:
            Processing statistics
        """
        try:
            # Find articles with low confidence scores
            filter_criteria = {
                "processed": True,
                "confidence_score": {"$lt": confidence_threshold}
            }
            
            articles = await self.db_manager.get_articles(
                filter_criteria=filter_criteria,
                limit=limit
            )
            
            if not articles:
                logger.info("No failed articles found for reprocessing")
                return self.stats
                
            logger.info(f"Reprocessing {len(articles)} failed articles")
            
            # Convert to ArticleInput format
            article_inputs = [
                ArticleInput(url=article["url"], article=article["article"])
                for article in articles
            ]
            
            # Process articles
            processed_articles = await self.process_batch(article_inputs)
            
            # Update existing records
            if processed_articles:
                await self.db_manager.update_processed_articles(processed_articles)
                logger.info(f"Updated {len(processed_articles)} reprocessed articles")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error in reprocess_failed_articles: {e}")
            raise
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing statistics"""
        success_rate = (self.stats.successful / self.stats.processed * 100) if self.stats.processed > 0 else 0
        
        return {
            "total_articles": self.stats.total_articles,
            "processed": self.stats.processed,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "success_rate": f"{success_rate:.2f}%",
            "duration_seconds": self.stats.duration,
            "articles_per_second": self.stats.processed / self.stats.duration if self.stats.duration > 0 else 0,
            "start_time": self.stats.start_time.isoformat(),
            "end_time": self.stats.end_time.isoformat() if self.stats.end_time else None
        }

class AnalysisRunner:
    """High-level runner for different analysis scenarios"""
    
    @staticmethod
    async def run_full_analysis(
        use_local_model: bool = True,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete analysis on all unprocessed articles
        
        Args:
            use_local_model: Whether to use local LLM
            limit: Maximum articles to process
            batch_size: Batch size for processing
            
        Returns:
            Processing summary
        """
        processor = ArticleProcessor(use_local_model=use_local_model)
        
        if batch_size:
            config.processing.batch_size = batch_size
            
        stats = await processor.process_unprocessed_articles(limit=limit)
        
        return processor.get_processing_summary()
    
    @staticmethod
    async def run_sample_analysis(
        sample_size: int = 10,
        use_local_model: bool = True
    ) -> Dict[str, Any]:
        """
        Run analysis on a small sample for testing
        
        Args:
            sample_size: Number of articles to analyze
            use_local_model: Whether to use local LLM
            
        Returns:
            Processing summary and sample results
        """
        processor = ArticleProcessor(use_local_model=use_local_model)
        
        # Get sample articles
        db_manager = MongoDBManager()
        articles_data = await db_manager.get_unprocessed_articles(limit=sample_size)
        
        if not articles_data:
            return {"error": "No articles found for sampling"}
        
        # Convert to ArticleInput
        articles = [
            ArticleInput(url=article["url"], article=article["article"])
            for article in articles_data
        ]
        
        # Process sample
        results = await processor.process_batch(articles)
        
        # Save results
        if results:
            await db_manager.save_processed_articles(results)
        
        # Return summary with sample data
        summary = processor.get_processing_summary()
        summary["sample_results"] = [
            {
                "url": result.url,
                "summary_length": len(result.summary),
                "bias_detected": any([
                    result.bias.political,
                    result.bias.gender,
                    result.bias.cultural,
                    result.bias.ideology
                ]),
                "confidence": result.confidence_score
            }
            for result in results[:5]  # Show first 5 results
        ]
        
        return summary
    
    @staticmethod
    async def run_bias_analysis_report(
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate bias analysis report from processed articles
        
        Args:
            limit: Maximum articles to include in report
            
        Returns:
            Bias analysis statistics
        """
        db_manager = MongoDBManager()
        
        # Get processed articles
        articles = await db_manager.get_processed_articles(limit=limit)
        
        if not articles:
            return {"error": "No processed articles found"}
        
        # Calculate bias statistics
        total_articles = len(articles)
        bias_stats = {
            "political": sum(1 for a in articles if a.get("bias", {}).get("political", 0)),
            "gender": sum(1 for a in articles if a.get("bias", {}).get("gender", 0)),
            "cultural": sum(1 for a in articles if a.get("bias", {}).get("cultural", 0)),
            "ideology": sum(1 for a in articles if a.get("bias", {}).get("ideology", 0))
        }
        
        # Calculate percentages
        bias_percentages = {
            bias_type: (count / total_articles * 100) if total_articles > 0 else 0
            for bias_type, count in bias_stats.items()
        }
        
        # Get confidence distribution
        confidence_scores = [a.get("confidence_score", 0) for a in articles]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total_processed_articles": total_articles,
            "bias_counts": bias_stats,
            "bias_percentages": bias_percentages,
            "average_confidence": avg_confidence,
            "high_confidence_articles": sum(1 for score in confidence_scores if score > 0.7),
            "low_confidence_articles": sum(1 for score in confidence_scores if score < 0.3)
        }

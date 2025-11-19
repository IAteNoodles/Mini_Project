"""
Database interface for MongoDB operations
"""
from typing import List, Optional, Dict, Any, Iterator
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId
from loguru import logger

from models import ArticleInput, ArticleOutput
from config import config

class DatabaseManager:
    """Manages MongoDB database operations"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            connection_string: MongoDB connection string, uses config default if None
        """
        self.connection_string = connection_string or config.database.connection_string
        self.database_name = config.database.database_name
        self.collection_name = config.database.collection_name
        
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
        self._collection: Optional[Collection] = None
    
    def connect(self) -> None:
        """Establish database connection"""
        try:
            self._client = MongoClient(self.connection_string)
            self._database = self._client[self.database_name]
            self._collection = self._database[self.collection_name]
            
            # Test connection
            self._client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}.{self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection"""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_articles(self, 
                    batch_size: Optional[int] = None,
                    filter_query: Optional[Dict[str, Any]] = None) -> Iterator[List[ArticleInput]]:
        """
        Get articles from database in batches
        
        Args:
            batch_size: Number of articles per batch, uses config default if None
            filter_query: MongoDB filter query
            
        Yields:
            List of ArticleInput objects
        """
        if not self._collection:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        batch_size = batch_size or config.batch_size
        filter_query = filter_query or {}
        
        try:
            cursor = self._collection.find(filter_query)
            total_count = self._collection.count_documents(filter_query)
            logger.info(f"Found {total_count} articles matching filter")
            
            batch = []
            for doc in cursor:
                try:
                    article = ArticleInput(
                        id=str(doc['_id']),
                        url=doc.get('url', ''),
                        article=doc.get('article', ''),
                        topic=doc.get('topic')
                    )
                    batch.append(article)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                        
                except Exception as e:
                    logger.warning(f"Failed to parse document {doc.get('_id')}: {e}")
                    continue
            
            # Yield remaining articles
            if batch:
                yield batch
                
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            raise
    
    def get_article_by_id(self, article_id: str) -> Optional[ArticleInput]:
        """
        Get a single article by ID
        
        Args:
            article_id: MongoDB ObjectId as string
            
        Returns:
            ArticleInput object or None if not found
        """
        if not self._collection:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            doc = self._collection.find_one({'_id': ObjectId(article_id)})
            if not doc:
                return None
            
            return ArticleInput(
                id=str(doc['_id']),
                url=doc.get('url', ''),
                article=doc.get('article', ''),
                topic=doc.get('topic')
            )
            
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {e}")
            return None
    
    def save_analysis_result(self, result: ArticleOutput) -> bool:
        """
        Save analysis result to a separate collection
        
        Args:
            result: ArticleOutput object with analysis results
            
        Returns:
            True if successful, False otherwise
        """
        if not self._database:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            results_collection = self._database['analysis_results']
            doc = result.dict()
            doc['bias'] = result.bias.dict()  # Flatten bias object
            
            results_collection.insert_one(doc)
            logger.debug(f"Saved analysis result for article: {result.url}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            return False
    
    def get_unprocessed_articles(self, limit: Optional[int] = None) -> Iterator[List[ArticleInput]]:
        """
        Get articles that haven't been processed yet
        
        Args:
            limit: Maximum number of articles to return
            
        Yields:
            List of ArticleInput objects
        """
        if not self._database:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            # Get processed article URLs
            results_collection = self._database['analysis_results']
            processed_urls = set(
                doc['url'] for doc in results_collection.find({}, {'url': 1})
            )
            
            logger.info(f"Found {len(processed_urls)} already processed articles")
            
            # Filter out processed articles
            filter_query = {'url': {'$nin': list(processed_urls)}}
            if limit:
                filter_query['$limit'] = limit
            
            yield from self.get_articles(filter_query=filter_query)
            
        except Exception as e:
            logger.error(f"Error getting unprocessed articles: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with collection statistics
        """
        if not self._database:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            total_articles = self._collection.count_documents({})
            
            results_collection = self._database['analysis_results']
            processed_articles = results_collection.count_documents({})
            
            return {
                'total_articles': total_articles,
                'processed_articles': processed_articles,
                'pending_articles': total_articles - processed_articles,
                'processing_rate': (processed_articles / total_articles * 100) if total_articles > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

"""Vector storage implementation using SQLite with sqlite-vss extension."""

import logging
import numpy as np
from typing import List, Optional, Dict, Any

from src.core.database import get_database
from src.storage.models import (
    PromptModel, PromptEmbeddingModel,
    ReportModel, ReportEmbeddingModel
)

logger = logging.getLogger(__name__)

class VectorStorage:
    """Handles vector storage operations."""
    
    def __init__(self):
        """Initialize vector storage."""
        self.db = get_database()
        self._setup_vector_search()

    def _setup_vector_search(self):
        """Set up vector search capabilities."""
        try:
            with self.db.atomic():
                # Enable SQLite extensions
                self.db.execute_sql('PRAGMA foreign_keys = ON')
                self.db.execute_sql('SELECT load_extension("sqlite_vss")')
                
                # Create virtual tables for vector search if they don't exist
                self.db.execute_sql(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS prompt_embedding_index USING vss0(embedding)"
                )
                self.db.execute_sql(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS report_embedding_index USING vss0(embedding)"
                )
                logger.info("Vector search setup completed")
        except Exception as e:
            logger.error(f"Error setting up vector search: {e}")
            raise

    def insert_prompt(self, prompt_text: str, embedding: List[float],
                     context: str, prompt_type_name: str = 'other',
                     model_name: Optional[str] = None,
                     quality_score: Optional[float] = None) -> int:
        """Insert a prompt and its embedding.
        
        Args:
            prompt_text: The text of the prompt
            embedding: Vector embedding of the prompt
            context: Context information (e.g., application, window)
            prompt_type_name: Type of prompt (default: 'other')
            model_name: Name of the LLM model used
            quality_score: User feedback score
            
        Returns:
            int: ID of the inserted prompt
        """
        try:
            with self.db.atomic():
                # Create prompt
                prompt = PromptModel.create(
                    prompt_text=prompt_text,
                    prompt_type=prompt_type_name,
                    model_name=model_name,
                    context=context,
                    quality_score=quality_score
                )
                
                # Store embedding
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                PromptEmbeddingModel.create(
                    prompt=prompt,
                    embedding=embedding_blob
                )
                
                # Add to vector index
                self.db.execute_sql(
                    "INSERT INTO prompt_embedding_index (rowid, embedding) VALUES (?, ?)",
                    (prompt.prompt_id, embedding_blob)
                )
                
                return prompt.prompt_id
                
        except Exception as e:
            logger.error(f"Error inserting prompt: {e}")
            raise

    def search_similar_prompts(self, query_embedding: List[float],
                             limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar prompts using vector similarity.
        
        Args:
            query_embedding: Vector embedding to search for
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar prompts with metadata
        """
        try:
            query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
            
            cursor = self.db.execute_sql("""
                SELECT 
                    p.prompt_id,
                    p.prompt_text,
                    p.prompt_type_name,
                    p.model_name,
                    p.context,
                    p.timestamp,
                    p.quality_score,
                    vss_distance(pe.embedding, ?) as distance
                FROM promptmodel p
                JOIN promptembeddingmodel pe ON p.prompt_id = pe.prompt_id
                JOIN prompt_embedding_index pei ON p.prompt_id = pei.rowid
                ORDER BY vss_distance(pe.embedding, ?) ASC
                LIMIT ?
            """, (query_blob, query_blob, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'prompt_id': row[0],
                    'prompt_text': row[1],
                    'prompt_type': row[2],
                    'model_name': row[3],
                    'context': row[4],
                    'timestamp': row[5],
                    'quality_score': row[6],
                    'distance': row[7]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar prompts: {e}")
            return []

    def insert_report(self, report_text: str, embedding: List[float],
                     interval_type_name: str, timestamp: str,
                     period_end: str) -> int:
        """Insert a report and its embedding.
        
        Args:
            report_text: The content of the report
            embedding: Vector embedding of the report
            interval_type_name: Type of interval (hourly, daily, etc.)
            timestamp: Start time of the report interval
            period_end: End time of the report interval
            
        Returns:
            int: ID of the inserted report
        """
        try:
            with self.db.atomic():
                # Create report
                report = ReportModel.create(
                    report_text=report_text,
                    interval_type=interval_type_name,
                    timestamp=timestamp,
                    period_end=period_end
                )
                
                # Store embedding
                embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                ReportEmbeddingModel.create(
                    report=report,
                    embedding=embedding_blob
                )
                
                # Add to vector index
                self.db.execute_sql(
                    "INSERT INTO report_embedding_index (rowid, embedding) VALUES (?, ?)",
                    (report.report_id, embedding_blob)
                )
                
                return report.report_id
                
        except Exception as e:
            logger.error(f"Error inserting report: {e}")
            raise

    def search_similar_reports(self, query_embedding: List[float],
                             limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar reports using vector similarity.
        
        Args:
            query_embedding: Vector embedding to search for
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar reports with metadata
        """
        try:
            query_blob = np.array(query_embedding, dtype=np.float32).tobytes()
            
            cursor = self.db.execute_sql("""
                SELECT 
                    r.report_id,
                    r.report_text,
                    r.interval_type_name,
                    r.timestamp,
                    r.period_end,
                    vss_distance(re.embedding, ?) as distance
                FROM reportmodel r
                JOIN reportembeddingmodel re ON r.report_id = re.report_id
                JOIN report_embedding_index rei ON r.report_id = rei.rowid
                ORDER BY vss_distance(re.embedding, ?) ASC
                LIMIT ?
            """, (query_blob, query_blob, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'report_id': row[0],
                    'report_text': row[1],
                    'interval_type': row[2],
                    'timestamp': row[3],
                    'period_end': row[4],
                    'distance': row[5]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar reports: {e}")
            return [] 
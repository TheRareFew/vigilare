"""Core database functionality."""

import logging
import os
from contextlib import contextmanager
from typing import Optional
from pathlib import Path

from peewee import SqliteDatabase, DatabaseProxy
from aw_core.dirs import get_data_dir

logger = logging.getLogger(__name__)

# Global database proxy
database_proxy = DatabaseProxy()

def get_aw_db_path(testing: bool = False) -> str:
    """Get the ActivityWatch database path.
    
    Args:
        testing: Whether to use testing database
        
    Returns:
        str: Path to the ActivityWatch database
    """
    data_dir = get_data_dir("aw-server")
    filename = f"peewee-sqlite{'-testing' if testing else ''}.v2.db"
    return os.path.join(data_dir, filename)

class DatabaseManager:
    """Manages database connections and initialization."""
    
    def __init__(self, testing: bool = False):
        """Initialize database manager.
        
        Args:
            testing: Whether to use testing database
        """
        self.db_path = get_aw_db_path(testing)
        self.database = None
        logger.info(f"Initialized database manager with path: {self.db_path}")
    
    def initialize(self) -> bool:
        """Initialize the database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Create database connection
            self.database = SqliteDatabase(
                self.db_path,
                pragmas={
                    'journal_mode': 'wal',      # Write-Ahead Logging
                    'cache_size': -1024 * 32,   # 32MB cache
                    'foreign_keys': 1,          # Enable foreign key support
                    'ignore_check_constraints': 0,
                    'synchronous': 0            # Disable synchronous writes
                }
            )
            
            # Initialize the proxy
            database_proxy.initialize(self.database)
            
            # Create tables
            self._create_tables()
            
            logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def _create_tables(self):
        """Create database tables."""
        from src.storage.models import (
            IntervalTypeModel, PromptTypeModel,
            ScreenshotModel, PromptModel,
            PromptEmbeddingModel, ReportModel,
            ReportEmbeddingModel, AppClassificationModel
        )
        
        models = [
            IntervalTypeModel,
            PromptTypeModel,
            ScreenshotModel,
            PromptModel,
            PromptEmbeddingModel,
            ReportModel,
            ReportEmbeddingModel,
            AppClassificationModel
        ]
        
        with self.database:
            self.database.create_tables(models)
    
    @contextmanager
    def get_connection(self):
        """Get a database connection.
        
        Yields:
            SqliteDatabase: Database connection
        """
        if not self.database:
            raise Exception("Database not initialized")
        
        try:
            yield self.database
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.database:
            self.database.close()
            logger.info("Database connection closed")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

def init_database(testing: bool = False) -> bool:
    """Initialize the database.
    
    Args:
        testing: Whether to use testing database
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _db_manager
    
    try:
        _db_manager = DatabaseManager(testing)
        return _db_manager.initialize()
    except Exception as e:
        logger.error(f"Error in init_database: {e}")
        return False

def get_database() -> SqliteDatabase:
    """Get the database instance.
    
    Returns:
        SqliteDatabase: Database instance
        
    Raises:
        Exception: If database is not initialized
    """
    if not _db_manager or not _db_manager.database:
        raise Exception("Database not initialized. Call init_database first.")
    return _db_manager.database

def close_database():
    """Close the database connection."""
    global _db_manager
    
    if _db_manager:
        _db_manager.close()
        _db_manager = None 
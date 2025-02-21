"""Window tracking functionality."""

import logging
import re
from typing import Dict, Optional

from src.core.aw_client import ActivityWatchClient
from src.storage.models import AppClassificationModel

logger = logging.getLogger(__name__)

class WindowTracker:
    """Tracks and classifies windows/applications."""
    
    def __init__(self, aw_client: ActivityWatchClient):
        """Initialize window tracker.
        
        Args:
            aw_client: ActivityWatch client instance
        """
        self.aw_client = aw_client
        logger.info("Initialized window tracker")

    def get_app_classification(self, app_name: str, window_title: str) -> Optional[Dict[str, str]]:
        """Get classification for an application/window combination.
        
        Args:
            app_name: Name of the application
            window_title: Window title
            
        Returns:
            Optional[Dict[str, str]]: Classification info or None if not found
        """
        try:
            # Query all classifications for this app
            classifications = (AppClassificationModel
                .select()
                .where(AppClassificationModel.application_name == app_name)
                .order_by(AppClassificationModel.confidence.desc()))
            
            # Check each classification's regex pattern
            for classification in classifications:
                if re.search(classification.window_title_regex, window_title):
                    return {
                        'category': classification.category,
                        'confidence': float(classification.confidence),
                        'manual_override': classification.manual_override
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting app classification: {e}")
            return None

    def classify_window(self, window_info: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Classify a window based on its information.
        
        Args:
            window_info: Window information (app name and title)
            
        Returns:
            Optional[Dict[str, str]]: Classification info or None if not found
        """
        try:
            app_name = window_info.get('app', '')
            window_title = window_info.get('title', '')
            
            if not app_name or not window_title:
                return None
            
            classification = self.get_app_classification(app_name, window_title)
            
            if classification:
                return {
                    **window_info,
                    **classification
                }
            
            return window_info
            
        except Exception as e:
            logger.error(f"Error classifying window: {e}")
            return None

    def add_classification(self, app_name: str, title_pattern: str,
                         category: str, confidence: float = 1.0,
                         manual_override: bool = True) -> bool:
        """Add a new application classification.
        
        Args:
            app_name: Name of the application
            title_pattern: Regex pattern for window titles
            category: Category to assign
            confidence: Confidence score (0-1)
            manual_override: Whether this is a manual classification
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate regex pattern
            re.compile(title_pattern)
            
            classification = AppClassificationModel.create(
                application_name=app_name,
                window_title_regex=title_pattern,
                category=category,
                confidence=confidence,
                manual_override=manual_override
            )
            
            logger.info(f"Added classification for {app_name}: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding classification: {e}")
            return False

    def get_current_window_classification(self) -> Optional[Dict[str, str]]:
        """Get classification for the currently active window.
        
        Returns:
            Optional[Dict[str, str]]: Classification info or None if not found
        """
        try:
            window_info = self.aw_client.get_current_window()
            if window_info:
                return self.classify_window(window_info)
            return None
            
        except Exception as e:
            logger.error(f"Error getting current window classification: {e}")
            return None 
"""Database operations for Vigilare.

This module contains specific database operations for managing screenshots,
prompts, reports, and application classifications. It uses the core database
connection provided by core.database.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json

from peewee import fn, JOIN

from src.core.database import get_database
from src.storage.models import (
    ScreenshotModel, PromptModel, PromptTypeModel,
    ReportModel, AppClassificationModel
)

logger = logging.getLogger(__name__)

def serialize_datetime(obj):
    """Helper function to serialize datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

class DatabaseOperations:
    """Handles database CRUD operations and queries for Vigilare data models."""
    
    def __init__(self):
        """Initialize database operations."""
        self.db = get_database()
        logger.info("Initialized database operations")

    def _serialize_context(self, context_str: str) -> Dict[str, Any]:
        """Helper to safely parse context JSON."""
        try:
            if isinstance(context_str, str):
                return json.loads(context_str)
            return context_str
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse context JSON: {context_str}")
            return {"error": "Invalid context format"}
        except Exception as e:
            logger.error(f"Error parsing context: {e}")
            return {"error": "Context parsing error"}

    def get_screenshot(self, screenshot_id: int) -> Optional[Dict[str, Any]]:
        """Get screenshot by ID.
        
        Args:
            screenshot_id: ID of the screenshot
            
        Returns:
            Optional[Dict[str, Any]]: Screenshot data or None if not found
        """
        try:
            screenshot = ScreenshotModel.get_by_id(screenshot_id)
            return {
                'image_id': screenshot.image_id,
                'image_path': screenshot.image_path,
                'ocr_text': screenshot.ocr_text,
                'image_summary': screenshot.image_summary,
                'context': self._serialize_context(screenshot.context),
                'code_insights': json.loads(screenshot.code_insights) if screenshot.code_insights else None,
                'timestamp': screenshot.timestamp.isoformat()
            }
        except ScreenshotModel.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error getting screenshot: {e}")
            return None

    def get_screenshots_in_range(self, start_time: datetime,
                               end_time: datetime) -> List[Dict[str, Any]]:
        """Get screenshots within a time range.
        
        Args:
            start_time: Start of the range
            end_time: End of the range
            
        Returns:
            List[Dict[str, Any]]: List of screenshots
        """
        try:
            screenshots = (ScreenshotModel
                .select()
                .where(
                    (ScreenshotModel.timestamp >= start_time) &
                    (ScreenshotModel.timestamp <= end_time)
                )
                .order_by(ScreenshotModel.timestamp))
            
            return [{
                'image_id': s.image_id,
                'image_path': s.image_path,
                'ocr_text': s.ocr_text,
                'image_summary': s.image_summary,
                'context': self._serialize_context(s.context),
                'code_insights': json.loads(s.code_insights) if s.code_insights else None,
                'timestamp': s.timestamp.isoformat()
            } for s in screenshots]
        except Exception as e:
            logger.error(f"Error getting screenshots in range: {e}")
            return []

    def get_prompt(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """Get prompt by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Optional[Dict[str, Any]]: Prompt data or None if not found
        """
        try:
            prompt = (PromptModel
                .select(PromptModel, PromptTypeModel)
                .join(PromptTypeModel)
                .where(PromptModel.prompt_id == prompt_id)
                .get())
            
            return {
                'prompt_id': prompt.prompt_id,
                'prompt_text': prompt.prompt_text,
                'prompt_type': prompt.prompt_type.prompt_type_name,
                'model_name': prompt.model_name,
                'llm_tool_used': prompt.llm_tool_used,
                'timestamp': prompt.timestamp.isoformat(),
                'quality_score': float(prompt.quality_score) if prompt.quality_score else None,
                'confidence': float(prompt.confidence) if prompt.confidence else None,
                'context': self._serialize_context(prompt.context)
            }
        except PromptModel.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            return None

    def get_prompts_by_type(self, prompt_type: str) -> List[Dict[str, Any]]:
        """Get prompts by type.
        
        Args:
            prompt_type: Type of prompts to get
            
        Returns:
            List[Dict[str, Any]]: List of prompts
        """
        try:
            prompts = (PromptModel
                .select(PromptModel, PromptTypeModel)
                .join(PromptTypeModel)
                .where(PromptTypeModel.prompt_type_name == prompt_type)
                .order_by(PromptModel.timestamp.desc()))
            
            return [{
                'prompt_id': p.prompt_id,
                'prompt_text': p.prompt_text,
                'prompt_type': p.prompt_type.prompt_type_name,
                'model_name': p.model_name,
                'llm_tool_used': p.llm_tool_used,
                'timestamp': p.timestamp.isoformat(),
                'quality_score': float(p.quality_score) if p.quality_score else None,
                'confidence': float(p.confidence) if p.confidence else None,
                'context': self._serialize_context(p.context)
            } for p in prompts]
        except Exception as e:
            logger.error(f"Error getting prompts by type: {e}")
            return []

    def get_report(self, report_id: int) -> Optional[Dict[str, Any]]:
        """Get report by ID.
        
        Args:
            report_id: ID of the report
            
        Returns:
            Optional[Dict[str, Any]]: Report data or None if not found
        """
        try:
            report = ReportModel.get_by_id(report_id)
            return {
                'report_id': report.report_id,
                'report_text': report.report_text,
                'interval_type': report.interval_type.interval_name,
                'timestamp': report.timestamp.isoformat(),
                'period_end': report.period_end.isoformat()
            }
        except ReportModel.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error getting report: {e}")
            return None

    def get_app_classifications(self) -> List[Dict[str, Any]]:
        """Get all application classifications.
        
        Returns:
            List[Dict[str, Any]]: List of classifications
        """
        try:
            classifications = (AppClassificationModel
                .select()
                .order_by(
                    AppClassificationModel.application_name,
                    AppClassificationModel.confidence.desc()
                ))
            
            return [{
                'app_classification_id': c.app_classification_id,
                'application_name': c.application_name,
                'window_title_regex': c.window_title_regex,
                'category': c.category,
                'confidence': float(c.confidence),
                'manual_override': c.manual_override
            } for c in classifications]
        except Exception as e:
            logger.error(f"Error getting app classifications: {e}")
            return []

    def delete_old_data(self, days_to_keep: int = 30) -> bool:
        """Delete data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.db.atomic():
                # Delete old screenshots
                ScreenshotModel.delete().where(
                    ScreenshotModel.timestamp < cutoff_date
                ).execute()
                
                # Delete old prompts
                PromptModel.delete().where(
                    PromptModel.timestamp < cutoff_date
                ).execute()
                
                # Delete old reports
                ReportModel.delete().where(
                    ReportModel.timestamp < cutoff_date
                ).execute()
            
            logger.info(f"Deleted data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            stats = {
                'screenshots': ScreenshotModel.select().count(),
                'prompts': PromptModel.select().count(),
                'reports': ReportModel.select().count(),
                'app_classifications': AppClassificationModel.select().count(),
                'prompt_types': PromptTypeModel.select().count(),
                'latest_screenshot': None,
                'latest_prompt': None,
                'latest_report': None
            }
            
            # Get latest entries
            latest_screenshot = (ScreenshotModel
                .select()
                .order_by(ScreenshotModel.timestamp.desc())
                .first())
            if latest_screenshot:
                stats['latest_screenshot'] = latest_screenshot.timestamp.isoformat()
            
            latest_prompt = (PromptModel
                .select()
                .order_by(PromptModel.timestamp.desc())
                .first())
            if latest_prompt:
                stats['latest_prompt'] = latest_prompt.timestamp.isoformat()
            
            latest_report = (ReportModel
                .select()
                .order_by(ReportModel.timestamp.desc())
                .first())
            if latest_report:
                stats['latest_report'] = latest_report.timestamp.isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'error': str(e)
            } 
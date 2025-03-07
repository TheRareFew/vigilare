"""Database operations for Vigilare.

This module contains specific database operations for managing screenshots,
prompts, reports, and application classifications. It uses the core database
connection provided by core.database.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
import os
import traceback

from peewee import fn, JOIN

from src.core.database import get_database
from src.storage.models import (
    ScreenshotModel, PromptModel, PromptTypeModel,
    ReportModel, AppClassificationModel,
    IntervalTypeModel, CursorProjectModel, CursorChatModel
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
                'image_path': os.path.abspath(screenshot.image_path),
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
                'image_path': os.path.abspath(s.image_path),
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
            screenshot_count = ScreenshotModel.select().count()
            prompt_count = PromptModel.select().count()
            report_count = ReportModel.select().count()
            app_classification_count = AppClassificationModel.select().count()
            cursor_project_count = CursorProjectModel.select().count()
            cursor_chat_count = CursorChatModel.select().count()
            
            return {
                'screenshot_count': screenshot_count,
                'prompt_count': prompt_count,
                'report_count': report_count,
                'app_classification_count': app_classification_count,
                'cursor_project_count': cursor_project_count,
                'cursor_chat_count': cursor_chat_count,
                'database_size_bytes': os.path.getsize(database.db_path) if hasattr(database, 'db_path') else 0
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'error': str(e)
            }

    def update_cursor_project(self, project_name: str, project_path: str = None) -> bool:
        """Update or create a Cursor project entry.
        
        Args:
            project_name: Name of the Cursor project
            project_path: Path to the Cursor project data (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Updating Cursor project: '{project_name}'")
            
            # Set all projects as inactive first
            inactive_count = CursorProjectModel.update(is_active=False).execute()
            logger.debug(f"Set {inactive_count} projects to inactive")
            
            # If no project path is provided, try to find it
            cursor_data_path = project_path
            if not cursor_data_path:
                logger.debug("No cursor data path provided, attempting to determine from workspace storage")
                # Use the workspace storage function to find the most recent workspace
                from src.utils.helpers import get_most_recent_workspace_dir
                cursor_data_path = get_most_recent_workspace_dir()
                logger.debug(f"Determined cursor data path from workspace storage: '{cursor_data_path}'")
            
            # Extract the actual project path from the Cursor database
            actual_project_path = None
            if cursor_data_path:
                from src.utils.helpers import extract_project_path_from_cursor_db
                actual_project_path = extract_project_path_from_cursor_db(cursor_data_path)
                logger.debug(f"Extracted actual project path: '{actual_project_path}'")
            
            # Try to get existing project
            try:
                logger.debug(f"Checking if project '{project_name}' already exists")
                project = CursorProjectModel.get(CursorProjectModel.project_name == project_name)
                logger.debug(f"Found existing project: {project_name}")
                
                # Update project
                project.is_active = True
                project.last_accessed = datetime.now()
                
                # Update cursor data path if we have one
                if cursor_data_path:
                    logger.debug(f"Updating cursor data path to: '{cursor_data_path}'")
                    project.cursor_data_path = cursor_data_path
                
                # Update actual project path if we have one
                if actual_project_path:
                    logger.debug(f"Updating actual project path to: '{actual_project_path}'")
                    project.project_path = actual_project_path
                    
                project.save()
                logger.info(f"Updated Cursor project: '{project_name}' (cursor data path: '{cursor_data_path}', project path: '{actual_project_path}')")
                
            except CursorProjectModel.DoesNotExist:
                # Create new project
                logger.debug(f"Project '{project_name}' does not exist, creating new entry")
                CursorProjectModel.create(
                    project_name=project_name,
                    project_path=actual_project_path or "",
                    cursor_data_path=cursor_data_path or "",
                    is_active=True,
                    last_accessed=datetime.now()
                )
                logger.info(f"Created new Cursor project: '{project_name}' (cursor data path: '{cursor_data_path}', project path: '{actual_project_path}')")
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating Cursor project: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
            
    def get_active_cursor_project(self) -> Optional[Dict[str, Any]]:
        """Get the active Cursor project.
        
        Returns:
            Optional[Dict[str, Any]]: Active project data or None if not found
        """
        try:
            project = CursorProjectModel.get(CursorProjectModel.is_active == True)
            return {
                'project_id': project.project_id,
                'project_name': project.project_name,
                'project_path': project.project_path,
                'cursor_data_path': project.cursor_data_path,
                'last_accessed': project.last_accessed.isoformat()
            }
        except CursorProjectModel.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"Error getting active Cursor project: {e}")
            return None
            
    def get_all_cursor_projects(self) -> List[Dict[str, Any]]:
        """Get all cursor projects.
        
        Returns:
            List[Dict[str, Any]]: List of cursor projects
        """
        try:
            projects = list(CursorProjectModel.select().dicts())
            return projects
        except Exception as e:
            logger.error(f"Error getting cursor projects: {e}")
            return []

    def add_cursor_chat(self, 
                       cursor_project_id: Optional[int], 
                       prompt_key: str, 
                       prompt_id: str, 
                       prompt_text: str, 
                       response_text: Optional[str] = None, 
                       files: Optional[Dict[str, Any]] = None, 
                       timestamp: Optional[datetime] = None, 
                       model_name: Optional[str] = None) -> Optional[int]:
        """Add a cursor chat to the database.
        
        Args:
            cursor_project_id: ID of the cursor project (optional)
            prompt_key: Key from the Cursor database (e.g., 'aiService.prompts')
            prompt_id: Unique identifier for the prompt within Cursor
            prompt_text: The actual prompt text
            response_text: The response text (optional)
            files: Associated files (optional)
            timestamp: When the prompt was created (optional)
            model_name: The model used for the response (optional)
            
        Returns:
            Optional[int]: ID of the created chat or None if failed
        """
        try:
            # Validate inputs
            if not prompt_key or not prompt_id:
                logger.error("Missing required fields: prompt_key or prompt_id")
                return None
                
            # Log the inputs
            logger.debug(f"Adding cursor chat - prompt_key: {prompt_key}, prompt_id: {prompt_id}")
            logger.debug(f"Prompt text (first 50 chars): {prompt_text[:50] if prompt_text else 'None'}")
            
            # Check if this chat already exists
            try:
                existing_chat = CursorChatModel.select().where(
                    (CursorChatModel.prompt_key == prompt_key) & 
                    (CursorChatModel.prompt_id == prompt_id)
                ).first()
                
                if existing_chat:
                    logger.debug(f"Cursor chat already exists: {prompt_key} - {prompt_id}")
                    return existing_chat.chat_id
            except Exception as e:
                logger.error(f"Error checking for existing chat: {e}")
                # Continue with the insertion attempt
            
            # Prepare files as JSON string if provided
            files_json = None
            if files:
                try:
                    files_json = json.dumps(files)
                except Exception as e:
                    logger.error(f"Error serializing files to JSON: {e}")
            
            # Use current time if timestamp not provided
            if timestamp is None:
                timestamp = datetime.now()
                
            # Truncate prompt_text and response_text if they are too long
            # SQLite has a limit of 1 billion bytes per cell, but we'll be more conservative
            max_text_length = 100000  # 100K characters should be plenty
            
            if prompt_text and len(prompt_text) > max_text_length:
                logger.warning(f"Truncating prompt_text from {len(prompt_text)} to {max_text_length} characters")
                prompt_text = prompt_text[:max_text_length] + "... [truncated]"
                
            if response_text and len(response_text) > max_text_length:
                logger.warning(f"Truncating response_text from {len(response_text)} to {max_text_length} characters")
                response_text = response_text[:max_text_length] + "... [truncated]"
                
            # Create new chat
            with self.db.atomic():
                chat = CursorChatModel.create(
                    cursor_project=cursor_project_id,
                    prompt_key=prompt_key,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text or "",
                    response_text=response_text,
                    files=files_json,
                    timestamp=timestamp,
                    model_name=model_name,
                    processed_at=datetime.now()
                )
                
                logger.info(f"Added cursor chat: {prompt_key} - {prompt_id} with ID: {chat.chat_id}")
                return chat.chat_id
            
        except Exception as e:
            logger.error(f"Error adding cursor chat: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def get_cursor_chats(self, 
                        cursor_project_id: Optional[int] = None, 
                        start_time: Optional[datetime] = None, 
                        end_time: Optional[datetime] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get cursor chats from the database.
        
        Args:
            cursor_project_id: ID of the cursor project (optional)
            start_time: Start time for filtering (optional)
            end_time: End time for filtering (optional)
            limit: Maximum number of chats to return
            
        Returns:
            List[Dict[str, Any]]: List of cursor chats
        """
        try:
            query = CursorChatModel.select()
            
            # Apply filters if provided
            if cursor_project_id is not None:
                query = query.where(CursorChatModel.cursor_project == cursor_project_id)
                
            if start_time is not None:
                query = query.where(CursorChatModel.timestamp >= start_time)
                
            if end_time is not None:
                query = query.where(CursorChatModel.timestamp <= end_time)
                
            # Order by timestamp descending and limit results
            query = query.order_by(CursorChatModel.timestamp.desc()).limit(limit)
            
            # Convert to dictionaries
            chats = list(query.dicts())
            
            # Parse files JSON if present
            for chat in chats:
                if chat.get('files'):
                    try:
                        chat['files'] = json.loads(chat['files'])
                    except json.JSONDecodeError:
                        chat['files'] = None
            
            return chats
            
        except Exception as e:
            logger.error(f"Error getting cursor chats: {e}")
            return [] 
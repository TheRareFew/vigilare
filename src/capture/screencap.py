"""Screen capture functionality."""

import logging
import os
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab, Image
from peewee import fn

from src.core.aw_client import ActivityWatchClient
from src.storage.models import ScreenshotModel, PromptModel, PromptTypeModel
from src.utils.helpers import get_screenshot_path
from src.vision.blur import process_screenshot
from src.analysis.image_analyzer import ImageAnalyzer
from src.core.database import get_data_dir
from src.capture.cursor_tracker import CursorTracker
from src.storage.operations import DatabaseOperations

logger = logging.getLogger(__name__)

class ScreenCapture:
    """Handles screen capture functionality."""
    
    def __init__(self, config: Dict, aw_client: ActivityWatchClient):
        """Initialize screen capture.
        
        Args:
            config: Screenshot configuration
            aw_client: ActivityWatch client instance
        """
        self.config = config
        self.aw_client = aw_client
        self.quality = config.get('quality', 90)  # Reduced default quality
        self.blur_sensitive = config.get('blur_sensitive', True)  # Control blurring
        self.enable_ner = config.get('enable_ner', True)  # Control NER
        self.screenshots_dir = config.get('screenshots_dir', os.path.join(get_data_dir(), 'screenshots'))
        os.makedirs(self.screenshots_dir, exist_ok=True)
        self.max_dimension = config.get('max_dimension', 1800)  # Max width/height for screenshots
        
        # Initialize image analyzer for LLM analysis
        self.image_analyzer = ImageAnalyzer(
            model_name=config.get('analyzer', {}).get('model_name', 'gpt-4o-mini'),
            temperature=config.get('analyzer', {}).get('temperature', 0),
            max_tokens=config.get('analyzer', {}).get('max_tokens', 4096),
            api_key=config.get('analyzer', {}).get('api_key')
        )
        
        # Initialize database operations and cursor tracker
        self.db_ops = DatabaseOperations()
        self.cursor_tracker = CursorTracker(self.db_ops)
        
        logger.info("Initialized screen capture with config: %s", config)

    def capture_screenshot(self) -> Optional[str]:
        """Capture a screenshot.
        
        Returns:
            Optional[str]: Path to the saved screenshot or None if capture failed
        """
        try:
            logger.debug("Starting screenshot capture")
            
            # Capture screen
            logger.debug("Capturing screen with ImageGrab")
            screenshot = ImageGrab.grab(bbox=None, include_layered_windows=False)
            logger.debug(f"Initial screenshot size: {screenshot.size}")
            
            # Convert to RGB (removes alpha channel if present)
            screenshot = screenshot.convert('RGB')
            logger.debug("Converted to RGB format")
            
            # Resize if the image is too large while maintaining aspect ratio
            max_dim = max(screenshot.size)
            if max_dim > self.max_dimension:
                ratio = self.max_dimension / max_dim
                new_size = tuple(int(dim * ratio) for dim in screenshot.size)
                logger.debug(f"Resizing image from {screenshot.size} to {new_size}")
                screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
            
            # Process screenshot if blur is enabled
            if self.blur_sensitive:
                logger.debug("Processing screenshot with blur")
                screenshot = process_screenshot(screenshot, enable_ner=self.enable_ner)
            
            # Generate path and save
            path = get_screenshot_path(self.screenshots_dir)
            logger.debug(f"Saving screenshot to: {path}")
            screenshot.save(path, 'JPEG', quality=self.quality, optimize=True, progressive=True)
            
            logger.debug(f"Successfully captured and saved screenshot: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}", exc_info=True)
            return None

    def store_screenshot(self, image_path: str, window_info: Dict[str, str] = None):
        """Store screenshot metadata in the database.
        
        Args:
            image_path: Path to the screenshot image
            window_info: Window information
        """
        try:
            if not window_info:
                window_info = self.aw_client.get_current_window()
                
            if not window_info:
                logger.warning("No window information available, skipping database storage")
                return
                
            logger.debug(f"Window info: {window_info}")
                
            # Check if this is Cursor and process chat data if it is
            app = window_info.get('app', '')
            if app and 'cursor' in app.lower():
                logger.info(f"Detected Cursor window: {app}, processing chat data")
                # Get active project
                active_project = self.db_ops.get_active_cursor_project()
                project_id = active_project['project_id'] if active_project else None
                
                # Process all cursor chat data
                self.cursor_tracker.process_all_cursor_chat_data(project_id)
                
            # Create context dictionary
            context = {
                'window_title': window_info.get('title', ''),
                'application_name': window_info.get('app', '')
            }
            
            logger.debug(f"Storing screenshot metadata for: {image_path}")
            if window_info:
                logger.debug(f"Window info: {window_info}")
            else:
                window_info = {}
                logger.debug("No window info provided")
            
            # Create screenshot model
            screenshot = ScreenshotModel.create(
                image_path=image_path,
                context=context
            )
            logger.debug(f"Created screenshot record with ID: {screenshot.image_id}")
            
            # Always analyze the screenshot with LLM, even if NER is disabled
            try:
                # Load image for analysis
                logger.debug("Loading image for analysis")
                image = Image.open(image_path)
                
                # Analyze image with context
                logger.debug("Starting image analysis")
                analysis = self.image_analyzer.analyze_image(
                    image,
                    context=context
                )
                
                # Update screenshot with analysis results
                if analysis:
                    logger.debug("Processing analysis results")
                    # Store full_analysis in image_summary field
                    screenshot.image_summary = analysis.get('full_analysis', '')
                    
                    # Store code insights if present
                    if 'code_insights' in analysis:
                        screenshot.set_code_insights(analysis['code_insights'])
                    
                    # Store detected prompts
                    if 'prompts' in analysis:
                        for prompt_data in analysis['prompts']:
                            # Get or create prompt type
                            prompt_type, _ = PromptTypeModel.get_or_create(
                                prompt_type_name=prompt_data.get('prompt_type', 'other')
                            )
                            
                            # Check for duplicate prompt
                            prompt_text = prompt_data.get('prompt_text', '').strip()
                            if not prompt_text:
                                continue
                                
                            # Look for exact match of prompt text and type
                            existing_prompt = (PromptModel
                                .select()
                                .join(PromptTypeModel)
                                .where(
                                    (fn.LOWER(PromptModel.prompt_text) == prompt_text.lower()) &
                                    (PromptTypeModel.prompt_type_id == prompt_type.prompt_type_id)
                                )
                                .first())
                            
                            if not existing_prompt:
                                # Create new prompt record only if no duplicate exists
                                PromptModel.create(
                                    prompt_text=prompt_text,
                                    prompt_type=prompt_type,
                                    model_name=prompt_data.get('model_name'),
                                    llm_tool_used=prompt_data.get('llm_tool_used'),
                                    confidence=prompt_data.get('confidence', 0.0),
                                    context=context
                                )
                                logger.debug(f"Stored new prompt: {prompt_text[:50]}...")
                            else:
                                logger.debug(f"Skipped duplicate prompt: {prompt_text[:50]}...")
                    
                    screenshot.save()
                    logger.debug(f"Analyzed and stored screenshot metadata: {screenshot.image_id}")
                
            except Exception as e:
                logger.error(f"Error analyzing screenshot: {e}")
                # Still return the screenshot even if analysis fails
            
            return screenshot
            
        except Exception as e:
            logger.error(f"Error storing screenshot metadata: {e}")
            return None 
"""Daemon process for running periodic tasks."""

import logging
import random
import time
from datetime import datetime
from typing import Dict

from src.capture.screencap import ScreenCapture
from src.core.aw_client import ActivityWatchClient
from src.storage.operations import DatabaseOperations

logger = logging.getLogger(__name__)

class Daemon:
    """Main daemon process for Vigilare."""
    
    def __init__(self, config: Dict, aw_client: ActivityWatchClient):
        """Initialize daemon.
        
        Args:
            config: Configuration dictionary
            aw_client: ActivityWatch client instance
        """
        self.config = config
        self.aw_client = aw_client
        self.current_activity_level = 'low'
        self.next_screenshot_time = 0
        self.last_cursor_check_time = 0
        self.cursor_check_interval = 5  # Check for Cursor projects every 5 seconds
        self.db_ops = DatabaseOperations()
        
        # Get screenshot configuration from capture section
        screenshot_config = config.get('capture', {}).get('screenshot', {})
        self.screen_capture = ScreenCapture(screenshot_config, aw_client)
        
        # Screenshot intervals
        self.min_interval = screenshot_config.get('min_interval', 60)
        self.max_interval = screenshot_config.get('max_interval', 120)
        
        # Activity monitoring settings
        activity_config = screenshot_config.get('activity', {})
        self.check_interval = activity_config.get('check_interval', 10)
        self.lookback_window = activity_config.get('lookback_window', 60)
        
        # Activity thresholds from config
        self.activity_thresholds = screenshot_config.get('activity_thresholds', {
            'high': {
                'presses': 20,    # More than 20 keypresses
                'clicks': 10,     # More than 10 clicks
                'movement': 500   # More than 500 pixels of mouse movement
            },
            'medium': {
                'presses': 10,    # 10-20 keypresses
                'clicks': 5,      # 5-10 clicks
                'movement': 200   # 200-500 pixels of mouse movement
            }
        })
        
        # Initialize screenshot timing
        self._update_next_screenshot_time()
        
        logger.info("Daemon initialized")
        
        # Log the thresholds being used
        logger.info("Using activity thresholds:")
        for level, thresholds in self.activity_thresholds.items():
            logger.info(f"{level.upper()}: {thresholds}")
            
        logger.info(f"Screenshot intervals: min={self.min_interval}s, max={self.max_interval}s")
        logger.info(f"Activity check interval: {self.check_interval}s")
        logger.info(f"Activity lookback window: {self.lookback_window}s")

    def _get_activity_level(self) -> str:
        """Determine activity level based on input metrics.
        
        Returns:
            str: Activity level ('high', 'medium', or 'low')
        """
        metrics = self.aw_client.get_input_activity(self.lookback_window)
        total_movement = metrics['deltaX'] + metrics['deltaY']
        
        # Check for high activity
        high_threshold = self.activity_thresholds['high']
        if (metrics['presses'] >= high_threshold['presses'] or
            metrics['clicks'] >= high_threshold['clicks'] or
            total_movement >= high_threshold['movement']):
            
            # Log which threshold triggered high activity
            reasons = []
            if metrics['presses'] >= high_threshold['presses']:
                reasons.append(f"keypresses ({metrics['presses']} >= {high_threshold['presses']})")
            if metrics['clicks'] >= high_threshold['clicks']:
                reasons.append(f"clicks ({metrics['clicks']} >= {high_threshold['clicks']})")
            if total_movement >= high_threshold['movement']:
                reasons.append(f"mouse movement ({total_movement}px >= {high_threshold['movement']}px)")
                
            logger.info(f"High activity detected due to: {', '.join(reasons)}")
            return 'high'
            
        # Check for medium activity
        medium_threshold = self.activity_thresholds['medium']
        if (metrics['presses'] >= medium_threshold['presses'] or
            metrics['clicks'] >= medium_threshold['clicks'] or
            total_movement >= medium_threshold['movement']):
            
            # Log which threshold triggered medium activity
            reasons = []
            if metrics['presses'] >= medium_threshold['presses']:
                reasons.append(f"keypresses ({metrics['presses']} >= {medium_threshold['presses']})")
            if metrics['clicks'] >= medium_threshold['clicks']:
                reasons.append(f"clicks ({metrics['clicks']} >= {medium_threshold['clicks']})")
            if total_movement >= medium_threshold['movement']:
                reasons.append(f"mouse movement ({total_movement}px >= {medium_threshold['movement']}px)")
                
            logger.info(f"Medium activity detected due to: {', '.join(reasons)}")
            return 'medium'
        
        # Log low activity metrics
        logger.info(
            f"Low activity detected: {metrics['presses']} keypresses, "
            f"{metrics['clicks']} clicks, {total_movement}px movement "
            f"(below medium thresholds: {medium_threshold['presses']} presses, "
            f"{medium_threshold['clicks']} clicks, {medium_threshold['movement']}px movement)"
        )
        return 'low'

    def _get_screenshot_interval(self, activity_level: str) -> float:
        """Get screenshot interval based on activity level.
        
        Args:
            activity_level: Current activity level ('high', 'medium', or 'low')
            
        Returns:
            float: Number of seconds to wait before next screenshot
        """
        if activity_level == 'high':
            # Use shorter interval for high activity
            interval = self.min_interval
            logger.debug(f"Using minimum interval for high activity: {interval:.2f}s")
            return interval
        elif activity_level == 'medium':
            # Use middle interval for medium activity
            interval = (self.min_interval + self.max_interval) / 2
            logger.debug(f"Using medium interval for medium activity: {interval:.2f}s")
            return interval
        else:
            # Use longer interval for low activity
            interval = self.max_interval
            logger.debug(f"Using maximum interval for low activity: {interval:.2f}s")
            return interval

    def _should_take_screenshot(self) -> bool:
        """Check if it's time to take a screenshot based on current activity and timing."""
        now = time.time()
        
        # If no next screenshot time is set, set it
        if self.next_screenshot_time is None:
            self.next_screenshot_time = now + self._get_screenshot_interval(self.current_activity_level)
            return False
            
        return now >= self.next_screenshot_time

    def _update_next_screenshot_time(self):
        """Update the next screenshot time based on current activity level."""
        interval = self._get_screenshot_interval(self.current_activity_level)
        current_time = time.time()
        
        # Always set next screenshot time relative to current time
        self.next_screenshot_time = current_time + interval
        logger.debug(f"Next screenshot scheduled in {interval:.2f}s")

    def start(self):
        """Start the daemon process."""
        logger.info("Starting daemon process")
        
        try:
            while True:
                current_time = time.time()
                
                # Check if user is active
                is_active = self.aw_client.is_user_active()
                logger.debug(f"User active status: {is_active}")
                
                if is_active:
                    # Get previous activity level for comparison
                    previous_level = self.current_activity_level
                    
                    # Update activity level
                    self.current_activity_level = self._get_activity_level()
                    
                    # If activity increased, update screenshot timing
                    if (
                        (previous_level == 'low' and self.current_activity_level in ['medium', 'high']) or
                        (previous_level == 'medium' and self.current_activity_level == 'high')
                    ):
                        self._update_next_screenshot_time()
                    
                    # Check for Cursor projects periodically
                    if current_time - self.last_cursor_check_time > self.cursor_check_interval:
                        logger.debug("Checking for Cursor project")
                        self._check_cursor_project()
                        self.last_cursor_check_time = current_time
                    
                    # Check if we should take a screenshot
                    if self._should_take_screenshot():
                        # Attempt to get current window info but don't require it
                        window_info = self.aw_client.get_current_window()
                        if window_info:
                            logger.debug(f"Current window info: {window_info}")
                        else:
                            logger.debug("No window info available")
                        
                        # Capture and store screenshot
                        logger.debug("Attempting to capture screenshot")
                        image_path = self.screen_capture.capture_screenshot()
                        if image_path:
                            logger.debug(f"Screenshot captured successfully at: {image_path}")
                            self.screen_capture.store_screenshot(image_path, window_info)
                            # Set next screenshot time after successful capture
                            self.next_screenshot_time = time.time() + self._get_screenshot_interval(self.current_activity_level)
                            logger.debug(f"Next screenshot scheduled in {self._get_screenshot_interval(self.current_activity_level):.2f}s")
                        else:
                            logger.warning("Failed to capture screenshot")
                else:
                    logger.debug("User inactive - skipping screenshot")
                    self.current_activity_level = 'low'
                    self._update_next_screenshot_time()
                
                # Sleep for the activity check interval
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Daemon process stopped")
        except Exception as e:
            logger.error(f"Error in daemon process: {e}")
            raise

    def _check_cursor_project(self):
        """Check for and update Cursor project information."""
        try:
            cursor_project = self.aw_client.get_cursor_project_from_window()
            if cursor_project:
                project_name = cursor_project.get("project_name")
                project_path = cursor_project.get("project_path")
                
                if project_name:
                    logger.debug(f"Detected Cursor project: {project_name}")
                    self.db_ops.update_cursor_project(project_name, project_path)
        except Exception as e:
            logger.error(f"Error checking Cursor project: {e}")

"""Daemon process for running periodic tasks."""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict
import re

from src.capture.screencap import ScreenCapture
from src.core.aw_client import ActivityWatchClient
from src.storage.operations import DatabaseOperations
from src.analysis.report_generator import ReportGenerator

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
        
        # Initialize report generator
        report_config = config.get('report', {})
        self.report_enabled = report_config.get('enabled', True)
        self.report_retry_interval = report_config.get('retry_interval', 900)  # Default: 15 minutes
        
        if self.report_enabled:
            self.report_generator = ReportGenerator(
                client_id=config.get('aw_client_id', 'vigilare_daemon'),
                model_name=report_config.get('model_name', 'gpt-4o-mini'),
                testing=True  # Always use testing mode to match server
            )
            
            # Initialize next report time
            self.next_report_time = self._calculate_next_report_time()
            logger.info(f"Automatic hourly reports enabled. Next report at: {datetime.fromtimestamp(self.next_report_time)}")
        else:
            self.report_generator = None
            self.next_report_time = None
            logger.info("Automatic hourly reports disabled")
        
        logger.info("Daemon initialized")
        
        # Log the thresholds being used
        logger.info("Using activity thresholds:")
        for level, thresholds in self.activity_thresholds.items():
            logger.info(f"{level.upper()}: {thresholds}")
            
        logger.info(f"Screenshot intervals: min={self.min_interval}s, max={self.max_interval}s")
        logger.info(f"Activity check interval: {self.check_interval}s")
        logger.info(f"Activity lookback window: {self.lookback_window}s")
        logger.info(f"Next hourly report scheduled at: {datetime.fromtimestamp(self.next_report_time)}")

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

    def _calculate_next_report_time(self) -> float:
        """Calculate the next time to generate an hourly report.
        
        Returns:
            float: Unix timestamp for the next report time
        """
        now = datetime.now()
        # Set next report time to the next hour
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour.timestamp()
    
    def _generate_hourly_report(self):
        """Generate an hourly productivity report."""
        if not self.report_enabled or not self.report_generator:
            logger.warning("Report generation is disabled. Skipping hourly report.")
            return
            
        try:
            logger.info("Generating hourly productivity report")
            report = self.report_generator.generate_hourly_report()
            
            if report:
                logger.info(f"Hourly report generated successfully with ID: {report.report_id}")
                # Schedule next report
                self.next_report_time = self._calculate_next_report_time()
                logger.info(f"Next hourly report scheduled at: {datetime.fromtimestamp(self.next_report_time)}")
            else:
                logger.error("Failed to generate hourly report")
                # Retry based on configured retry interval
                self.next_report_time = time.time() + self.report_retry_interval
                logger.info(f"Will retry report generation at: {datetime.fromtimestamp(self.next_report_time)}")
        except Exception as e:
            logger.error(f"Error generating hourly report: {e}")
            # Retry based on configured retry interval
            self.next_report_time = time.time() + self.report_retry_interval
            logger.info(f"Will retry report generation at: {datetime.fromtimestamp(self.next_report_time)}")

    def start(self):
        """Start the daemon process."""
        logger.info("Starting daemon process")
        
        try:
            while True:
                current_time = time.time()
                
                # Check if it's time to generate an hourly report
                if self.report_enabled and self.next_report_time and current_time >= self.next_report_time:
                    self._generate_hourly_report()
                
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
            if self.report_generator:
                self.report_generator.close()
        except Exception as e:
            logger.error(f"Error in daemon process: {e}")
            if self.report_generator:
                self.report_generator.close()
            raise

    def _check_cursor_project(self):
        """Check for and update Cursor project information."""
        try:
            logger.debug("Checking for Cursor project")
            
            # Get current window info
            window_info = self.aw_client.get_current_window()
            
            if window_info:
                app = window_info.get("app", "")
                title = window_info.get("title", "")
                logger.debug(f"Window info: app='{app}', title='{title}'")
                
                # Check if this is a Cursor window
                if app and "cursor" in app.lower():
                    logger.debug(f"Detected Cursor window: {app}")
                    
                    # Extract project name from window title
                    cursor_pattern = r'.*? - (.*?) - Cursor'
                    match = re.search(cursor_pattern, title)
                    
                    if match:
                        project_name = match.group(1)
                        logger.debug(f"Extracted project name: '{project_name}' from title: '{title}'")
                        
                        # Get cursor project info including paths
                        project_info = self.aw_client.get_cursor_project_from_window()
                        
                        if project_info:
                            # Update the project in the database with all available info
                            logger.debug(f"Updating Cursor project in database: {project_name}")
                            cursor_data_path = project_info.get("cursor_data_path", "")
                            success = self.db_ops.update_cursor_project(project_name, cursor_data_path)
                            logger.debug(f"Database update {'successful' if success else 'failed'}")
                        else:
                            # Fallback to just updating with the name
                            logger.debug(f"Updating Cursor project in database with name only: {project_name}")
                            success = self.db_ops.update_cursor_project(project_name)
                            logger.debug(f"Database update {'successful' if success else 'failed'}")
                    else:
                        logger.debug(f"Could not extract project name from title: '{title}'")
                else:
                    logger.debug(f"Not a Cursor window: {app}")
            else:
                logger.debug("No window info available - this may happen if the window watcher is not running or no window events were captured in the last second")
                # Try to check if the window watcher is running
                try:
                    buckets = self.aw_client.client.get_buckets()
                    window_buckets = [b for b in buckets if "window" in b.lower()]
                    if not window_buckets:
                        logger.warning("No window watcher buckets found. Make sure aw-watcher-window is running.")
                    else:
                        logger.debug(f"Window watcher buckets found: {window_buckets}")
                except Exception as bucket_err:
                    logger.warning(f"Could not check window watcher buckets: {bucket_err}")
            
        except Exception as e:
            logger.error(f"Error checking Cursor project: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

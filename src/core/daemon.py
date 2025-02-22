"""Daemon process for running periodic tasks."""

import logging
import random
import time
from typing import Dict

from src.capture.screencap import ScreenCapture
from src.core.aw_client import ActivityWatchClient

logger = logging.getLogger(__name__)

class Daemon:
    """Daemon process that runs periodic tasks."""
    
    def __init__(self, config: Dict, aw_client: ActivityWatchClient):
        """Initialize daemon.
        
        Args:
            config: Application configuration
            aw_client: ActivityWatch client instance
        """
        self.config = config
        self.aw_client = aw_client
        self.screen_capture = ScreenCapture(config['screenshot'], aw_client)
        
        # Screenshot intervals
        self.min_interval = config['screenshot']['min_interval']
        self.max_interval = config['screenshot']['max_interval']
        
        logger.info("Initialized daemon process")

    def start(self):
        """Start the daemon process."""
        logger.info("Starting daemon process")
        
        try:
            while True:
                # Check if user is active
                is_active = self.aw_client.is_user_active()
                #is_active = True
                logger.debug(f"User active status: {is_active}")
                
                if is_active:
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
                    else:
                        logger.warning("Failed to capture screenshot")
                else:
                    logger.debug("User inactive - skipping screenshot")
                
                # Use random interval between min and max
                #interval = random.uniform(self.min_interval, self.max_interval)
                interval = 60
                logger.debug(f"Sleeping for {interval:.2f} seconds")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Daemon process stopped")
        except Exception as e:
            logger.error(f"Error in daemon process: {e}")
            raise

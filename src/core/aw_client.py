"""ActivityWatch client integration."""

import logging
import platform
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import re

from aw_client import ActivityWatchClient as AWClient
from aw_core.models import Event

logger = logging.getLogger(__name__)

class ActivityWatchClient:
    """Client for interacting with ActivityWatch."""
    
    def __init__(self, client_id: str, testing: bool = False, server_url: Optional[str] = None):
        """Initialize ActivityWatch client.
        
        Args:
            client_id: Client identifier
            testing: Whether to run in testing mode
            server_url: Optional server URL to connect to
        """
        if server_url:
            parsed_url = urlparse(server_url)
            host = parsed_url.hostname
            port = parsed_url.port
            self.client = AWClient(client_id, host=host, port=port, testing=testing)
        else:
            self.client = AWClient(client_id, testing=testing)
            
        self.client_id = client_id
        self.testing = testing
        self.hostname = platform.node()
        
        logger.info(f"Initialized ActivityWatch client with ID: {client_id} (testing={testing})")
        
        # Ensure buckets exist
        try:
            self._ensure_buckets_exist()
        except Exception as e:
            logger.error(f"Error ensuring buckets exist: {e}")
            raise

    def _ensure_buckets_exist(self):
        """Ensure required buckets exist."""
        # Create window bucket
        full_bucket_id = f"aw-watcher-window_{self.hostname}"
        logger.info(f"Creating bucket: {full_bucket_id}")
        self.client.create_bucket(full_bucket_id, "currentwindow", queued=True)

    def is_user_active(self, lookback_seconds: int = 60) -> bool:
        """Check if the user has been active in the last N seconds.
        
        Args:
            lookback_seconds: Number of seconds to look back for activity
            
        Returns:
            bool: True if user was active, False otherwise
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(seconds=lookback_seconds)
            timeperiods = [(start, now)]

            # Query both keyboard and mouse events and combine them
            query = """
                events = flood(query_bucket(find_bucket("aw-watcher-window_")));
                not_afk = flood(query_bucket(find_bucket("aw-watcher-afk_")));
                not_afk = filter_keyvals(not_afk, "status", ["not-afk"]);
                events = filter_period_intersect(events, not_afk);
                RETURN = events;
            """
            
            logger.debug(f"Checking user activity from {start} to {now}")
            results = self.client.query(query, timeperiods)
            is_active = bool(results and results[0] and len(results[0]) > 0)
            logger.debug(f"Activity check results: {results}")
            logger.debug(f"User is {'active' if is_active else 'inactive'}")
            return is_active
            
        except Exception as e:
            if "bucket" in str(e).lower() and "not found" in str(e).lower():
                logger.warning("AFK watcher not running - no activity data available")
                return False
            logger.error(f"Error checking user activity: {e}")
            return False

    def get_current_window(self) -> Optional[Dict[str, str]]:
        """Get information about the currently active window.
        
        Returns:
            Optional[Dict[str, str]]: Window information or None if not available
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(seconds=1)
            timeperiods = [(start, now)]

            query = """
                window_events = flood(query_bucket(find_bucket("aw-watcher-window_")));
                RETURN = window_events;
            """

            logger.debug(f"Getting current window info from {start} to {now}")
            results = self.client.query(query, timeperiods)
            logger.debug(f"Window query results: {results}")
            
            if results and results[0]:
                window_event = results[0][0]
                window_info = {
                    "app": window_event["data"].get("app", ""),
                    "title": window_event["data"].get("title", "")
                }
                logger.debug(f"Current window info: {window_info}")
                return window_info
            logger.debug("No window info available")
            return None
            
        except Exception as e:
            if "bucket" in str(e).lower() and "not found" in str(e).lower():
                logger.warning("Window watcher not running - no window events available")
                return None
            logger.error(f"Error getting current window: {e}")
            return None

    def get_cursor_project_from_window(self) -> Optional[Dict[str, str]]:
        """Extract Cursor project information from the current window title.
        
        Returns:
            Optional[Dict[str, str]]: Project information or None if not available
        """
        try:
            logger.debug("Attempting to extract Cursor project from window title")
            window_info = self.get_current_window()
            
            if not window_info:
                logger.debug("No window info available")
                return None
                
            app = window_info.get("app", "")
            title = window_info.get("title", "")
            logger.debug(f"Window info: app='{app}', title='{title}'")
            
            # Check if this is a Cursor window
            if not app or "cursor" not in app.lower():
                logger.debug(f"Not a Cursor window: {app}")
                return None
                
            logger.debug(f"Detected Cursor window: {app}")
            
            # Extract project name from title
            # Pattern: "filename - project_name - Cursor"
            cursor_pattern = r'.*? - (.*?) - Cursor'
            match = re.search(cursor_pattern, title)
            
            if match:
                project_name = match.group(1)
                logger.debug(f"Extracted project name: '{project_name}' from title: '{title}'")
                
                return {
                    "project_name": project_name,
                    "project_path": ""  # We'll determine the path using helpers.get_most_recent_workspace_dir()
                }
            else:
                logger.debug(f"Could not extract project name from title: '{title}'")
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting Cursor project info: {e}")
            return None

    def get_current_vscode_file(self, lookback_seconds: int = 10) -> Optional[Dict[str, str]]:
        """Get information about the currently open file in VSCode/Cursor.
        
        Args:
            lookback_seconds: Number of seconds to look back for VSCode events
            
        Returns:
            Optional[Dict[str, str]]: File information or None if not available
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(seconds=lookback_seconds)
            timeperiods = [(start, now)]

            # Query for VSCode watcher events
            query = """
                vscode_events = flood(query_bucket(find_bucket("aw-watcher-vscode")));
                vscode_events = sort_by_timestamp(vscode_events);
                RETURN = vscode_events;
            """

            logger.debug(f"Getting current VSCode file info from {start} to {now}")
            logger.debug(f"Using testing mode: {self.testing} (port: {5666 if self.testing else 5600})")
            results = self.client.query(query, timeperiods)
            
            if not results or not results[0]:
                logger.debug("No VSCode events found in the specified time period")
                return None
                
            # Get the most recent event
            vscode_event = results[0][-1]  # Last event is most recent
            file_info = {
                "language": vscode_event["data"].get("language", ""),
                "project": vscode_event["data"].get("project", ""),
                "file": vscode_event["data"].get("file", "")
            }
            logger.debug(f"Current VSCode file info: {file_info}")
            return file_info
            
        except Exception as e:
            if "bucket" in str(e).lower() and "not found" in str(e).lower():
                logger.warning("VSCode watcher not running or bucket not found - no VSCode events available")
                return None
            elif "connection" in str(e).lower():
                logger.error(f"Connection error to ActivityWatch server: {e}")
                logger.info(f"Make sure ActivityWatch server is running on port {5666 if self.testing else 5600}")
                return None
            else:
                logger.error(f"Error getting current VSCode file: {e}")
                return None
            
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get the content of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[str]: File content or None if not available
        """
        try:
            # Normalize file path for Windows
            normalized_path = file_path
            
            # Check if file exists
            if not os.path.exists(normalized_path):
                logger.warning(f"File does not exist: {normalized_path}")
                # Try to handle Windows path format
                if '\\' in normalized_path and platform.system() == 'Windows':
                    # Try alternative path format
                    alternative_path = normalized_path.replace('\\', '/')
                    if os.path.exists(alternative_path):
                        normalized_path = alternative_path
                        logger.info(f"Using alternative path format: {normalized_path}")
                    else:
                        return None
                else:
                    return None
                
            # Read file content
            logger.debug(f"Reading file: {normalized_path}")
            with open(normalized_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"Read {len(content)} characters from {normalized_path}")
            return content
            
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"Unicode decode error with utf-8, trying with latin-1 encoding")
            try:
                with open(normalized_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.debug(f"Successfully read file with latin-1 encoding")
                return content
            except Exception as e:
                logger.error(f"Error reading file with latin-1 encoding: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def close(self):
        """Close the ActivityWatch client connection."""
        try:
            self.client.disconnect()
        except Exception as e:
            logger.error(f"Error closing ActivityWatch client: {e}")

    def get_input_activity(self, lookback_seconds: int = 60) -> Dict[str, int]:
        """Get input activity metrics for the last N seconds.
        
        Args:
            lookback_seconds: Number of seconds to look back for activity
            
        Returns:
            Dict with activity metrics (presses, clicks, deltaX, deltaY, scrollX, scrollY)
        """
        try:
            now = datetime.now(timezone.utc)
            start = now - timedelta(seconds=lookback_seconds)
            timeperiods = [(start, now)]

            # Query input events with proper time window handling
            query = """
                events = flood(query_bucket(find_bucket("aw-watcher-input_")));
                events = sort_by_timestamp(events);
                events = merge_events_by_keys(events, ["presses", "clicks", "deltaX", "deltaY", "scrollX", "scrollY"]);
                RETURN = events;
            """
            
            logger.debug(f"Getting input activity from {start} to {now}")
            results = self.client.query(query, timeperiods)
            
            # Initialize metrics
            metrics = {
                "presses": 0,
                "clicks": 0,
                "deltaX": 0,
                "deltaY": 0,
                "scrollX": 0,
                "scrollY": 0
            }
            
            # Sum up all input events
            event_count = 0
            if results and results[0]:
                event_count = len(results[0])
                for event in results[0]:
                    data = event.get("data", {})
                    metrics["presses"] += data.get("presses", 0)
                    metrics["clicks"] += data.get("clicks", 0)
                    metrics["deltaX"] += abs(data.get("deltaX", 0))
                    metrics["deltaY"] += abs(data.get("deltaY", 0))
                    metrics["scrollX"] += abs(data.get("scrollX", 0))
                    metrics["scrollY"] += abs(data.get("scrollY", 0))
            
            total_movement = metrics["deltaX"] + metrics["deltaY"]
            total_scroll = metrics["scrollX"] + metrics["scrollY"]
            
            logger.info(
                f"Activity metrics for past {lookback_seconds}s: "
                f"{event_count} events, "
                f"{metrics['presses']} keypresses, "
                f"{metrics['clicks']} clicks, "
                f"{total_movement}px mouse movement, "
                f"{total_scroll}px scroll"
            )
            return metrics
            
        except Exception as e:
            if "bucket" in str(e).lower() and "not found" in str(e).lower():
                logger.warning("Input watcher not running - no input events available")
                return {k: 0 for k in ["presses", "clicks", "deltaX", "deltaY", "scrollX", "scrollY"]}
            logger.error(f"Error getting input activity: {e}")
            return {k: 0 for k in ["presses", "clicks", "deltaX", "deltaY", "scrollX", "scrollY"]} 
"""Helper utilities for Vigilare."""

import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def ensure_dir(directory: str):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)

def get_screenshot_path(base_dir: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a path for a screenshot file.
    
    Args:
        base_dir: Base directory for screenshots
        timestamp: Timestamp for the screenshot (defaults to current time)
        
    Returns:
        str: Path for the screenshot file
    """
    timestamp = timestamp or datetime.now()
    date_dir = timestamp.strftime('%Y-%m-%d')
    filename = timestamp.strftime('%Y%m%d_%H%M%S.jpg')
    
    # Create date-based subdirectory
    full_dir = os.path.join(base_dir, date_dir)
    ensure_dir(full_dir)
    
    return os.path.join(full_dir, filename)

def format_time_delta(seconds: float) -> str:
    """Format a time delta in a human-readable format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{remaining_seconds}s")
    
    return " ".join(parts)

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to be safe for all operating systems.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and periods
    filename = filename.strip('. ')
    
    # Ensure filename isn't too long (Windows has a 255 character limit)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def get_file_size(file_path: str) -> int:
    """Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: Size of the file in bytes
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, IOError):
        return 0

def format_file_size(size_in_bytes: int) -> str:
    """Format a file size in a human-readable format.
    
    Args:
        size_in_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f}{unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f}TB"

def get_cursor_workspace_storage_path() -> str:
    """Get the path to the Cursor workspaceStorage directory.
    
    Returns:
        str: Path to the Cursor workspaceStorage directory
    """
    user_home = os.path.expanduser("~")
    path = os.path.join(user_home, "AppData", "Roaming", "Cursor", "User", "workspaceStorage")
    logger.debug(f"Cursor workspaceStorage path: '{path}'")
    return path

def get_most_recent_workspace_dir() -> str:
    """Get the most recently modified directory within the Cursor workspaceStorage directory.
    
    Returns:
        str: Path to the most recently modified workspace directory
    """
    workspace_storage_path = get_cursor_workspace_storage_path()
    
    if not os.path.exists(workspace_storage_path):
        logger.debug(f"Workspace storage path does not exist: '{workspace_storage_path}'")
        return ""
    
    logger.debug(f"Scanning workspace storage directory: '{workspace_storage_path}'")
    dirs = [os.path.join(workspace_storage_path, d) for d in os.listdir(workspace_storage_path) 
            if os.path.isdir(os.path.join(workspace_storage_path, d))]
    
    if not dirs:
        logger.debug("No workspace directories found")
        return ""
    
    logger.debug(f"Found {len(dirs)} workspace directories")
    
    # Sort directories by modification time (most recent first)
    most_recent_dir = max(dirs, key=os.path.getmtime)
    
    # Get the modification time for logging
    mod_time = datetime.fromtimestamp(os.path.getmtime(most_recent_dir))
    logger.debug(f"Most recent workspace directory: '{most_recent_dir}' (modified: {mod_time.isoformat()})")
    
    return most_recent_dir 
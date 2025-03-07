"""Helper utilities for Vigilare."""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import re
import json
import sqlite3
from pathlib import Path
from urllib.parse import unquote

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

def get_most_recent_workspace_dir() -> Optional[str]:
    """Get the most recently used Cursor workspace directory.
    
    Returns:
        Optional[str]: Path to the most recent workspace directory or None if not found
    """
    try:
        # Get the Cursor workspace storage directory
        cursor_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Cursor", "User", "workspaceStorage")
        
        if not os.path.exists(cursor_dir):
            logger.warning(f"Cursor workspace storage directory not found: {cursor_dir}")
            return None
            
        # Get all workspace directories
        workspace_dirs = [os.path.join(cursor_dir, d) for d in os.listdir(cursor_dir) 
                         if os.path.isdir(os.path.join(cursor_dir, d))]
        
        if not workspace_dirs:
            logger.warning("No Cursor workspace directories found")
            return None
            
        # Sort by modification time (most recent first)
        workspace_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        
        # Return the most recent directory
        return workspace_dirs[0]
        
    except Exception as e:
        logger.error(f"Error getting most recent workspace directory: {e}")
        return None

def extract_project_path_from_cursor_db(cursor_data_path: str) -> Optional[str]:
    """Extract the actual project path from the Cursor SQLite database.
    
    Args:
        cursor_data_path: Path to the Cursor workspace storage directory
        
    Returns:
        Optional[str]: The actual project path or None if not found
    """
    try:
        if not cursor_data_path or not os.path.exists(cursor_data_path):
            logger.warning(f"Invalid Cursor data path: {cursor_data_path}")
            return None
            
        # Path to the SQLite database
        db_path = os.path.join(cursor_data_path, "state.vscdb")
        
        if not os.path.exists(db_path):
            logger.warning(f"Cursor state database not found: {db_path}")
            return None
            
        logger.debug(f"Connecting to Cursor database: {db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query for the debug.selectedroot key
        cursor.execute("SELECT value FROM ItemTable WHERE key = 'debug.selectedroot'")
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            logger.warning("No debug.selectedroot key found in Cursor database")
            return None
            
        # Extract the path from the result
        file_uri = result[0]
        logger.debug(f"Found project path URI: {file_uri}")
        
        # Parse the URI to get the actual path
        # Example: file:///c%3A/gauntlet/activitywatch_ai/.vscode/launch.json
        if file_uri.startswith("file:///"):
            # Remove the file:/// prefix
            path = file_uri[8:]
            
            # URL decode the path
            path = unquote(path)
            
            # Convert to proper Windows path if needed
            if path.startswith("c:") or path.startswith("C:"):
                path = path.replace("/", "\\")
            
            # Remove the .vscode/launch.json or any other file part
            path = Path(path)
            if ".vscode" in path.parts:
                # Remove .vscode and everything after it
                vscode_index = path.parts.index(".vscode")
                path = Path(*path.parts[:vscode_index])
            elif path.is_file():
                # If it's a file path, get the parent directory
                path = path.parent
                
            logger.debug(f"Extracted project path: {path}")
            return str(path)
        else:
            logger.warning(f"Unexpected URI format: {file_uri}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting project path from Cursor database: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None

def get_cursor_current_file_path(cursor_data_path: str, file_name: str) -> Optional[str]:
    """Get the full path to the current file open in Cursor.
    
    Args:
        cursor_data_path: Path to the Cursor workspace storage directory
        file_name: Name of the file currently open in Cursor
        
    Returns:
        Optional[str]: Full path to the file or None if not found
    """
    try:
        if not cursor_data_path or not os.path.exists(cursor_data_path):
            logger.warning(f"Invalid Cursor data path: {cursor_data_path}")
            return None
            
        # Path to the embeddable_files.txt
        embeddable_files_path = os.path.join(cursor_data_path, "anysphere.cursor-retrieval", "embeddable_files.txt")
        
        if not os.path.exists(embeddable_files_path):
            logger.warning(f"Embeddable files list not found: {embeddable_files_path}")
            return None
            
        logger.debug(f"Reading embeddable files from: {embeddable_files_path}")
        
        # Read the embeddable_files.txt file
        with open(embeddable_files_path, 'r', encoding='utf-8') as f:
            file_paths = f.read().splitlines()
        
        # Clean up the file name (remove any dots or other extras)
        clean_file_name = file_name.split('.')[0].strip()
        logger.debug(f"Looking for file with name: '{clean_file_name}'")
        
        # Find the file path that ends with the file name
        for path in file_paths:
            # Normalize path separators
            normalized_path = path.replace('\\', '/')
            
            # Get just the file name without extension
            path_file_name = os.path.basename(normalized_path).split('.')[0].strip()
            
            if path_file_name == clean_file_name:
                logger.debug(f"Found matching file: {path}")
                
                # Get the project path from the database
                project_info = get_active_cursor_project_info()
                
                if project_info and project_info.get('project_path'):
                    # Combine project path with the relative file path
                    full_path = os.path.join(project_info['project_path'], path)
                    logger.debug(f"Full file path: {full_path}")
                    return full_path
                else:
                    logger.warning("No active project path found in database")
                    return None
        
        logger.warning(f"No matching file found for: {file_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting Cursor file path: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None

def get_active_cursor_project_info() -> Optional[Dict[str, Any]]:
    """Get information about the active Cursor project from the database.
    
    Returns:
        Optional[Dict[str, Any]]: Project information or None if not found
    """
    try:
        from src.storage.operations import DatabaseOperations
        
        db_ops = DatabaseOperations()
        project_info = db_ops.get_active_cursor_project()
        
        return project_info
        
    except Exception as e:
        logger.error(f"Error getting active Cursor project info: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None 
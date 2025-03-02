"""Helper utilities for Vigilare."""

import os
from datetime import datetime
from typing import Optional

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
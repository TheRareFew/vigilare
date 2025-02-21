"""Logging configuration for Vigilare."""

import logging
import logging.config
import os
from typing import Optional

import yaml

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def setup_logging(config_path: str, debug: bool = False):
    """Set up logging configuration.
    
    Args:
        config_path: Path to the logging configuration file
        debug: Whether to enable debug logging
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure log directory exists and fix paths
        log_handlers = config.get('handlers', {})
        for handler in log_handlers.values():
            if 'filename' in handler:
                # Make path absolute relative to project root
                handler['filename'] = os.path.abspath(os.path.join(PROJECT_ROOT, handler['filename']))
                log_dir = os.path.dirname(handler['filename'])
                os.makedirs(log_dir, exist_ok=True)
        
        # Override log levels if debug is enabled
        if debug:
            config['handlers']['console']['level'] = 'DEBUG'
            config['loggers']['vigilare']['level'] = 'DEBUG'
        
        logging.config.dictConfig(config)
        logger = logging.getLogger('vigilare')
        logger.info("Logging configured successfully")
        
    except Exception as e:
        # Set up basic logging if configuration fails
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Error configuring logging: {e}")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Name of the logger (defaults to 'vigilare')
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name or 'vigilare') 
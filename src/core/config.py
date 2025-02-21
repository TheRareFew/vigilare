"""Configuration management for Vigilare."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parents[2] / '.env'
load_dotenv(env_path)

def get_required_env(key: str) -> str:
    """Get a required environment variable.
    
    Args:
        key: Environment variable key
        
    Returns:
        str: Environment variable value
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value

def setup_openai_key():
    """Setup OpenAI API key from environment variable."""
    api_key = get_required_env('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = api_key 
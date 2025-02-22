"""Configuration management for Vigilare."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parents[2] / '.env'
logger.debug(f"Loading .env file from: {env_path}")
if env_path.exists():
    logger.debug(f".env file exists at {env_path}")
    # Debug: Print file contents
    with open(env_path) as f:
        contents = f.read()
        logger.debug(f"Contents of .env file:\n{contents}")
    load_dotenv(env_path, override=True)  # Force override of existing env vars
    logger.debug(f"Loaded environment variables. LANGCHAIN_API_KEY present: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
    # Debug: Print all environment variables starting with LANGCHAIN
    for key, value in os.environ.items():
        if key.startswith('LANGCHAIN'):
            logger.debug(f"Found env var: {key}=<present>")
else:
    logger.warning(f".env file not found at {env_path}")

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
    logger.debug(f"Getting required env var {key}: {'present' if value else 'missing'}")
    if value is None:
        if key == 'OPENAI_API_KEY':
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable. "
                "You can get an API key from https://platform.openai.com/api-keys"
            )
        elif key == 'LANGCHAIN_API_KEY':
            raise ValueError(
                "LangSmith API key not found. Please set the LANGCHAIN_API_KEY environment variable. "
                "You can get an API key from https://smith.langchain.com/settings"
            )
        else:
            raise ValueError(f"Required environment variable '{key}' is not set")
    return value

def setup_openai_key():
    """Setup OpenAI API key from environment variable."""
    api_key = get_required_env('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = api_key

def setup_langchain():
    """Setup LangChain environment variables."""
    logger.debug("Setting up LangChain environment")
    # Get values from already loaded .env file
    api_key = os.getenv('LANGCHAIN_API_KEY')
    logger.debug(f"Found LANGCHAIN_API_KEY in environment: {bool(api_key)}")
    project = os.getenv('LANGCHAIN_PROJECT', 'vigilare')
    
    # Set required environment variables for LangChain
    if api_key:
        os.environ['LANGCHAIN_API_KEY'] = api_key
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = project
        logger.debug("Successfully set up LangChain environment variables")
    else:
        logger.error(f"LANGCHAIN_API_KEY not found in environment or .env file at {env_path}")
        raise ValueError(
            "LangSmith API key not found in .env file. Please set LANGCHAIN_API_KEY in your .env file. "
            "You can get an API key from https://smith.langchain.com/settings"
        )

def setup_environment():
    """Setup all required environment variables."""
    setup_openai_key()
    setup_langchain() 
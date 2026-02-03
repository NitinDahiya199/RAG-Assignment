"""
Utility Functions
Helper functions for the Document Q&A AI Agent.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_api_key() -> Optional[str]:
    """
    Get Gemini API key from environment variables.
    
    Returns:
        API key string or None if not found
    """
    return os.getenv("GEMINI_API_KEY")


def validate_api_key() -> bool:
    """
    Validate that the API key is set.
    
    Returns:
        True if API key is set, False otherwise
    """
    api_key = get_api_key()
    if not api_key:
        logging.warning("GEMINI_API_KEY not found in environment variables")
        return False
    return True

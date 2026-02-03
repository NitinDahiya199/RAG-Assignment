"""
Input Validator Module
Validates and sanitizes user inputs for security and correctness.
"""

import os
import re
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class InputValidator:
    """
    Validates and sanitizes user inputs.
    """
    
    def __init__(self):
        """Initialize the InputValidator."""
        # Allowed file extensions
        self.allowed_extensions = {'.pdf'}
        
        # Maximum file size (100 MB)
        self.max_file_size = 100 * 1024 * 1024
        
        # Maximum query length (5000 characters)
        self.max_query_length = 5000
        
        # Dangerous patterns to filter
        self.dangerous_patterns = [
            r'\.\./',  # Path traversal
            r'<script',  # XSS attempts
            r'eval\s*\(',  # Code execution
            r'exec\s*\(',
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty."
        
        if not isinstance(query, str):
            return False, "Query must be a string."
        
        if len(query) > self.max_query_length:
            return False, f"Query too long. Maximum length is {self.max_query_length} characters."
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected in query: {pattern}")
                return False, "Query contains invalid characters or patterns."
        
        return True, None
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize user query.
        
        Args:
            query: User query string
            
        Returns:
            Sanitized query string
        """
        # Remove null bytes
        query = query.replace('\x00', '')
        
        # Strip whitespace
        query = query.strip()
        
        # Limit length
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
            logger.warning(f"Query truncated to {self.max_query_length} characters")
        
        return query
    
    def validate_file_path(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path:
            return False, "File path cannot be empty."
        
        # Check for path traversal
        if '..' in file_path or file_path.startswith('/'):
            return False, "Invalid file path."
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        # Check file extension
        path = Path(file_path)
        if path.suffix.lower() not in self.allowed_extensions:
            return False, f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}"
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return False, f"File too large. Maximum size is {self.max_file_size / (1024*1024):.1f} MB."
            
            if file_size == 0:
                return False, "File is empty."
        except OSError as e:
            return False, f"Error accessing file: {str(e)}"
        
        return True, None
    
    def validate_document_id(self, document_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate document identifier.
        
        Args:
            document_id: Document ID string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not document_id:
            return False, "Document ID cannot be empty."
        
        if not isinstance(document_id, str):
            return False, "Document ID must be a string."
        
        # Check for dangerous characters
        if re.search(r'[<>"\']', document_id):
            return False, "Document ID contains invalid characters."
        
        # Limit length
        if len(document_id) > 255:
            return False, "Document ID too long. Maximum length is 255 characters."
        
        return True, None
    
    def validate_api_key(self, api_key: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate API key format (basic validation).
        
        Args:
            api_key: API key string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, "API key is required."
        
        if not isinstance(api_key, str):
            return False, "API key must be a string."
        
        if len(api_key) < 10:
            return False, "API key appears to be invalid (too short)."
        
        # Basic format check (adjust based on actual API key format)
        if len(api_key) > 500:
            return False, "API key appears to be invalid (too long)."
        
        return True, None
    
    def validate_search_params(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate search parameters.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate query
        is_valid, error = self.validate_query(query)
        if not is_valid:
            return False, error
        
        # Validate max_results
        if max_results is not None:
            if not isinstance(max_results, int):
                return False, "max_results must be an integer."
            
            if max_results < 1:
                return False, "max_results must be at least 1."
            
            if max_results > 100:
                return False, "max_results cannot exceed 100."
        
        return True, None

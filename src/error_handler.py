"""
Error Handler Module
Provides comprehensive error handling and user-friendly error messages.
"""

import logging
from typing import Dict, Optional, Any
import traceback

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class ErrorHandler:
    """
    Handles errors gracefully and provides user-friendly messages.
    """
    
    def __init__(self):
        """Initialize the ErrorHandler."""
        self.error_messages = {
            "api_rate_limit": "API rate limit exceeded. Please try again in a few moments.",
            "api_network_error": "Network error connecting to API. Please check your internet connection.",
            "api_invalid_key": "Invalid API key. Please check your configuration.",
            "document_not_found": "The requested document could not be found.",
            "document_processing_error": "Error processing document. The file may be corrupted or in an unsupported format.",
            "query_processing_error": "Error processing your query. Please try rephrasing your question.",
            "no_results_found": "No relevant information found. Try different search terms or check if documents are indexed.",
            "vector_store_error": "Error accessing document database. Please try again.",
            "embedding_error": "Error generating embeddings. Please try again.",
            "general_error": "An unexpected error occurred. Please try again or contact support."
        }
    
    def handle_error(
        self,
        error: Exception,
        error_type: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Handle an error and return user-friendly response.
        
        Args:
            error: The exception that occurred
            error_type: Type of error (for specific handling)
            context: Additional context about the error
            
        Returns:
            Dictionary with error information and user message
        """
        error_name = type(error).__name__
        error_message = str(error)
        
        # Log the error
        logger.error(
            f"Error occurred: {error_name} - {error_message}",
            exc_info=True,
            extra=context or {}
        )
        
        # Determine error type if not provided
        if not error_type:
            error_type = self._classify_error(error)
        
        # Get user-friendly message
        user_message = self._get_user_message(error_type, error_message)
        
        # Build error response
        error_response = {
            "success": False,
            "error_type": error_type,
            "user_message": user_message,
            "technical_details": error_message if logger.level == logging.DEBUG else None,
            "context": context
        }
        
        return error_response
    
    def _classify_error(self, error: Exception) -> str:
        """
        Classify error type based on exception.
        
        Args:
            error: The exception
            
        Returns:
            Error type string
        """
        error_name = type(error).__name__
        error_message = str(error).lower()
        
        # API errors
        if "rate limit" in error_message or "429" in error_message:
            return "api_rate_limit"
        if "network" in error_message or "connection" in error_message or "timeout" in error_message:
            return "api_network_error"
        if "api key" in error_message or "authentication" in error_message or "401" in error_message:
            return "api_invalid_key"
        
        # Document errors
        if "file not found" in error_message or "FileNotFoundError" in error_name:
            return "document_not_found"
        if "pdf" in error_message or "document" in error_message:
            return "document_processing_error"
        
        # Query errors
        if "query" in error_message or "search" in error_message:
            return "query_processing_error"
        
        # Vector store errors
        if "vector" in error_message or "embedding" in error_message:
            return "vector_store_error"
        
        # Default
        return "general_error"
    
    def _get_user_message(self, error_type: str, technical_message: str) -> str:
        """
        Get user-friendly error message.
        
        Args:
            error_type: Type of error
            technical_message: Technical error message
            
        Returns:
            User-friendly message
        """
        return self.error_messages.get(
            error_type,
            self.error_messages["general_error"]
        )
    
    def handle_api_error(self, error: Exception, api_name: str = "API") -> Dict[str, Any]:
        """
        Handle API-specific errors.
        
        Args:
            error: The API error
            api_name: Name of the API (e.g., "Gemini", "Arxiv")
            
        Returns:
            Error response dictionary
        """
        error_message = str(error).lower()
        
        if "rate limit" in error_message or "429" in error_message:
            return self.handle_error(error, "api_rate_limit", {"api": api_name})
        elif "network" in error_message or "connection" in error_message:
            return self.handle_error(error, "api_network_error", {"api": api_name})
        elif "key" in error_message or "auth" in error_message:
            return self.handle_error(error, "api_invalid_key", {"api": api_name})
        else:
            return self.handle_error(error, "general_error", {"api": api_name})
    
    def handle_processing_error(
        self,
        error: Exception,
        operation: str = "processing"
    ) -> Dict[str, Any]:
        """
        Handle document/processing errors.
        
        Args:
            error: The processing error
            operation: Type of operation (e.g., "document processing", "embedding")
            
        Returns:
            Error response dictionary
        """
        return self.handle_error(
            error,
            "document_processing_error",
            {"operation": operation}
        )
    
    def handle_query_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Handle query processing errors.
        
        Args:
            error: The query error
            query: The query that caused the error
            
        Returns:
            Error response dictionary
        """
        return self.handle_error(
            error,
            "query_processing_error",
            {"query": query[:100]}  # Limit query length in context
        )
    
    def create_safe_response(self, error_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a safe response that can be returned to user.
        
        Args:
            error_response: Error response dictionary
            
        Returns:
            Safe response dictionary
        """
        return {
            "answer": error_response.get("user_message", "An error occurred."),
            "sources": [],
            "confidence": 0.0,
            "error": True,
            "error_type": error_response.get("error_type")
        }

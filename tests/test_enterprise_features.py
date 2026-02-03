"""
Tests for Enterprise Features (Phase 6).
"""

import pytest
import time
from src.conversation_manager import ConversationManager
from src.response_cache import ResponseCache
from src.error_handler import ErrorHandler
from src.rate_limiter import RateLimiter
from src.input_validator import InputValidator


def test_conversation_manager():
    """Test ConversationManager."""
    manager = ConversationManager(max_history=5)
    
    # Add turns
    manager.add_turn("What is AI?", {"answer": "AI is..."})
    manager.add_turn("What about ML?", {"answer": "ML is..."})
    
    assert len(manager.history) == 2
    
    # Get context
    context = manager.get_context(include_recent=2)
    assert len(context) == 2
    
    # Test history limit
    for i in range(10):
        manager.add_turn(f"Query {i}", {"answer": f"Answer {i}"})
    
    assert len(manager.history) == 5  # Should be limited to max_history


def test_response_cache():
    """Test ResponseCache."""
    cache = ResponseCache(ttl=60, max_size=10)
    
    query = "What is AI?"
    response = {"answer": "AI is artificial intelligence", "confidence": 0.9}
    
    # Set cache
    cache.set(query, response)
    
    # Get from cache
    cached = cache.get(query)
    assert cached == response
    
    # Test cache miss
    cached = cache.get("Different query")
    assert cached is None


def test_error_handler():
    """Test ErrorHandler."""
    handler = ErrorHandler()
    
    # Test API error
    error = Exception("Rate limit exceeded")
    result = handler.handle_api_error(error, "Gemini")
    
    assert result["success"] is False
    assert "user_message" in result
    assert result["error_type"] == "api_rate_limit"
    
    # Test safe response
    safe = handler.create_safe_response(result)
    assert "answer" in safe
    assert safe["error"] is True


def test_rate_limiter():
    """Test RateLimiter."""
    limiter = RateLimiter(max_requests=3, time_window=60)
    
    # First 3 requests should be allowed
    assert limiter.is_allowed()[0] is True
    assert limiter.is_allowed()[0] is True
    assert limiter.is_allowed()[0] is True
    
    # 4th request should be blocked
    allowed, message = limiter.is_allowed()
    assert allowed is False
    assert message is not None
    
    # Test remaining
    limiter.reset()
    remaining = limiter.get_remaining()
    assert remaining == 3


def test_input_validator():
    """Test InputValidator."""
    validator = InputValidator()
    
    # Valid query
    is_valid, error = validator.validate_query("What is AI?")
    assert is_valid is True
    assert error is None
    
    # Empty query
    is_valid, error = validator.validate_query("")
    assert is_valid is False
    
    # Too long query
    long_query = "A" * 10000
    is_valid, error = validator.validate_query(long_query)
    assert is_valid is False
    
    # Sanitize query
    sanitized = validator.sanitize_query("  What is AI?  ")
    assert sanitized == "What is AI?"
    
    # Test file path validation
    is_valid, error = validator.validate_file_path("nonexistent.pdf")
    assert is_valid is False


def test_integration():
    """Test integration of enterprise features."""
    from src.query_engine import QueryEngine
    from src.vector_store import VectorStore
    
    # Initialize with all features
    vector_store = VectorStore()
    engine = QueryEngine(
        vector_store=vector_store,
        enable_caching=True,
        enable_rate_limiting=True,
        enable_conversation=True
    )
    
    # Test that all features are initialized
    assert engine.cache is not None
    assert engine.rate_limiter is not None
    assert engine.conversation_manager is not None
    assert engine.error_handler is not None
    assert engine.input_validator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

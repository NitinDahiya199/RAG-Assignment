"""
Example usage of Enterprise Features (Phase 6).
Demonstrates conversation management, caching, error handling, and rate limiting.
"""

from src.query_engine import QueryEngine
from src.vector_store import VectorStore
from src.conversation_manager import ConversationManager
from src.response_cache import ResponseCache
from src.error_handler import ErrorHandler
from src.rate_limiter import RateLimiter
from src.input_validator import InputValidator
from src.utils import setup_logging, get_api_key

setup_logging("INFO")


def main():
    """Example usage of Enterprise Features."""
    
    print("=" * 60)
    print("Document Q&A AI Agent - Enterprise Features Demo (Phase 6)")
    print("=" * 60)
    print()
    
    # Example 1: Conversation Management
    print("=" * 60)
    print("EXAMPLE 1: Conversation Management")
    print("=" * 60)
    
    manager = ConversationManager(max_history=5)
    
    manager.add_turn("What is machine learning?", {"answer": "ML is a subset of AI..."})
    manager.add_turn("What about deep learning?", {"answer": "Deep learning uses neural networks..."})
    
    print(f"Conversation turns: {len(manager.history)}")
    print(f"History summary: {manager.get_history_summary()}")
    
    context = manager.get_context_string(include_recent=2)
    print(f"\nRecent context:\n{context[:200]}...")
    print()
    
    # Example 2: Response Caching
    print("=" * 60)
    print("EXAMPLE 2: Response Caching")
    print("=" * 60)
    
    cache = ResponseCache(ttl=3600, max_size=100)
    
    query = "What is artificial intelligence?"
    response = {"answer": "AI is...", "confidence": 0.9}
    
    # Cache response
    cache.set(query, response)
    print(f"Cached response for: {query}")
    
    # Retrieve from cache
    cached = cache.get(query)
    if cached:
        print("Cache hit! Retrieved from cache")
    else:
        print("Cache miss")
    
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    print()
    
    # Example 3: Error Handling
    print("=" * 60)
    print("EXAMPLE 3: Error Handling")
    print("=" * 60)
    
    handler = ErrorHandler()
    
    # Simulate different error types
    errors = [
        (Exception("Rate limit exceeded"), "api_rate_limit"),
        (FileNotFoundError("Document not found"), "document_not_found"),
        (ValueError("Invalid query"), "query_processing_error")
    ]
    
    for error, error_type in errors:
        result = handler.handle_error(error, error_type)
        print(f"Error Type: {error_type}")
        print(f"User Message: {result['user_message']}")
        print()
    
    # Example 4: Rate Limiting
    print("=" * 60)
    print("EXAMPLE 4: Rate Limiting")
    print("=" * 60)
    
    limiter = RateLimiter(max_requests=3, time_window=60)
    
    print("Making requests...")
    for i in range(5):
        allowed, message = limiter.is_allowed(user_id="test_user", operation=f"query_{i}")
        if allowed:
            print(f"  Request {i+1}: Allowed")
        else:
            print(f"  Request {i+1}: Blocked - {message}")
    
    stats = limiter.get_stats("test_user")
    print(f"\nRate limit stats: {stats}")
    print()
    
    # Example 5: Input Validation
    print("=" * 60)
    print("EXAMPLE 5: Input Validation")
    print("=" * 60)
    
    validator = InputValidator()
    
    test_inputs = [
        "What is AI?",
        "",
        "A" * 10000,  # Too long
        "test<script>alert('xss')</script>",  # Dangerous pattern
    ]
    
    for test_input in test_inputs:
        is_valid, error = validator.validate_query(test_input)
        print(f"Input: {test_input[:50]}...")
        print(f"  Valid: {is_valid}")
        if not is_valid:
            print(f"  Error: {error}")
        print()
    
    # Example 6: Integrated QueryEngine
    print("=" * 60)
    print("EXAMPLE 6: QueryEngine with Enterprise Features")
    print("=" * 60)
    
    vector_store = VectorStore()
    query_engine = QueryEngine(
        api_key=get_api_key(),
        vector_store=vector_store,
        enable_caching=True,
        enable_rate_limiting=True,
        enable_conversation=True
    )
    
    print("QueryEngine initialized with:")
    print(f"  - Caching: {query_engine.cache is not None}")
    print(f"  - Rate Limiting: {query_engine.rate_limiter is not None}")
    print(f"  - Conversation: {query_engine.conversation_manager is not None}")
    print(f"  - Error Handler: {query_engine.error_handler is not None}")
    print(f"  - Input Validator: {query_engine.input_validator is not None}")
    print()
    
    # Test query with validation
    test_query = "What is machine learning?"
    print(f"Testing query: {test_query}")
    
    # This will use all enterprise features:
    # - Input validation
    # - Rate limiting check
    # - Cache lookup
    # - Error handling
    # - Conversation tracking
    
    print("Query would be processed with all enterprise features enabled.")
    print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Enterprise Features Summary:")
    print("1. Conversation Management: Tracks multi-turn conversations")
    print("2. Response Caching: Improves performance for repeated queries")
    print("3. Error Handling: Graceful error handling with user-friendly messages")
    print("4. Rate Limiting: Prevents API abuse and manages costs")
    print("5. Input Validation: Ensures security and data integrity")


if __name__ == "__main__":
    main()

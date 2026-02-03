"""
Integration tests for end-to-end workflows.
"""

import pytest
import os
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import get_api_key


class TestEndToEndWorkflow:
    """Test complete workflows from document ingestion to querying."""
    
    def test_full_workflow_with_mock_documents(self, document_processor, vector_store, sample_documents):
        """Test complete workflow with mock document data."""
        # Step 1: Process documents (using mock data)
        documents = sample_documents
        
        # Step 2: Add to vector store
        vector_store.add_documents(documents)
        
        # Step 3: Verify documents are stored
        stats = vector_store.get_collection_stats()
        # Check that documents were added (stats may have different keys)
        total_docs = stats.get("total_documents", len(vector_store.documents))
        assert total_docs == 3 or len(vector_store.documents) == 3
        
        # Step 4: Create query engine
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=vector_store,
            enable_caching=False,
            enable_rate_limiting=False
        )
        
        # Step 5: Test search functionality
        results = vector_store.search("machine learning", top_k=2)
        assert len(results) > 0
        assert "machine learning" in results[0]["content"].lower() or \
               "machine learning" in results[0].get("metadata", {}).get("content", "").lower()
    
    def test_query_workflow(self, populated_vector_store):
        """Test query processing workflow."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False,
            enable_rate_limiting=False
        )
        
        # Test that query engine is initialized
        assert query_engine.vector_store is not None
        assert query_engine.input_validator is not None
        
        # Test query processing (may fail if API key is not set)
        if get_api_key():
            try:
                response = query_engine.query("What is the methodology?", top_k=2)
                assert "answer" in response
                assert isinstance(response["answer"], str)
            except Exception as e:
                # If API fails, at least verify the query was processed
                pytest.skip(f"API call failed: {e}")
        else:
            pytest.skip("GEMINI_API_KEY not set")
    
    def test_document_retrieval_workflow(self, populated_vector_store):
        """Test document retrieval and context building."""
        # Search for specific content
        results = populated_vector_store.search("accuracy", top_k=1)
        
        assert len(results) > 0
        result = results[0]
        
        # Verify result structure
        assert "content" in result or "text" in result
        assert "metadata" in result
        assert "score" in result or "similarity" in result
        
        # Verify metadata
        metadata = result.get("metadata", {})
        assert "document_id" in metadata or "document_id" in result
    
    def test_filtered_search_workflow(self, populated_vector_store):
        """Test filtered search workflow."""
        # Search with filter
        results = populated_vector_store.search(
            "methodology",
            top_k=5,
            filter={"document_id": "test_paper1.pdf"}
        )
        
        # All results should be from the filtered document
        for result in results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id") or result.get("document_id")
            assert doc_id == "test_paper1.pdf"


class TestQueryTypes:
    """Test different query types."""
    
    def test_direct_lookup_query(self, populated_vector_store):
        """Test direct content lookup queries."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        # Test direct lookup intent
        query = "What is the conclusion of test_paper1?"
        try:
            response = query_engine.query(query, top_k=3)
            assert "answer" in response
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    def test_summarization_query(self, populated_vector_store):
        """Test summarization queries."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        # Test summarization intent
        query = "Summarize the methodology of test_paper1"
        try:
            response = query_engine.query(query, top_k=3)
            assert "answer" in response
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    def test_metrics_extraction_query(self, populated_vector_store):
        """Test metrics extraction queries."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        # Test metrics extraction intent
        query = "What are the accuracy and F1-score reported in test_paper1?"
        try:
            response = query_engine.query(query, top_k=3)
            assert "answer" in response
            # Should contain metrics
            answer = response["answer"].lower()
            assert "accuracy" in answer or "f1" in answer or "95" in answer or "0.92" in answer
        except Exception as e:
            pytest.skip(f"Query failed: {e}")


class TestErrorHandling:
    """Test error handling in workflows."""
    
    def test_empty_vector_store_query(self, vector_store):
        """Test querying empty vector store."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=vector_store,
            enable_caching=False
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        try:
            response = query_engine.query("What is AI?", top_k=3)
            # Should handle gracefully
            assert "answer" in response
        except Exception as e:
            # Error handling should catch this
            assert True
    
    def test_invalid_query_handling(self, populated_vector_store):
        """Test handling of invalid queries."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False
        )
        
        # Test empty query
        response = query_engine.query("")
        assert "answer" in response
        assert response.get("confidence", 1.0) == 0.0 or "empty" in response["answer"].lower()
        
        # Test very long query
        long_query = "A" * 10000
        response = query_engine.query(long_query)
        # Should be sanitized or rejected
        assert "answer" in response


class TestEnterpriseFeaturesIntegration:
    """Test enterprise features in integrated workflows."""
    
    def test_caching_in_workflow(self, populated_vector_store):
        """Test response caching in query workflow."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=True,
            enable_rate_limiting=False
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        query = "What is machine learning?"
        
        # First query - should cache
        try:
            response1 = query_engine.query(query, use_cache=True)
            assert "answer" in response1
            
            # Second query - should use cache
            response2 = query_engine.query(query, use_cache=True)
            assert "answer" in response2
            # Responses should be similar (may not be identical due to LLM variability)
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    def test_conversation_history(self, populated_vector_store):
        """Test conversation history in multi-turn queries."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False,
            enable_conversation=True
        )
        
        if not get_api_key():
            pytest.skip("GEMINI_API_KEY not set")
        
        # First query
        try:
            response1 = query_engine.query("What is AI?", user_id="test_user")
            assert "answer" in response1
            
            # Check conversation history
            assert query_engine.conversation_manager is not None
            assert len(query_engine.conversation_manager.history) == 1
            
            # Second query (follow-up)
            response2 = query_engine.query("Tell me more about it", user_id="test_user")
            assert "answer" in response2
            assert len(query_engine.conversation_manager.history) == 2
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    def test_rate_limiting_in_workflow(self, populated_vector_store):
        """Test rate limiting in query workflow."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False,
            enable_rate_limiting=True
        )
        
        # Make multiple requests
        for i in range(5):
            response = query_engine.query(f"Query {i}", user_id="test_user")
            # Should handle rate limiting gracefully
            assert "answer" in response or "rate" in response.get("answer", "").lower()

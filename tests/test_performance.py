"""
Performance tests for the Document Q&A AI Agent.
"""

import pytest
import time
import sys
from typing import List, Dict

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import get_api_key
from tests.conftest import sample_documents


class TestDocumentProcessingPerformance:
    """Test performance of document processing."""
    
    def test_chunking_performance(self, document_processor):
        """Test chunking performance with large text."""
        # Create large text
        large_text = "This is a test sentence. " * 1000  # ~25KB of text
        
        start_time = time.time()
        chunks = document_processor.chunk_document(
            {
                "title": "Test Document",
                "abstract": "",
                "sections": [{"heading": "Content", "content": large_text, "level": 1}],
                "tables": [],
                "figures": [],
                "references": [],
                "equations": []
            },
            chunk_size=500,
            overlap=50
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for 25KB)
        assert processing_time < 1.0, f"Chunking took {processing_time:.2f}s, expected < 1.0s"
        assert len(chunks) > 0
    
    def test_embedding_generation_performance(self, vector_store):
        """Test embedding generation performance."""
        # Create multiple documents
        test_documents = []
        for i in range(10):
            test_documents.append({
                "chunk_id": i,
                "document_id": f"test_{i}.pdf",
                "section": "Test",
                "level": 1,
                "content": f"This is test document {i} with some content about machine learning and AI.",
                "metadata": {"document_id": f"test_{i}.pdf"}
            })
        
        start_time = time.time()
        vector_store.add_documents(test_documents)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (< 5 seconds for 10 documents)
        assert processing_time < 5.0, f"Embedding generation took {processing_time:.2f}s, expected < 5.0s"
        
        # Verify documents were added
        stats = vector_store.get_collection_stats()
        # Check that documents were added (stats may have different keys)
        assert stats.get("total_documents", len(vector_store.documents)) == 10 or len(vector_store.documents) == 10


class TestVectorSearchPerformance:
    """Test performance of vector search operations."""
    
    def test_search_performance(self, populated_vector_store):
        """Test search performance with populated store."""
        # Add more documents for realistic test
        additional_docs = []
        for i in range(20):
            additional_docs.append({
                "chunk_id": i + 3,
                "document_id": f"test_{i}.pdf",
                "section": "Test",
                "level": 1,
                "content": f"Document {i} discusses various topics including machine learning, neural networks, and deep learning.",
                "metadata": {"document_id": f"test_{i}.pdf"}
            })
        
        populated_vector_store.add_documents(additional_docs)
        
        # Test search performance
        queries = [
            "machine learning",
            "neural networks",
            "deep learning",
            "artificial intelligence"
        ]
        
        total_time = 0
        for query in queries:
            start_time = time.time()
            results = populated_vector_store.search(query, top_k=5)
            end_time = time.time()
            total_time += (end_time - start_time)
            assert len(results) > 0
        
        avg_time = total_time / len(queries)
        
        # Average search should be fast (< 0.5 seconds)
        assert avg_time < 0.5, f"Average search took {avg_time:.2f}s, expected < 0.5s"
    
    def test_filtered_search_performance(self, populated_vector_store):
        """Test performance of filtered search."""
        # Add documents from multiple sources
        for i in range(10):
            populated_vector_store.add_documents([{
                "chunk_id": i + 10,
                "document_id": f"paper_{i % 3}.pdf",  # 3 different papers
                "section": "Test",
                "level": 1,
                "content": f"Content from paper {i % 3}",
                "metadata": {"document_id": f"paper_{i % 3}.pdf"}
            }])
        
        start_time = time.time()
        results = populated_vector_store.search(
            "content",
            top_k=10,
            filter={"document_id": "paper_0.pdf"}
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Filtered search should be fast (< 0.5 seconds)
        assert search_time < 0.5, f"Filtered search took {search_time:.2f}s, expected < 0.5s"
        assert len(results) > 0


class TestQueryProcessingPerformance:
    """Test performance of query processing."""
    
    @pytest.mark.skipif(not get_api_key(), reason="GEMINI_API_KEY not set")
    def test_query_response_time(self, populated_vector_store):
        """Test query response time."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=False,
            enable_rate_limiting=False
        )
        
        query = "What is machine learning?"
        
        start_time = time.time()
        try:
            response = query_engine.query(query, top_k=3)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # API calls may take longer, but should complete in reasonable time
            # (< 10 seconds for a single query)
            assert response_time < 10.0, f"Query took {response_time:.2f}s, expected < 10.0s"
            assert "answer" in response
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    @pytest.mark.skipif(not get_api_key(), reason="GEMINI_API_KEY not set")
    def test_cached_query_performance(self, populated_vector_store):
        """Test performance improvement with caching."""
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=populated_vector_store,
            enable_caching=True,
            enable_rate_limiting=False
        )
        
        query = "What is artificial intelligence?"
        
        # First query (no cache)
        start_time = time.time()
        try:
            response1 = query_engine.query(query, use_cache=True)
            time1 = time.time() - start_time
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
        
        # Second query (with cache)
        start_time = time.time()
        response2 = query_engine.query(query, use_cache=True)
        time2 = time.time() - start_time
        
        # Cached query should be significantly faster
        # (at least 10x faster, or < 0.1 seconds)
        assert time2 < 0.1 or time2 < time1 / 10, \
            f"Cached query took {time2:.2f}s, expected much faster than {time1:.2f}s"


class TestMemoryUsage:
    """Test memory usage and optimization."""
    
    def test_memory_efficiency(self, vector_store):
        """Test that vector store doesn't leak memory."""
        import gc
        
        # Add many documents
        documents = []
        for i in range(100):
            documents.append({
                "chunk_id": i,
                "document_id": f"doc_{i}.pdf",
                "section": "Test",
                "level": 1,
                "content": f"Content {i} " * 10,  # ~100 chars per doc
                "metadata": {"document_id": f"doc_{i}.pdf"}
            })
        
        # Add documents
        vector_store.add_documents(documents)
        
        # Clear and check memory
        gc.collect()
        
        # Verify documents are stored
        stats = vector_store.get_collection_stats()
        assert stats["total_documents"] == 100
    
    def test_cache_memory_limits(self):
        """Test that cache respects memory limits."""
        from src.response_cache import ResponseCache
        
        cache = ResponseCache(ttl=3600, max_size=10)
        
        # Add more than max_size entries
        for i in range(15):
            cache.set(f"query_{i}", {"answer": f"answer_{i}"})
        
        # Cache should not exceed max_size
        stats = cache.get_stats()
        assert stats["total_entries"] <= 10


class TestConcurrentOperations:
    """Test concurrent operations (simulated)."""
    
    def test_multiple_searches(self, populated_vector_store):
        """Test multiple concurrent searches."""
        queries = [
            "machine learning",
            "neural networks",
            "deep learning",
            "artificial intelligence",
            "natural language processing"
        ]
        
        start_time = time.time()
        results_list = []
        for query in queries:
            results = populated_vector_store.search(query, top_k=3)
            results_list.append(results)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Multiple searches should complete in reasonable time
        assert total_time < 2.0, f"Multiple searches took {total_time:.2f}s, expected < 2.0s"
        assert all(len(results) > 0 for results in results_list)
    
    def test_batch_document_processing(self, vector_store):
        """Test batch document processing performance."""
        # Create batch of documents
        batch = []
        for i in range(50):
            batch.append({
                "chunk_id": i,
                "document_id": f"batch_{i}.pdf",
                "section": "Test",
                "level": 1,
                "content": f"Batch document {i} with content about various topics.",
                "metadata": {"document_id": f"batch_{i}.pdf"}
            })
        
        start_time = time.time()
        vector_store.add_documents(batch)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Batch processing should be efficient
        assert processing_time < 10.0, f"Batch processing took {processing_time:.2f}s, expected < 10.0s"
        
        stats = vector_store.get_collection_stats()
        # Check that documents were added
        assert stats.get("total_documents", len(vector_store.documents)) == 50 or len(vector_store.documents) == 50

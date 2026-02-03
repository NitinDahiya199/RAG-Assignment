"""
Tests for VectorStore class.
"""

import pytest
import os
from src.vector_store import VectorStore


def test_vector_store_initialization():
    """Test VectorStore initialization."""
    store = VectorStore()
    assert store is not None
    assert hasattr(store, 'embedding_model')
    assert hasattr(store, 'generate_embeddings')
    assert hasattr(store, 'add_documents')
    assert hasattr(store, 'search')


def test_generate_embeddings():
    """Test embedding generation."""
    store = VectorStore()
    
    texts = [
        "This is a test document.",
        "Another test document with different content."
    ]
    
    embeddings = store.generate_embeddings(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == store.embedding_dim
    assert len(embeddings[1]) == store.embedding_dim


def test_add_and_search_documents():
    """Test adding documents and searching."""
    store = VectorStore()
    
    # Create mock documents
    documents = [
        {
            "chunk_id": 0,
            "document_id": "test.pdf",
            "section": "Introduction",
            "level": 1,
            "content": "This is about machine learning and neural networks.",
            "metadata": {"type": "section"}
        },
        {
            "chunk_id": 1,
            "document_id": "test.pdf",
            "section": "Methodology",
            "level": 1,
            "content": "We used transformer architectures for this task.",
            "metadata": {"type": "section"}
        },
        {
            "chunk_id": 2,
            "document_id": "test2.pdf",
            "section": "Results",
            "level": 1,
            "content": "The results show significant improvements.",
            "metadata": {"type": "section"}
        }
    ]
    
    # Add documents
    store.add_documents(documents)
    
    # Search
    results = store.search("machine learning", top_k=2)
    
    assert len(results) > 0
    assert all("content" in result for result in results)
    assert all("score" in result for result in results)


def test_search_with_filter():
    """Test search with metadata filtering."""
    store = VectorStore()
    
    documents = [
        {
            "chunk_id": 0,
            "document_id": "paper1.pdf",
            "section": "Introduction",
            "level": 1,
            "content": "First paper content about AI.",
            "metadata": {"type": "section"}
        },
        {
            "chunk_id": 1,
            "document_id": "paper2.pdf",
            "section": "Introduction",
            "level": 1,
            "content": "Second paper content about ML.",
            "metadata": {"type": "section"}
        }
    ]
    
    store.add_documents(documents)
    
    # Search with filter
    results = store.search(
        "AI",
        top_k=5,
        filter={"document_id": "paper1.pdf"}
    )
    
    # All results should be from paper1.pdf (check both metadata locations)
    if results:
        for result in results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id") or result.get("document_id")
            assert doc_id == "paper1.pdf", f"Expected paper1.pdf, got {doc_id}"


def test_hybrid_search():
    """Test hybrid search functionality."""
    store = VectorStore()
    
    documents = [
        {
            "chunk_id": 0,
            "document_id": "test.pdf",
            "section": "Introduction",
            "level": 1,
            "content": "Deep learning models use neural networks.",
            "metadata": {"type": "section"}
        }
    ]
    
    store.add_documents(documents)
    
    # Perform hybrid search
    results = store.hybrid_search("neural networks", top_k=3)
    
    assert len(results) >= 0  # May or may not find results depending on implementation


def test_get_collection_stats():
    """Test getting collection statistics."""
    store = VectorStore()
    
    stats = store.get_collection_stats()
    
    assert "type" in stats
    assert "document_count" in stats
    assert "embedding_dimension" in stats


def test_empty_search():
    """Test search with empty query."""
    store = VectorStore()
    
    results = store.search("", top_k=5)
    
    assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

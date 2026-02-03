"""
Tests for QueryEngine class.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.query_engine import QueryEngine


def test_query_engine_initialization():
    """Test QueryEngine initialization."""
    engine = QueryEngine()
    assert engine is not None
    assert hasattr(engine, 'query')
    assert hasattr(engine, 'direct_lookup')
    assert hasattr(engine, 'summarize')
    assert hasattr(engine, 'extract_metrics')


def test_classify_intent():
    """Test intent classification."""
    engine = QueryEngine()
    
    # Test direct lookup
    assert engine._classify_intent("What is the conclusion of Paper X?") == "direct_lookup"
    
    # Test summarization
    assert engine._classify_intent("Summarize the methodology") == "summarization"
    
    # Test metrics extraction
    assert engine._classify_intent("What is the accuracy and F1-score?") == "metrics_extraction"
    
    # Test general query
    assert engine._classify_intent("What is machine learning?") == "general"


def test_extract_metric_names():
    """Test metric name extraction."""
    engine = QueryEngine()
    
    metrics = engine._extract_metric_names("What is the accuracy and F1-score?")
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    
    metrics = engine._extract_metric_names("Show me precision and recall")
    assert "precision" in metrics
    assert "recall" in metrics


def test_build_context_string():
    """Test context string building."""
    engine = QueryEngine()
    
    context = [
        {
            "content": "This is test content 1",
            "metadata": {"document_id": "test.pdf", "section": "Introduction"}
        },
        {
            "content": "This is test content 2",
            "metadata": {"document_id": "test.pdf", "section": "Methodology"}
        }
    ]
    
    context_str = engine._build_context_string(context)
    assert "test.pdf" in context_str
    assert "Introduction" in context_str
    assert "Methodology" in context_str
    assert "test content 1" in context_str


def test_calculate_confidence():
    """Test confidence calculation."""
    engine = QueryEngine()
    
    # High confidence context
    context = [
        {"score": 0.9},
        {"score": 0.85},
        {"score": 0.8}
    ]
    confidence = engine._calculate_confidence(context, "general")
    assert confidence > 0.8
    
    # Low confidence context
    context = [
        {"score": 0.3},
        {"score": 0.2}
    ]
    confidence = engine._calculate_confidence(context, "general")
    assert confidence < 0.5
    
    # Empty context
    confidence = engine._calculate_confidence([], "general")
    assert confidence == 0.0


def test_format_response():
    """Test response formatting."""
    engine = QueryEngine()
    
    answer = "This is a test answer"
    context = [
        {
            "content": "Test content",
            "metadata": {"document_id": "test.pdf", "section": "Introduction"},
            "score": 0.8
        }
    ]
    
    response = engine._format_response(answer, context, "general")
    
    assert response["answer"] == answer
    assert len(response["sources"]) > 0
    assert "confidence" in response
    assert response["intent"] == "general"


def test_query_without_model():
    """Test query when model is not available."""
    engine = QueryEngine(api_key=None)
    
    # Mock vector store
    mock_vector_store = Mock()
    mock_vector_store.search.return_value = [
        {
            "content": "Test content",
            "metadata": {"document_id": "test.pdf", "section": "Introduction"},
            "score": 0.8
        }
    ]
    engine.vector_store = mock_vector_store
    
    response = engine.query("What is this?")
    
    # Should still return a response structure
    assert "answer" in response
    assert "sources" in response


def test_direct_lookup():
    """Test direct lookup functionality."""
    engine = QueryEngine()
    
    # Mock vector store
    mock_vector_store = Mock()
    mock_vector_store.search.return_value = [
        {
            "content": "The conclusion is that the method works well.",
            "metadata": {"document_id": "paper_x.pdf", "section": "Conclusion"},
            "score": 0.9
        }
    ]
    engine.vector_store = mock_vector_store
    
    response = engine.direct_lookup("What is the conclusion?", "paper_x.pdf")
    
    assert "answer" in response
    assert "sources" in response


def test_summarize():
    """Test summarization functionality."""
    engine = QueryEngine()
    
    # Mock vector store
    mock_vector_store = Mock()
    mock_vector_store.search.return_value = [
        {
            "content": "Methodology content here...",
            "metadata": {"document_id": "paper_c.pdf", "section": "Methodology"},
            "score": 0.85
        }
    ]
    engine.vector_store = mock_vector_store
    
    response = engine.summarize("Summarize the methodology", "paper_c.pdf")
    
    assert "answer" in response
    assert "sources" in response


def test_extract_metrics():
    """Test metrics extraction functionality."""
    engine = QueryEngine()
    
    # Mock vector store
    mock_vector_store = Mock()
    mock_vector_store.search.return_value = [
        {
            "content": "The accuracy was 95% and F1-score was 0.92.",
            "metadata": {"document_id": "paper_d.pdf", "section": "Results"},
            "score": 0.9
        }
    ]
    engine.vector_store = mock_vector_store
    
    response = engine.extract_metrics("What are the accuracy and F1-score?", "paper_d.pdf")
    
    assert "answer" in response
    assert "metrics" in response
    assert isinstance(response["metrics"], dict)


def test_parse_metrics_json():
    """Test JSON metrics parsing."""
    engine = QueryEngine()
    
    # Test with valid JSON
    text = '{"accuracy": 0.95, "f1_score": 0.92}'
    metrics = engine._parse_metrics_json(text)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.95
    
    # Test with text containing JSON
    text = 'The results are: {"accuracy": 0.95, "f1_score": 0.92}'
    metrics = engine._parse_metrics_json(text)
    assert "accuracy" in metrics
    
    # Test with pattern matching fallback
    text = "Accuracy: 0.95, F1-score: 0.92"
    metrics = engine._parse_metrics_json(text)
    assert len(metrics) > 0


def test_empty_query():
    """Test handling of empty query."""
    engine = QueryEngine()
    
    response = engine.query("")
    
    assert "answer" in response
    assert "please provide" in response["answer"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

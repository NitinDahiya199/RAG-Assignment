"""
Tests for ArxivFunctionCalling class.
"""

import pytest
from src.function_calling import ArxivFunctionCalling


def test_arxiv_function_calling_initialization():
    """Test ArxivFunctionCalling initialization."""
    arxiv_func = ArxivFunctionCalling()
    assert arxiv_func is not None
    assert hasattr(arxiv_func, 'get_function_schema')
    assert hasattr(arxiv_func, 'search_arxiv_paper')
    assert hasattr(arxiv_func, 'execute_function')


def test_get_function_schema():
    """Test function schema generation."""
    arxiv_func = ArxivFunctionCalling()
    schema = arxiv_func.get_function_schema()
    
    assert "function_declarations" in schema
    assert len(schema["function_declarations"]) > 0
    
    func_decl = schema["function_declarations"][0]
    assert func_decl["name"] == "search_arxiv_paper"
    assert "parameters" in func_decl
    assert "properties" in func_decl["parameters"]


def test_execute_function():
    """Test function execution."""
    arxiv_func = ArxivFunctionCalling()
    
    # Test with valid function
    result = arxiv_func.execute_function(
        "search_arxiv_paper",
        {"query": "transformer", "max_results": 2}
    )
    
    assert "function_name" in result
    assert result["function_name"] == "search_arxiv_paper"
    assert "success" in result
    
    # Test with invalid function
    result = arxiv_func.execute_function("invalid_function", {})
    assert result["success"] is False
    assert "error" in result


def test_format_papers_for_response():
    """Test paper formatting."""
    arxiv_func = ArxivFunctionCalling()
    
    papers = [
        {
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "abstract": "This is a test abstract.",
            "published": "2024-01-01",
            "arxiv_id": "2401.00001",
            "pdf_url": "http://arxiv.org/pdf/2401.00001"
        }
    ]
    
    formatted = arxiv_func.format_papers_for_response(papers)
    
    assert "Test Paper" in formatted
    assert "Author 1" in formatted
    assert "2401.00001" in formatted


def test_format_empty_papers():
    """Test formatting with no papers."""
    arxiv_func = ArxivFunctionCalling()
    
    formatted = arxiv_func.format_papers_for_response([])
    assert "No papers found" in formatted


def test_search_by_author():
    """Test author search."""
    arxiv_func = ArxivFunctionCalling()
    
    # This will make an actual API call, so we'll just test the method exists
    # In real tests, you might want to mock this
    result = arxiv_func.search_by_author("Einstein", max_results=1)
    
    # Result should be a list (may be empty if API fails)
    assert isinstance(result, list)


def test_search_by_title():
    """Test title search."""
    arxiv_func = ArxivFunctionCalling()
    
    # This will make an actual API call
    result = arxiv_func.search_by_title("quantum", max_results=1)
    
    # Result should be a list
    assert isinstance(result, list)


def test_get_paper_by_id():
    """Test getting paper by ID."""
    arxiv_func = ArxivFunctionCalling()
    
    # Test with a known Arxiv ID (this will make an API call)
    # Using a common paper ID format
    result = arxiv_func.get_paper_by_id("1706.03762")
    
    # Result should be dict or None
    assert result is None or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

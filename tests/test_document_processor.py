"""
Tests for DocumentProcessor class.
"""

import pytest
import os
from pathlib import Path
from src.document_processor import DocumentProcessor


def test_document_processor_initialization():
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor()
    assert processor is not None
    assert hasattr(processor, 'process_pdf')
    assert hasattr(processor, 'extract_structure')


def test_process_nonexistent_pdf():
    """Test processing a non-existent PDF file."""
    processor = DocumentProcessor()
    
    with pytest.raises(FileNotFoundError):
        processor.process_pdf("nonexistent_file.pdf")


def test_chunk_document():
    """Test document chunking functionality."""
    processor = DocumentProcessor()
    
    # Create a mock document
    mock_document = {
        "file_name": "test.pdf",
        "abstract": "This is a test abstract. " * 10,
        "sections": [
            {
                "heading": "Introduction",
                "level": 1,
                "content": "This is the introduction content. " * 50
            },
            {
                "heading": "Methodology",
                "level": 1,
                "content": "This is methodology content. " * 30
            }
        ]
    }
    
    chunks = processor.chunk_document(mock_document, chunk_size=200, overlap=50)
    
    assert len(chunks) > 0
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("content" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)


def test_extract_sections():
    """Test section extraction from text."""
    processor = DocumentProcessor()
    
    test_text = """
    1. Introduction
    This is the introduction content.
    
    2. Methodology
    This is methodology content.
    
    2.1 Data Collection
    Data collection details here.
    """
    
    sections = processor._extract_sections(test_text, None)
    
    assert len(sections) > 0
    assert any(s["heading"] == "Introduction" for s in sections)


def test_extract_abstract():
    """Test abstract extraction."""
    processor = DocumentProcessor()
    
    test_text = """
    Abstract
    
    This is the abstract content. It describes the main findings
    and contributions of the paper. The abstract should be informative.
    
    1. Introduction
    """
    
    abstract = processor._extract_abstract(test_text)
    assert len(abstract) > 0
    assert "abstract content" in abstract.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

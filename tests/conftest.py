"""
Pytest configuration and fixtures for testing.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import setup_logging

# Setup logging for tests
setup_logging("WARNING")  # Reduce noise in test output


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file path (placeholder)."""
    pdf_path = temp_dir / "sample.pdf"
    # Note: In real tests, you would create an actual PDF file
    # For now, this is a placeholder
    return pdf_path


@pytest.fixture
def document_processor() -> DocumentProcessor:
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor()


@pytest.fixture
def vector_store() -> VectorStore:
    """Create a VectorStore instance for testing."""
    return VectorStore()


@pytest.fixture
def query_engine(vector_store: VectorStore) -> QueryEngine:
    """Create a QueryEngine instance for testing."""
    # Use a test API key or None for testing
    api_key = os.getenv("GEMINI_API_KEY")
    return QueryEngine(
        api_key=api_key,
        vector_store=vector_store,
        enable_caching=False,  # Disable caching for tests
        enable_rate_limiting=False,  # Disable rate limiting for tests
        enable_conversation=False  # Disable conversation for tests
    )


@pytest.fixture
def sample_documents() -> list[dict]:
    """Sample document chunks for testing."""
    return [
        {
            "chunk_id": 0,
            "document_id": "test_paper1.pdf",
            "section": "Introduction",
            "level": 1,
            "content": "This paper introduces a novel approach to machine learning.",
            "metadata": {
                "type": "section",
                "document_id": "test_paper1.pdf",
                "page": 1
            }
        },
        {
            "chunk_id": 1,
            "document_id": "test_paper1.pdf",
            "section": "Methodology",
            "level": 1,
            "content": "Our methodology uses deep neural networks with attention mechanisms.",
            "metadata": {
                "type": "section",
                "document_id": "test_paper1.pdf",
                "page": 2
            }
        },
        {
            "chunk_id": 2,
            "document_id": "test_paper1.pdf",
            "section": "Results",
            "level": 1,
            "content": "We achieved 95% accuracy and 0.92 F1-score on the test dataset.",
            "metadata": {
                "type": "section",
                "document_id": "test_paper1.pdf",
                "page": 3
            }
        }
    ]


@pytest.fixture
def populated_vector_store(vector_store: VectorStore, sample_documents: list[dict]) -> VectorStore:
    """Create a VectorStore with sample documents."""
    vector_store.add_documents(sample_documents)
    return vector_store

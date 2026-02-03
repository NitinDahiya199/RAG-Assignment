# 📚 Document Q&A AI Agent - Enterprise-Ready Prototype

A lightweight AI agent built with Google Gemini API that ingests PDF documents, extracts structured content, and answers questions about the documents with enterprise-grade features including context management, error handling, response optimization, and security best practices.

## ✨ Features

### Core Features
- **Multi-Document Processing**: Handle multiple PDF documents simultaneously with batch processing
- **Structured Content Extraction**: Extract titles, abstracts, sections, tables, figures, references, and equations
- **Intelligent Query Interface**: 
  - Direct content lookup (e.g., "What is the conclusion of Paper X?")
  - Summarization (e.g., "Summarize the methodology of Paper C")
  - Evaluation metrics extraction (e.g., "What are the accuracy and F1-score reported in Paper D?")
- **Arxiv Integration** (Bonus): Function calling capability to search for papers on Arxiv based on user descriptions

### Enterprise Features
- **Context Management**: Multi-turn conversation support with history tracking
- **Response Caching**: Intelligent caching system to improve performance for repeated queries
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Rate Limiting**: API rate limiting to prevent abuse and manage costs
- **Input Validation**: Security-focused input validation and sanitization
- **Logging**: Comprehensive logging system for debugging and monitoring

## 🛠️ Tech Stack

- **LLM API**: Google Gemini API (gemini-1.5-flash)
- **PDF Processing**: PyPDF, pdfplumber
- **Vector Store**: ChromaDB (with in-memory fallback)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Language**: Python 3.8+
- **Testing**: pytest, pytest-cov

## 📋 Prerequisites

- Python 3.8 or higher (tested with Python 3.14)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- pip (Python package manager)
- Git (for cloning the repository)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG-Assignment
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with ChromaDB on Python 3.14, the system will automatically fall back to an in-memory vector store.

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
# Then edit .env with your preferred editor
```

## 📖 Usage

### Basic Usage

```python
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import setup_logging

# Setup logging
setup_logging("INFO")

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()
query_engine = QueryEngine(
    api_key="your_api_key",  # Or set via GEMINI_API_KEY env var
    vector_store=vector_store,
    enable_caching=True,
    enable_rate_limiting=True,
    enable_conversation=True
)

# Process a PDF document
documents = processor.process_pdf("data/pdfs/paper1.pdf")

# Add to vector store
chunks = []
for doc in documents:
    chunks.extend(doc.get("chunks", []))

vector_store.add_documents(chunks)

# Query the documents
response = query_engine.query("What is the main contribution of this paper?")
print(response["answer"])
print(f"Confidence: {response['confidence']}")
print(f"Sources: {response['sources']}")
```

### Advanced Usage with Enterprise Features

```python
from src.query_engine import QueryEngine
from src.vector_store import VectorStore

# Initialize with all enterprise features
vector_store = VectorStore()
query_engine = QueryEngine(
    vector_store=vector_store,
    enable_caching=True,        # Enable response caching
    enable_rate_limiting=True,  # Enable rate limiting
    enable_conversation=True    # Enable conversation history
)

# Multi-turn conversation
response1 = query_engine.query("What is machine learning?", user_id="user123")
response2 = query_engine.query("Tell me more about it", user_id="user123")  # Uses context from previous query

# Check cache statistics
cache_stats = query_engine.cache.get_stats()
print(f"Cache entries: {cache_stats['total_entries']}")

# Check rate limit status
rate_limit_stats = query_engine.rate_limiter.get_stats("user123")
print(f"Remaining requests: {rate_limit_stats['remaining']}")
```

### Query Examples

#### 1. Direct Content Lookup
```python
response = query_engine.query("What is the conclusion of Paper X?")
```

#### 2. Summarization
```python
response = query_engine.query("Summarize the methodology of Paper C")
```

#### 3. Metrics Extraction
```python
response = query_engine.query("What are the accuracy and F1-score reported in Paper D?")
# Response includes structured JSON with extracted metrics
```

#### 4. Arxiv Search (Bonus Feature)
```python
response = query_engine.query("Find papers about transformer architectures published in 2023")
# The system will automatically use Arxiv function calling if needed
```

### Processing Multiple Documents

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Process multiple PDFs
pdf_files = ["data/pdfs/paper1.pdf", "data/pdfs/paper2.pdf", "data/pdfs/paper3.pdf"]
all_documents = processor.process_multiple_pdfs(pdf_files)

# Add all chunks to vector store
all_chunks = []
for doc in all_documents:
    all_chunks.extend(doc.get("chunks", []))

vector_store.add_documents(all_chunks)
```

## 📁 Project Structure

```
RAG-Assignment/
├── src/                          # Source code
│   ├── __init__.py
│   ├── document_processor.py     # PDF ingestion & extraction
│   ├── vector_store.py           # Embedding & retrieval
│   ├── query_engine.py           # Query processing & LLM interaction
│   ├── function_calling.py       # Arxiv API integration (Bonus)
│   ├── conversation_manager.py   # Conversation history management
│   ├── response_cache.py         # Response caching
│   ├── error_handler.py          # Error handling
│   ├── rate_limiter.py           # Rate limiting
│   ├── input_validator.py        # Input validation
│   ├── utils.py                  # Helper functions
│   └── main.py                   # Main entry point
├── data/
│   └── pdfs/                     # Input PDF documents (add your PDFs here)
├── outputs/
│   └── processed/                # Processed document chunks
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   ├── test_query_engine.py
│   ├── test_function_calling.py
│   ├── test_enterprise_features.py
│   ├── test_integration.py      # Integration tests
│   └── test_performance.py      # Performance tests
├── docs/                         # Documentation
│   ├── PHASE3_VECTOR_STORE.md
│   ├── PHASE4_QUERY_INTERFACE.md
│   ├── PHASE5_FUNCTION_CALLING.md
│   ├── PHASE6_ENTERPRISE_FEATURES.md
│   └── PHASE7_TESTING.md
├── example_*.py                  # Example usage scripts
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                     # This file
└── roadmap.md                   # Development roadmap
```

## 🔌 API Reference

### DocumentProcessor

```python
class DocumentProcessor:
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file and extract structured content."""
        
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """Process multiple PDF files in batch."""
        
    def chunk_document(self, document: Dict, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Chunk document into smaller pieces for vector storage."""
```

### VectorStore

```python
class VectorStore:
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store with embeddings."""
        
    def search(self, query: str, top_k: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents using semantic similarity."""
        
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store."""
```

### QueryEngine

```python
class QueryEngine:
    def query(
        self,
        question: str,
        context: Optional[List[Dict]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict:
        """Process a user query and generate a response."""
        
    def direct_lookup(self, question: str, document_id: str) -> Dict:
        """Direct lookup in a specific document."""
        
    def summarize(self, question: str, document_id: Optional[str] = None) -> Dict:
        """Summarize content from documents."""
        
    def extract_metrics(self, question: str, document_id: Optional[str] = None) -> Dict:
        """Extract evaluation metrics from documents."""
```

### Enterprise Features

```python
# Conversation Manager
conversation_manager = ConversationManager(max_history=10)
conversation_manager.add_turn(query, response)
context = conversation_manager.get_context()

# Response Cache
cache = ResponseCache(ttl=3600, max_size=1000)
cached_response = cache.get(query)
cache.set(query, response)

# Rate Limiter
rate_limiter = RateLimiter(max_requests=60, time_window=60)
is_allowed, message = rate_limiter.is_allowed(user_id="user123")

# Error Handler
error_handler = ErrorHandler()
error_response = error_handler.handle_error(exception, error_type)

# Input Validator
validator = InputValidator()
is_valid, error = validator.validate_query(query)
sanitized = validator.sanitize_query(query)
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests
pytest tests/test_document_processor.py -v
pytest tests/test_vector_store.py -v
pytest tests/test_query_engine.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v
```

### Test Results

- **57+ tests passing**
- **Comprehensive coverage** of all modules
- **Integration tests** for end-to-end workflows
- **Performance benchmarks** for optimization

## 🔒 Security

- **API Key Security**: API keys stored in `.env` file (not committed to git)
- **Input Validation**: Comprehensive input validation and sanitization
- **Error Handling**: Graceful error handling for API failures
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Path Validation**: File path validation to prevent directory traversal attacks
- **XSS Protection**: Input sanitization to prevent script injection

## 🐛 Troubleshooting

### Common Issues

#### 1. ChromaDB Installation Issues
**Problem**: `ModuleNotFoundError: No module named 'chromadb'`

**Solution**: The system automatically falls back to an in-memory vector store. For ChromaDB support:
```bash
pip install chromadb
```

#### 2. API Key Not Found
**Problem**: `GEMINI_API_KEY not found in environment variables`

**Solution**: 
1. Create a `.env` file in the project root
2. Add your API key: `GEMINI_API_KEY=your_key_here`
3. Ensure `python-dotenv` is installed

#### 3. PDF Processing Errors
**Problem**: `Error processing document`

**Solution**:
- Ensure PDF files are not corrupted
- Check file permissions
- Verify PDF format is supported
- Check logs for detailed error messages

#### 4. Embedding Model Download Issues
**Problem**: Slow or failed model downloads

**Solution**:
- Set `HF_TOKEN` environment variable for faster downloads
- Check internet connection
- Model will be cached after first download

#### 5. Rate Limiting Errors
**Problem**: `Rate limit exceeded`

**Solution**:
- Wait for the time window to reset
- Adjust rate limiter settings in `QueryEngine` initialization
- Use caching to reduce API calls

### Getting Help

1. Check the logs for detailed error messages
2. Review the documentation in `docs/` directory
3. Check the roadmap for development status
4. Review example scripts in `example_*.py` files

## 📈 Performance

### Benchmarks

- **Document Processing**: < 1 second per page
- **Embedding Generation**: < 0.5 seconds per document
- **Vector Search**: < 0.5 seconds average
- **Query Processing**: < 10 seconds (depends on API response time)
- **Cached Queries**: < 0.1 seconds

### Optimization Tips

1. **Enable Caching**: Reduces API calls for repeated queries
2. **Batch Processing**: Process multiple documents together
3. **Chunk Size Tuning**: Adjust chunk size based on document type
4. **Use Filters**: Filter searches by document ID for faster results

## 🚧 Future Improvements

### Planned Enhancements

1. **Persistent Vector Store**: Add support for persistent ChromaDB storage
2. **Advanced Reranking**: Implement more sophisticated reranking algorithms
3. **Multi-modal Support**: Enhanced support for images and tables in PDFs
4. **Web Interface**: Create a web UI for easier interaction
5. **API Server**: REST API server for integration with other systems
6. **Streaming Responses**: Support for streaming LLM responses
7. **Custom Embeddings**: Support for custom embedding models
8. **Document Versioning**: Track document versions and updates
9. **Advanced Analytics**: Query analytics and usage statistics
10. **Export Functionality**: Export processed documents and results

### Contributing

This is an assessment project. For questions or suggestions, please refer to the roadmap.

## 📄 License

This project is created for assessment purposes.

## 🙏 Acknowledgments

- **Google Gemini API** for LLM capabilities
- **Open source libraries** for PDF processing and vector storage:
  - PyPDF, pdfplumber for PDF processing
  - ChromaDB for vector storage
  - Sentence Transformers for embeddings
  - Arxiv API for paper search

## 📚 Additional Resources

- [Roadmap](roadmap.md) - Detailed development phases
- [Phase Documentation](docs/) - Technical documentation for each phase
- [Example Scripts](example_*.py) - Usage examples

---

**Note**: This is a prototype implementation. For production use, additional security, scalability, and performance optimizations would be required.

**Status**: ✅ All phases complete (1-7). Ready for demo preparation.

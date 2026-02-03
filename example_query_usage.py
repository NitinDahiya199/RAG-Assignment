"""
Example usage of QueryEngine for Phase 4.
This script demonstrates how to use the query interface functionality.
"""

import os
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import setup_logging, get_api_key

# Set up logging
setup_logging("INFO")


def main():
    """Example usage of QueryEngine."""
    
    print("=" * 60)
    print("Document Q&A AI Agent - Query Interface Demo")
    print("=" * 60)
    print()
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        print("⚠️  Warning: GEMINI_API_KEY not found in environment variables")
        print("   Query engine will have limited functionality")
        print("   Set your API key in .env file")
        print()
    
    # Initialize components
    print("Initializing components...")
    processor = DocumentProcessor()
    vector_store = VectorStore()
    query_engine = QueryEngine(api_key=api_key, vector_store=vector_store)
    
    print("✅ Components initialized")
    print()
    
    # Example 1: Process a PDF and add to vector store
    pdf_path = "data/pdfs/sample.pdf"
    
    if os.path.exists(pdf_path):
        print("=" * 60)
        print("EXAMPLE 1: Processing PDF and Indexing")
        print("=" * 60)
        
        document = processor.process_pdf(pdf_path)
        chunks = processor.chunk_document(document, chunk_size=500, overlap=100)
        
        print(f"Processed {len(chunks)} chunks from {document.get('file_name', 'unknown')}")
        
        # Add to vector store
        vector_store.add_documents(chunks)
        print("✅ Documents indexed in vector store")
        print()
        
        # Example 2: General Query
        print("=" * 60)
        print("EXAMPLE 2: General Query")
        print("=" * 60)
        
        query = "What is the main topic of this document?"
        print(f"Query: {query}")
        print()
        
        response = query_engine.query(query, top_k=3)
        
        print(f"Answer: {response.get('answer', 'N/A')}")
        print(f"Confidence: {response.get('confidence', 0.0):.2f}")
        print(f"Sources: {len(response.get('sources', []))} documents")
        print()
        
        # Example 3: Direct Content Lookup
        print("=" * 60)
        print("EXAMPLE 3: Direct Content Lookup")
        print("=" * 60)
        
        query = "What is the conclusion?"
        doc_id = document.get("file_name", "sample.pdf")
        print(f"Query: {query}")
        print(f"Document: {doc_id}")
        print()
        
        response = query_engine.direct_lookup(query, doc_id)
        
        print(f"Answer: {response.get('answer', 'N/A')[:200]}...")
        print(f"Intent: {response.get('intent', 'N/A')}")
        print()
        
        # Example 4: Summarization
        print("=" * 60)
        print("EXAMPLE 4: Summarization")
        print("=" * 60)
        
        query = "Summarize the methodology"
        print(f"Query: {query}")
        print(f"Document: {doc_id}")
        print()
        
        response = query_engine.summarize(query, doc_id)
        
        print(f"Summary: {response.get('answer', 'N/A')[:300]}...")
        print(f"Context chunks used: {response.get('context_chunks_used', 0)}")
        print()
        
        # Example 5: Metrics Extraction
        print("=" * 60)
        print("EXAMPLE 5: Metrics Extraction")
        print("=" * 60)
        
        query = "What are the accuracy and F1-score?"
        print(f"Query: {query}")
        print(f"Document: {doc_id}")
        print()
        
        response = query_engine.extract_metrics(query, doc_id)
        
        print(f"Answer: {response.get('answer', 'N/A')[:200]}...")
        metrics = response.get('metrics', {})
        if metrics:
            print(f"Extracted Metrics: {metrics}")
        print()
        
    else:
        print(f"PDF file not found: {pdf_path}")
        print("\nTo test the query engine:")
        print("1. Place a PDF file in data/pdfs/")
        print("2. Update pdf_path in this script")
        print("3. Run: python example_query_usage.py")
        print()
        
        # Still demonstrate query engine without documents
        print("=" * 60)
        print("DEMO: Query Engine Features (without documents)")
        print("=" * 60)
        print()
        
        # Show intent classification
        test_queries = [
            "What is the conclusion of Paper X?",
            "Summarize the methodology",
            "What are the accuracy and F1-score?",
            "What is machine learning?"
        ]
        
        print("Intent Classification Examples:")
        for query in test_queries:
            intent = query_engine._classify_intent(query)
            print(f"  Query: {query}")
            print(f"  Intent: {intent}")
            print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
FastAPI Backend Server for Document Q&A AI Agent
Provides REST API endpoints for the React frontend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.utils import get_api_key, setup_logging

# Setup logging
setup_logging("INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A AI Agent API",
    description="Enterprise-Ready Document Q&A AI Agent with Gemini API",
    version="1.0.0"
)

# CORS middleware - allow all localhost ports for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # Allow direct access
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
document_processor = DocumentProcessor()
vector_store = VectorStore()
query_engine = None

# Initialize query engine
def init_query_engine():
    global query_engine
    if query_engine is None:
        query_engine = QueryEngine(
            api_key=get_api_key(),
            vector_store=vector_store,
            enable_caching=True,
            enable_rate_limiting=True,
            enable_conversation=True
        )
    return query_engine

@app.on_event("startup")
async def startup_event():
    """Initialize query engine on startup."""
    init_query_engine()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Document Q&A AI Agent API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store": vector_store.get_collection_stats(),
        "api_key_configured": get_api_key() is not None
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Processing result with document ID and chunks
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Process PDF
            result = document_processor.process_pdf(tmp_path)
            
            # Create chunks from processed document
            chunks = []
            if isinstance(result, dict):
                # Chunk the document into smaller pieces for vector storage
                chunks = document_processor.chunk_document(result)
            elif isinstance(result, list):
                for doc in result:
                    chunks.extend(document_processor.chunk_document(doc))
            
            # Add to vector store
            if chunks:
                vector_store.add_documents(chunks)
            
            # Get document metadata
            doc_metadata = {
                "filename": file.filename,
                "title": result.get("title", "Unknown") if isinstance(result, dict) else "Unknown",
                "chunks_count": len(chunks),
                "status": "processed"
            }
            
            return {
                "success": True,
                "message": "Document processed successfully",
                "document": doc_metadata,
                "chunks_added": len(chunks)
            }
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_documents(
    question: str = Form(...),
    user_id: Optional[str] = Form(None),
    top_k: int = Form(5)
):
    """
    Query the document collection.
    
    Args:
        question: User's question
        user_id: Optional user identifier
        top_k: Number of results to retrieve
        
    Returns:
        Query response with answer and sources
    """
    try:
        # Initialize query engine if needed
        engine = init_query_engine()
        
        if not engine.model:
            raise HTTPException(
                status_code=503,
                detail="Query engine not available. Please check API key configuration."
            )
        
        # Process query
        response = engine.query(
            question=question,
            top_k=top_k,
            user_id=user_id,
            use_cache=True
        )
        
        return {
            "success": True,
            "question": question,
            "answer": response.get("answer", ""),
            "confidence": response.get("confidence", 0.0),
            "sources": response.get("sources", []),
            "intent": response.get("intent", "general")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get list of processed documents."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "success": True,
            "documents_count": stats.get("total_documents", 0),
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from vector store."""
    try:
        vector_store.clear_collection()
        return {
            "success": True,
            "message": "All documents cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/conversation/{user_id}")
async def get_conversation(user_id: str):
    """Get conversation history for a user."""
    try:
        engine = init_query_engine()
        if engine.conversation_manager:
            history = engine.conversation_manager.get_history_summary()
            context = engine.conversation_manager.get_context_string()
            return {
                "success": True,
                "user_id": user_id,
                "history": history,
                "context": context
            }
        return {
            "success": True,
            "user_id": user_id,
            "history": {},
            "context": ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Clear conversation history for a user."""
    try:
        engine = init_query_engine()
        if engine.conversation_manager:
            engine.conversation_manager.clear_history()
            return {
                "success": True,
                "message": f"Conversation history cleared for user {user_id}"
            }
        return {
            "success": True,
            "message": "No conversation history to clear"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

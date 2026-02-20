"""
Vector Store Module
Handles embedding generation, vector storage, and similarity search.
"""

import os
import hashlib
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

# Try to import ChromaDB, fallback to in-memory storage if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB not available. Using in-memory vector store.")
    CHROMADB_AVAILABLE = False


class VectorStore:
    """
    Manages vector embeddings and similarity search for document chunks.
    Supports ChromaDB for persistent storage or in-memory storage as fallback.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the VectorStore.
        
        Args:
            collection_name: Name of the collection to use
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to persist ChromaDB data (None for in-memory)
        """
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Initialize vector database
        self.collection_name = collection_name
        self.persist_directory = persist_directory or "outputs/chroma_db"
        
        if CHROMADB_AVAILABLE:
            try:
                # Create persist directory if needed
                if persist_directory:
                    os.makedirs(persist_directory, exist_ok=True)
                
                # Initialize ChromaDB client
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory
                ) if persist_directory else chromadb.Client()
                
                # Get or create collection
                try:
                    self.collection = self.client.get_collection(name=collection_name)
                    logger.info(f"Loaded existing collection: {collection_name}")
                except:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                
                self.use_chromadb = True
                logger.info("ChromaDB initialized successfully")
                
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {str(e)}. Using in-memory storage.")
                self.use_chromadb = False
                self._init_in_memory()
        else:
            self.use_chromadb = False
            self._init_in_memory()
    
    def _init_in_memory(self):
        """Initialize in-memory storage as fallback."""
        self.documents = []  # List of document dicts
        self.embeddings = []  # List of embedding vectors
        self.ids = []  # List of document IDs
        logger.info("Initialized in-memory vector store")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Generate embeddings in batches for efficiency
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _generate_id(self, chunk: Dict) -> str:
        """Generate a unique ID for a chunk."""
        content = chunk.get("content", "")
        doc_id = chunk.get("document_id", "unknown")
        chunk_id = chunk.get("chunk_id", 0)
        
        # Create hash-based ID
        id_string = f"{doc_id}_{chunk_id}_{content[:50]}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def add_documents(self, documents: List[Dict], embeddings: Optional[List[List[float]]] = None):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with metadata
            embeddings: Optional pre-computed embeddings. If None, will generate them.
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [doc.get("content", "") for doc in documents]
            embeddings = self.generate_embeddings(texts)
        
        # Prepare data for storage
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Generate unique ID
            doc_id = self._generate_id(doc)
            ids.append(doc_id)
            
            # Extract text content
            texts.append(doc.get("content", ""))
            
            # Prepare metadata
            metadata = {
                "document_id": doc.get("document_id", "unknown"),
                "section": doc.get("section", ""),
                "level": str(doc.get("level", 0)),
                "chunk_id": str(doc.get("chunk_id", i)),
                "type": doc.get("metadata", {}).get("type", "unknown")
            }
            metadatas.append(metadata)
        
        # Store in vector database
        if self.use_chromadb:
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully added {len(documents)} documents to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding to ChromaDB: {str(e)}")
                raise
        else:
            # Store in memory
            self.documents.extend(documents)
            self.embeddings.extend(embeddings)
            self.ids.extend(ids)
            logger.info(f"Successfully added {len(documents)} documents to in-memory store")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            filter: Optional metadata filter (e.g., {"document_id": "paper1.pdf"})
            
        Returns:
            List of similar documents with metadata and scores
        """
        if not query:
            return []
        
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Perform search
        if self.use_chromadb:
            try:
                # Build where clause for filtering
                where_clause = None
                if filter:
                    where_clause = filter
                
                # Query ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause
                )
                
                # Format results
                formatted_results = []
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        result = {
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "distance": results['distances'][0][i] if 'distances' in results else None,
                            "score": 1 - results['distances'][0][i] if 'distances' in results else None
                        }
                        formatted_results.append(result)
                
                logger.info(f"Found {len(formatted_results)} results")
                return formatted_results
                
            except Exception as e:
                logger.error(f"Error searching ChromaDB: {str(e)}")
                return []
        else:
            # In-memory search using cosine similarity
            return self._search_in_memory(query_embedding, top_k, filter)
    
    def _search_in_memory(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """Perform in-memory similarity search."""
        if not self.embeddings:
            return []
        
        # Convert to numpy arrays for efficient computation
        query_vec = np.array(query_embedding)
        doc_embeddings = np.array(self.embeddings)
        
        # Compute cosine similarities
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        # Apply filters if provided
        indices = list(range(len(self.documents)))
        if filter:
            filtered_indices = []
            for i, doc in enumerate(self.documents):
                match = True
                for key, value in filter.items():
                    if doc.get(key) != value and doc.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
            indices = filtered_indices
            similarities = similarities[filtered_indices]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            actual_idx = indices[idx]
            doc = self.documents[actual_idx]
            # Build metadata with document_id and other fields
            metadata = {
                "document_id": doc.get("document_id", "unknown"),
                "section": doc.get("section", ""),
                "level": str(doc.get("level", 0)),
                "chunk_id": str(doc.get("chunk_id", actual_idx)),
                "type": doc.get("metadata", {}).get("type", "unknown")
            }
            result = {
                "id": self.ids[actual_idx],
                "content": doc.get("content", ""),
                "metadata": metadata,
                "distance": float(1 - similarities[idx]),
                "score": float(similarities[idx])
            }
            results.append(result)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            
        Returns:
            List of similar documents with combined scores
        """
        logger.info(f"Hybrid search for: '{query}'")
        
        # Semantic search
        semantic_results = self.search(query, top_k=top_k * 2)
        
        # Keyword search (simple TF-based)
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        # Combine results
        combined = self._merge_search_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Rerank and return top-k
        reranked = self._rerank_results(combined, query)
        
        return reranked[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword-based search using TF scoring."""
        query_terms = query.lower().split()
        
        if self.use_chromadb:
            # Use ChromaDB's text search if available
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                # Format similar to semantic search
                formatted_results = []
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        result = {
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i],
                            "score": 0.5  # Placeholder score
                        }
                        formatted_results.append(result)
                return formatted_results
            except:
                pass
        
        # Fallback: simple in-memory keyword matching
        scores = []
        for i, doc in enumerate(self.documents):
            content = doc.get("content", "").lower()
            score = sum(1 for term in query_terms if term in content) / len(query_terms)
            scores.append((i, score))
        
        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                result = {
                    "id": self.ids[idx],
                    "content": self.documents[idx].get("content", ""),
                    "metadata": self.documents[idx].get("metadata", {}),
                    "score": score
                }
                results.append(result)
        
        return results
    
    def _merge_search_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_weight: float,
        keyword_weight: float
    ) -> Dict[str, Dict]:
        """Merge semantic and keyword search results."""
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result["id"]
            combined[doc_id] = {
                **result,
                "semantic_score": result.get("score", 0) * semantic_weight,
                "keyword_score": 0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result["id"]
            if doc_id in combined:
                combined[doc_id]["keyword_score"] = result.get("score", 0) * keyword_weight
            else:
                combined[doc_id] = {
                    **result,
                    "semantic_score": 0,
                    "keyword_score": result.get("score", 0) * keyword_weight
                }
        
        # Calculate combined scores
        for doc_id in combined:
            combined[doc_id]["combined_score"] = (
                combined[doc_id]["semantic_score"] + combined[doc_id]["keyword_score"]
            )
        
        return combined
    
    def _rerank_results(self, results: Dict[str, Dict], query: str) -> List[Dict]:
        """Rerank results by combined score."""
        # Convert to list and sort by combined score
        results_list = list(results.values())
        results_list.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return results_list
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.use_chromadb:
            try:
                count = self.collection.count()
                return {
                    "type": "ChromaDB",
                    "collection_name": self.collection_name,
                    "document_count": count,
                    "embedding_dimension": self.embedding_dim
                }
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                return {"error": str(e)}
        else:
            return {
                "type": "In-Memory",
                "document_count": len(self.documents),
                "embedding_dimension": self.embedding_dim
            }
    
    def delete_collection(self):
        """Delete the collection (use with caution!)."""
        if self.use_chromadb:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}")
        else:
            self.documents = []
            self.embeddings = []
            self.ids = []
            logger.info("Cleared in-memory store")
    
    def clear_collection(self):
        """Clear all documents from the collection without deleting it."""
        if self.use_chromadb:
            try:
                # Delete and recreate collection
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Cleared collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Error clearing collection: {str(e)}")
        else:
            self.documents = []
            self.embeddings = []
            self.ids = []
            logger.info("Cleared in-memory store")

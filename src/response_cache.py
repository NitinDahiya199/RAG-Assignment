"""
Response Cache Module
Implements caching for query responses to improve performance.
"""

import hashlib
import json
import time
from typing import Dict, Optional, Any
import logging

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class ResponseCache:
    """
    Caches query responses to avoid redundant processing.
    """
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        Initialize the ResponseCache.
        
        Args:
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            max_size: Maximum number of cache entries
        """
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl
        self.max_size = max_size
        logger.info(f"Initialized ResponseCache (ttl={ttl}s, max_size={max_size})")
    
    def _generate_key(self, query: str, context_hash: Optional[str] = None) -> str:
        """
        Generate cache key from query and optional context.
        
        Args:
            query: User query
            context_hash: Optional hash of context
            
        Returns:
            Cache key string
        """
        key_string = query.lower().strip()
        if context_hash:
            key_string += context_hash
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        context_hash: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get cached response if available and valid.
        
        Args:
            query: User query
            context_hash: Optional hash of context
            
        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(query, context_hash)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            logger.debug(f"Cache entry expired for query: {query[:50]}")
            return None
        
        logger.debug(f"Cache hit for query: {query[:50]}")
        return entry["response"]
    
    def set(
        self,
        query: str,
        response: Dict,
        context_hash: Optional[str] = None
    ) -> None:
        """
        Store response in cache.
        
        Args:
            query: User query
            response: Response to cache
            context_hash: Optional hash of context
        """
        key = self._generate_key(query, context_hash)
        
        # Evict if cache is full (remove oldest)
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query[:100]  # Store query preview for debugging
        }
        
        logger.debug(f"Cached response for query: {query[:50]}")
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]["timestamp"]
        )
        del self.cache[oldest_key]
        logger.debug("Evicted oldest cache entry")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache = {}
        logger.info(f"Cleared {count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        now = time.time()
        expired_count = sum(
            1 for entry in self.cache.values()
            if now - entry["timestamp"] > self.ttl
        )
        
        return {
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "ttl": self.ttl
        }
    
    def invalidate(self, query: str, context_hash: Optional[str] = None) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            query: Query to invalidate
            context_hash: Optional context hash
            
        Returns:
            True if entry was found and removed
        """
        key = self._generate_key(query, context_hash)
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache entry for query: {query[:50]}")
            return True
        return False

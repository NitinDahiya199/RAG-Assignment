"""
Conversation Manager Module
Handles conversation history, context management, and multi-turn conversations.
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class ConversationManager:
    """
    Manages conversation history and context for multi-turn conversations.
    """
    
    def __init__(self, max_history: int = 10, max_context_tokens: int = 4000):
        """
        Initialize the ConversationManager.
        
        Args:
            max_history: Maximum number of conversation turns to keep
            max_context_tokens: Maximum tokens for context window
        """
        self.history: List[Dict] = []
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        logger.info(f"Initialized ConversationManager (max_history={max_history})")
    
    def add_turn(
        self,
        query: str,
        response: Dict,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a conversation turn to history.
        
        Args:
            query: User's query
            response: System's response
            metadata: Optional metadata (timestamp, session_id, etc.)
        """
        turn = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.history.append(turn)
        
        # Prune if exceeds max history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            logger.debug(f"Pruned conversation history to {self.max_history} turns")
    
    def get_context(self, include_recent: int = 3) -> List[Dict]:
        """
        Get recent conversation context.
        
        Args:
            include_recent: Number of recent turns to include
            
        Returns:
            List of recent conversation turns
        """
        return self.history[-include_recent:] if self.history else []
    
    def get_context_string(self, include_recent: int = 3) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            include_recent: Number of recent turns to include
            
        Returns:
            Formatted context string
        """
        context_turns = self.get_context(include_recent)
        
        if not context_turns:
            return ""
        
        context_parts = []
        for i, turn in enumerate(context_turns, 1):
            query = turn.get("query", "")
            answer = turn.get("response", {}).get("answer", "")
            
            context_parts.append(
                f"Previous Turn {i}:\n"
                f"Q: {query}\n"
                f"A: {answer[:200]}...\n"
            )
        
        return "\n".join(context_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_history_summary(self) -> Dict:
        """
        Get summary of conversation history.
        
        Returns:
            Dictionary with history statistics
        """
        return {
            "total_turns": len(self.history),
            "max_history": self.max_history,
            "oldest_turn": self.history[0].get("timestamp") if self.history else None,
            "newest_turn": self.history[-1].get("timestamp") if self.history else None
        }
    
    def prune_by_tokens(self, current_tokens: int) -> None:
        """
        Prune conversation history if approaching token limit.
        
        Args:
            current_tokens: Current token count
        """
        if current_tokens < self.max_context_tokens * 0.8:
            return  # No need to prune
        
        # Remove oldest turns until under limit
        while current_tokens > self.max_context_tokens * 0.7 and len(self.history) > 1:
            removed = self.history.pop(0)
            # Estimate tokens removed (rough: 1 token ≈ 4 characters)
            estimated_tokens = len(str(removed)) // 4
            current_tokens -= estimated_tokens
            logger.debug(f"Pruned conversation turn to manage token limit")
    
    def get_relevant_context(self, current_query: str) -> str:
        """
        Get relevant context for current query.
        
        Args:
            current_query: Current user query
            
        Returns:
            Relevant context string
        """
        # Simple relevance: include recent turns
        # Could be enhanced with semantic similarity
        return self.get_context_string(include_recent=3)

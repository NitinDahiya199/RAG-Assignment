"""
Rate Limiter Module
Implements rate limiting to prevent API abuse and manage costs.
"""

import time
from typing import Dict, Optional
from collections import defaultdict
import logging

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class RateLimiter:
    """
    Implements rate limiting for API calls and operations.
    """
    
    def __init__(
        self,
        max_requests: int = 60,
        time_window: int = 60,
        per_user: bool = False
    ):
        """
        Initialize the RateLimiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds (default: 60 seconds)
            per_user: If True, rate limit per user; if False, global limit
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.per_user = per_user
        
        # Track requests: {identifier: [(timestamp, ...), ...]}
        self.requests: Dict[str, list] = defaultdict(list)
        
        logger.info(
            f"Initialized RateLimiter: {max_requests} requests per {time_window}s "
            f"(per_user={per_user})"
        )
    
    def _get_identifier(self, user_id: Optional[str] = None) -> str:
        """
        Get identifier for rate limiting.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Identifier string
        """
        if self.per_user and user_id:
            return f"user_{user_id}"
        return "global"
    
    def is_allowed(
        self,
        user_id: Optional[str] = None,
        operation: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            user_id: Optional user identifier
            operation: Optional operation name for logging
            
        Returns:
            Tuple of (is_allowed, message_if_not_allowed)
        """
        identifier = self._get_identifier(user_id)
        now = time.time()
        
        # Clean old requests outside time window
        self.requests[identifier] = [
            ts for ts in self.requests[identifier]
            if now - ts < self.time_window
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            oldest_request = min(self.requests[identifier])
            wait_time = self.time_window - (now - oldest_request)
            
            message = (
                f"Rate limit exceeded. Maximum {self.max_requests} requests "
                f"per {self.time_window} seconds. Please wait {wait_time:.1f} seconds."
            )
            
            logger.warning(
                f"Rate limit exceeded for {identifier} "
                f"(operation: {operation or 'unknown'})"
            )
            
            return False, message
        
        # Record request
        self.requests[identifier].append(now)
        
        logger.debug(
            f"Request allowed for {identifier} "
            f"({len(self.requests[identifier])}/{self.max_requests} in window)"
        )
        
        return True, None
    
    def get_remaining(
        self,
        user_id: Optional[str] = None
    ) -> int:
        """
        Get remaining requests in current time window.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Number of remaining requests
        """
        identifier = self._get_identifier(user_id)
        now = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            ts for ts in self.requests[identifier]
            if now - ts < self.time_window
        ]
        
        remaining = self.max_requests - len(self.requests[identifier])
        return max(0, remaining)
    
    def reset(self, user_id: Optional[str] = None) -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            user_id: Optional user identifier
        """
        identifier = self._get_identifier(user_id)
        self.requests[identifier] = []
        logger.info(f"Reset rate limit for {identifier}")
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict:
        """
        Get rate limiting statistics.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Dictionary with statistics
        """
        identifier = self._get_identifier(user_id)
        now = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            ts for ts in self.requests[identifier]
            if now - ts < self.time_window
        ]
        
        return {
            "identifier": identifier,
            "requests_in_window": len(self.requests[identifier]),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "remaining": self.get_remaining(user_id)
        }

"""
Function Calling Module
Handles Arxiv API integration for paper lookup (Bonus Feature).
"""

import os
import re
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logging.getLogger(__name__).warning("Arxiv library not available. Install with: pip install arxiv")

from src.utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class ArxivFunctionCalling:
    """
    Integrates Arxiv API with Gemini function calling capability.
    Allows the AI agent to search for papers based on user descriptions.
    """
    
    def __init__(self):
        """Initialize the ArxivFunctionCalling."""
        if not ARXIV_AVAILABLE:
            logger.warning("Arxiv library not available. Function calling will be limited.")
    
    def get_function_schema(self) -> Dict:
        """
        Get the function schema for Arxiv search to be used with Gemini.
        
        Returns:
            Dictionary containing function schema in Gemini format
        """
        return {
            "function_declarations": [
                {
                    "name": "search_arxiv_paper",
                    "description": (
                        "Search for academic papers on Arxiv based on user query. "
                        "Use this function when the user asks to find, search, or look up papers, "
                        "or wants to discover research papers on a specific topic."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Search query string. Can be keywords, author name, paper title, "
                                    "or a combination. Examples: 'transformer architecture', "
                                    "'attention mechanism', 'author:Smith', 'BERT model'"
                                )
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (1-20). Default is 5 if not specified."
                            },
                            "sort_by": {
                                "type": "string",
                                "enum": ["relevance", "submittedDate", "lastUpdatedDate"],
                                "description": (
                                    "How to sort the search results. "
                                    "'relevance' sorts by relevance to query (default), "
                                    "'submittedDate' sorts by submission date (newest first), "
                                    "'lastUpdatedDate' sorts by last update date."
                                )
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
        }
    
    def search_arxiv_paper(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance"
    ) -> List[Dict]:
        """
        Search for papers on Arxiv based on query.
        
        Args:
            query: Search query string (keywords, author, title, etc.)
            max_results: Maximum number of results to return
            sort_by: Sort order ("relevance", "submittedDate", "lastUpdatedDate")
            
        Returns:
            List of paper dictionaries with metadata
        """
        if not ARXIV_AVAILABLE:
            logger.error("Arxiv library not available")
            return []
        
        if not query:
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Searching Arxiv for: '{query}' (max_results={max_results}, sort_by={sort_by})")
        
        try:
            # Map sort_by to Arxiv sort criterion
            sort_criterion_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate
            }
            sort_criterion = sort_criterion_map.get(sort_by, arxiv.SortCriterion.Relevance)
            
            # Perform search
            search = arxiv.Search(
                query=query,
                max_results=min(max_results, 20),  # Limit to 20
                sort_by=sort_criterion
            )
            
            # Extract results
            papers = []
            for result in search.results():
                try:
                    paper = {
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "published": result.published.isoformat() if result.published else None,
                        "arxiv_id": result.entry_id.split('/')[-1] if '/' in result.entry_id else result.entry_id,
                        "arxiv_url": result.entry_id,
                        "pdf_url": result.pdf_url,
                        "categories": list(result.categories),
                        "primary_category": result.primary_category if hasattr(result, 'primary_category') else None,
                        "doi": result.doi if hasattr(result, 'doi') and result.doi else None
                    }
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Error processing paper result: {str(e)}")
                    continue
            
            logger.info(f"Found {len(papers)} papers on Arxiv")
            return papers
            
        except arxiv.UnexpectedEmptyPageError:
            logger.warning("Arxiv search returned empty page")
            return []
        except Exception as e:
            logger.error(f"Error searching Arxiv: {str(e)}")
            return []
    
    def search_by_author(self, author_name: str, max_results: int = 5) -> List[Dict]:
        """
        Search for papers by author name.
        
        Args:
            author_name: Name of the author
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"au:{author_name}"
        return self.search_arxiv_paper(query, max_results=max_results)
    
    def search_by_title(self, title: str, max_results: int = 5) -> List[Dict]:
        """
        Search for papers by title.
        
        Args:
            title: Paper title or keywords from title
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"ti:{title}"
        return self.search_arxiv_paper(query, max_results=max_results)
    
    def execute_function(self, function_name: str, arguments: Dict) -> Dict:
        """
        Execute a function call from Gemini.
        
        Args:
            function_name: Name of the function to execute
            arguments: Function arguments as dictionary
            
        Returns:
            Function execution result
        """
        logger.info(f"Executing function: {function_name} with arguments: {arguments}")
        
        if function_name == "search_arxiv_paper":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            sort_by = arguments.get("sort_by", "relevance")
            
            papers = self.search_arxiv_paper(query, max_results=max_results, sort_by=sort_by)
            
            return {
                "function_name": function_name,
                "result": papers,
                "count": len(papers),
                "success": True
            }
        else:
            logger.error(f"Unknown function: {function_name}")
            return {
                "function_name": function_name,
                "result": None,
                "error": f"Unknown function: {function_name}",
                "success": False
            }
    
    def format_papers_for_response(self, papers: List[Dict]) -> str:
        """
        Format paper results for display in response.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Formatted string representation
        """
        if not papers:
            return "No papers found."
        
        formatted = []
        for i, paper in enumerate(papers, 1):
            paper_str = f"\n{i}. **{paper['title']}**\n"
            paper_str += f"   Authors: {', '.join(paper['authors'][:3])}"
            if len(paper['authors']) > 3:
                paper_str += f" et al."
            paper_str += "\n"
            
            if paper.get('published'):
                paper_str += f"   Published: {paper['published'][:10]}\n"
            
            paper_str += f"   Abstract: {paper['abstract'][:200]}...\n"
            paper_str += f"   Arxiv ID: {paper['arxiv_id']}\n"
            paper_str += f"   PDF: {paper['pdf_url']}\n"
            
            formatted.append(paper_str)
        
        return "\n".join(formatted)
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get a specific paper by Arxiv ID.
        
        Args:
            arxiv_id: Arxiv ID (e.g., "2301.12345" or "2301.12345v1")
            
        Returns:
            Paper dictionary or None if not found
        """
        if not ARXIV_AVAILABLE:
            return None
        
        try:
            # Remove version suffix if present
            arxiv_id = arxiv_id.split('v')[0]
            
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(search.results())
            
            if results:
                result = results[0]
                return {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": result.published.isoformat() if result.published else None,
                    "arxiv_id": arxiv_id,
                    "arxiv_url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "categories": list(result.categories)
                }
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {str(e)}")
        
        return None

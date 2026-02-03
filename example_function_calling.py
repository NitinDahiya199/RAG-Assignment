"""
Example usage of Function Calling (Arxiv Integration) for Phase 5.
This script demonstrates how to use the Arxiv function calling functionality.
"""

from src.function_calling import ArxivFunctionCalling
from src.query_engine import QueryEngine
from src.utils import setup_logging, get_api_key

# Set up logging
setup_logging("INFO")


def main():
    """Example usage of Arxiv Function Calling."""
    
    print("=" * 60)
    print("Document Q&A AI Agent - Function Calling Demo (Phase 5)")
    print("=" * 60)
    print()
    
    # Initialize Arxiv Function Calling
    print("Initializing Arxiv Function Calling...")
    arxiv_func = ArxivFunctionCalling()
    
    # Get function schema
    schema = arxiv_func.get_function_schema()
    print(f"Function schema created: {schema['function_declarations'][0]['name']}")
    print()
    
    # Example 1: Basic Arxiv Search
    print("=" * 60)
    print("EXAMPLE 1: Basic Arxiv Search")
    print("=" * 60)
    
    query = "transformer architecture"
    print(f"Searching for: '{query}'")
    print()
    
    papers = arxiv_func.search_arxiv_paper(query, max_results=3)
    
    if papers:
        print(f"Found {len(papers)} papers:")
        print()
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:2])}")
            print(f"   Arxiv ID: {paper['arxiv_id']}")
            print(f"   Abstract: {paper['abstract'][:150]}...")
            print()
    else:
        print("No papers found or Arxiv API unavailable")
        print()
    
    # Example 2: Search by Author
    print("=" * 60)
    print("EXAMPLE 2: Search by Author")
    print("=" * 60)
    
    author = "Vaswani"
    print(f"Searching for papers by: {author}")
    print()
    
    papers = arxiv_func.search_by_author(author, max_results=2)
    
    if papers:
        print(f"Found {len(papers)} papers:")
        for paper in papers:
            print(f"  - {paper['title']}")
            print(f"    Arxiv ID: {paper['arxiv_id']}")
            print()
    else:
        print("No papers found")
        print()
    
    # Example 3: Function Execution
    print("=" * 60)
    print("EXAMPLE 3: Function Execution")
    print("=" * 60)
    
    function_result = arxiv_func.execute_function(
        "search_arxiv_paper",
        {
            "query": "BERT model",
            "max_results": 2
        }
    )
    
    print(f"Function: {function_result['function_name']}")
    print(f"Success: {function_result['success']}")
    print(f"Results: {function_result['count']} papers")
    print()
    
    # Example 4: Format Papers for Response
    print("=" * 60)
    print("EXAMPLE 4: Formatted Paper Response")
    print("=" * 60)
    
    if papers:
        formatted = arxiv_func.format_papers_for_response(papers[:2])
        print(formatted)
        print()
    
    # Example 5: Integration with QueryEngine
    print("=" * 60)
    print("EXAMPLE 5: QueryEngine with Function Calling")
    print("=" * 60)
    
    api_key = get_api_key()
    if api_key:
        query_engine = QueryEngine(api_key=api_key, enable_function_calling=True)
        
        query = "Find papers about attention mechanisms"
        print(f"Query: {query}")
        print()
        print("Note: This requires Gemini API to be configured.")
        print("The QueryEngine will automatically detect Arxiv search needs")
        print("and call the search_arxiv_paper function.")
        print()
    else:
        print("Warning: Gemini API key not found.")
        print("   Function calling integration requires API key.")
        print("   Set GEMINI_API_KEY in .env file to test this feature.")
        print()
    
    # Example 6: Get Paper by ID
    print("=" * 60)
    print("EXAMPLE 6: Get Paper by Arxiv ID")
    print("=" * 60)
    
    arxiv_id = "1706.03762"  # Attention Is All You Need
    print(f"Fetching paper: {arxiv_id}")
    print()
    
    paper = arxiv_func.get_paper_by_id(arxiv_id)
    
    if paper:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Published: {paper.get('published', 'N/A')}")
        print(f"PDF: {paper['pdf_url']}")
        print()
    else:
        print("Paper not found or API unavailable")
        print()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Usage Tips:")
    print("- Use search_arxiv_paper() for general keyword searches")
    print("- Use search_by_author() for author-specific searches")
    print("- Use search_by_title() for title-based searches")
    print("- Use get_paper_by_id() to fetch a specific paper")
    print("- Integrate with QueryEngine for automatic function calling")


if __name__ == "__main__":
    main()

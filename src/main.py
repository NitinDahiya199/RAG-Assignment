"""
Main Entry Point
Command-line interface for the Document Q&A AI Agent.
"""

import sys
import logging
from src.utils import setup_logging, validate_api_key

# Set up logging
setup_logging()


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("Document Q&A AI Agent - Enterprise-Ready Prototype")
    print("=" * 60)
    print()
    
    # Validate API key
    if not validate_api_key():
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        print("See .env.example for reference")
        sys.exit(1)
    
    print("✅ API key validated")
    print()
    print("🚀 Application ready!")
    print()
    print("Note: Full implementation will be available after Phase 2-4")
    print("See roadmap.md for development progress")


if __name__ == "__main__":
    main()

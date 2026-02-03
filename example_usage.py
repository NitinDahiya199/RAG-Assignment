"""
Example usage of DocumentProcessor for Phase 2.
This script demonstrates how to use the document processing functionality.
"""

import os
from src.document_processor import DocumentProcessor
from src.utils import setup_logging

# Set up logging
setup_logging("INFO")


def main():
    """Example usage of DocumentProcessor."""
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Example: Process a single PDF
    pdf_path = "data/pdfs/sample.pdf"  # Replace with your PDF path
    
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        
        # Process the PDF
        document = processor.process_pdf(pdf_path)
        
        # Display results
        print("\n" + "="*60)
        print("DOCUMENT PROCESSING RESULTS")
        print("="*60)
        
        print(f"\nTitle: {document.get('title', 'N/A')}")
        print(f"\nAbstract: {document.get('abstract', 'N/A')[:200]}...")
        print(f"\nTotal Pages: {document.get('metadata', {}).get('total_pages', 'N/A')}")
        print(f"\nSections Found: {len(document.get('sections', []))}")
        print(f"Tables Found: {len(document.get('tables', []))}")
        print(f"Figures Found: {len(document.get('figures', []))}")
        print(f"References Found: {len(document.get('references', []))}")
        
        # Show first few sections
        print("\n" + "-"*60)
        print("SECTIONS:")
        print("-"*60)
        for i, section in enumerate(document.get('sections', [])[:5], 1):
            print(f"\n{i}. {section['heading']} (Level {section['level']})")
            print(f"   Content preview: {section['content'][:100]}...")
        
        # Chunk the document
        print("\n" + "-"*60)
        print("CHUNKING DOCUMENT:")
        print("-"*60)
        chunks = processor.chunk_document(document, chunk_size=500, overlap=100)
        print(f"Total chunks created: {len(chunks)}")
        print(f"\nFirst chunk preview:")
        if chunks:
            print(f"  Section: {chunks[0].get('section', 'N/A')}")
            print(f"  Content: {chunks[0].get('content', '')[:150]}...")
    
    else:
        print(f"PDF file not found: {pdf_path}")
        print("\nTo test the processor:")
        print("1. Place a PDF file in data/pdfs/")
        print("2. Update pdf_path in this script")
        print("3. Run: python example_usage.py")
    
    # Example: Process multiple PDFs
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    pdf_directory = "data/pdfs"
    if os.path.exists(pdf_directory):
        pdf_files = [os.path.join(pdf_directory, f) 
                    for f in os.listdir(pdf_directory) 
                    if f.endswith('.pdf')]
        
        if pdf_files:
            print(f"\nFound {len(pdf_files)} PDF files")
            results = processor.process_multiple_pdfs(pdf_files)
            print(f"Successfully processed {len(results)} PDFs")
        else:
            print(f"\nNo PDF files found in {pdf_directory}")
    else:
        print(f"\nDirectory not found: {pdf_directory}")


if __name__ == "__main__":
    main()

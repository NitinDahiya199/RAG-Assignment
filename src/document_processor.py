"""
Document Processor Module
Handles PDF ingestion, extraction, and structuring of document content.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import pypdf
import pdfplumber
from pdfplumber import PDF
import google.generativeai as genai

from src.utils import get_api_key, setup_logging

logger = logging.getLogger(__name__)
setup_logging()


class DocumentProcessor:
    """
    Processes PDF documents and extracts structured content including:
    - Titles and headings
    - Abstracts
    - Sections
    - Tables
    - Figures
    - References
    - Equations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DocumentProcessor.
        
        Args:
            api_key: Google Gemini API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or get_api_key()
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Try multiple model names as fallback
            fallback_models = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
                'gemini-2.5-flash'
            ]
            self.vision_model = None
            for model_to_try in fallback_models:
                try:
                    self.vision_model = genai.GenerativeModel(model_to_try)
                    logger.info(f"Initialized DocumentProcessor with model: {model_to_try}")
                    break
                except Exception as model_error:
                    logger.debug(f"Failed to initialize model {model_to_try}: {str(model_error)}")
                    continue
            
            if self.vision_model is None:
                logger.warning("Failed to initialize any Gemini model. Multi-modal features will be disabled.")
        else:
            logger.warning("Gemini API key not found. Multi-modal features will be disabled.")
            self.vision_model = None
        
        # Common section patterns
        self.section_patterns = [
            r'^\d+\.?\s+[A-Z][^.]*$',  # Numbered sections: "1. Introduction"
            r'^[A-Z][A-Z\s]{3,}$',     # ALL CAPS headings
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # Title Case headings
        ]
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF document and extract structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing structured document data
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract basic text with pypdf
            raw_text = self._extract_text_pypdf(pdf_path)
            
            # Extract structured content with pdfplumber
            structured_data = self.extract_structure(pdf_path)
            
            # Combine results
            result = {
                "file_path": pdf_path,
                "file_name": os.path.basename(pdf_path),
                "raw_text": raw_text,
                **structured_data
            }
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_text_pypdf(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF using pypdf.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text_content = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"--- Page {page_num} ---\n{text}\n")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
                        continue
                
        except Exception as e:
            logger.error(f"Error reading PDF with pypdf: {str(e)}")
            raise
        
        return "\n".join(text_content)
    
    def extract_structure(self, pdf_path: str) -> Dict:
        """
        Extract structured content from PDF including sections, tables, figures.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with structured content
        """
        structured_data = {
            "title": "",
            "abstract": "",
            "sections": [],
            "tables": [],
            "figures": [],
            "references": [],
            "equations": [],
            "metadata": {}
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                structured_data["metadata"] = {
                    "total_pages": len(pdf.pages),
                    "file_size": os.path.getsize(pdf_path)
                }
                
                # Extract content from each page
                all_text = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append((page_num, page_text))
                    
                    # Extract tables from this page
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            structured_data["tables"].append({
                                "page": page_num,
                                "table_index": table_idx,
                                "data": table,
                                "caption": self._extract_table_caption(page_text, table_idx)
                            })
                
                # Process all text to extract structure
                full_text = "\n".join([text for _, text in all_text])
                
                # Extract title (usually first few lines or largest font)
                structured_data["title"] = self._extract_title(full_text, pdf)
                
                # Extract abstract
                structured_data["abstract"] = self._extract_abstract(full_text)
                
                # Extract sections
                structured_data["sections"] = self._extract_sections(full_text, pdf)
                
                # Extract references
                structured_data["references"] = self._extract_references(full_text)
                
                # Extract equations (basic pattern matching)
                structured_data["equations"] = self._extract_equations(full_text)
                
                # Extract figure captions
                structured_data["figures"] = self._extract_figure_captions(full_text)
                
        except Exception as e:
            logger.error(f"Error extracting structure from PDF: {str(e)}")
            raise
        
        return structured_data
    
    def _extract_title(self, text: str, pdf: PDF) -> str:
        """Extract document title from text."""
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        # Look for title patterns
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title (no periods, proper case)
                if not line.endswith('.') and line[0].isupper():
                    # Skip common non-title patterns
                    if not any(word in line.lower() for word in ['abstract', 'introduction', 'author', 'university']):
                        return line
        
        # Fallback: first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                return line
        
        return "Untitled Document"
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section from text."""
        # Common abstract patterns
        abstract_patterns = [
            r'(?i)abstract\s*\n\s*(.+?)(?=\n\s*(?:1\.|introduction|keywords|index terms))',
            r'(?i)abstract\s*\n\s*(.+?)(?=\n\s*\d+\.)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Minimum abstract length
                    return abstract[:2000]  # Limit abstract length
        
        return ""
    
    def _extract_sections(self, text: str, pdf: PDF) -> List[Dict]:
        """Extract sections with hierarchical structure."""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section heading
            is_heading = False
            level = 1
            
            # Pattern 1: Numbered sections (1., 1.1, 1.1.1)
            numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$', line)
            if numbered_match:
                is_heading = True
                level = len(numbered_match.group(1).split('.'))
            
            # Pattern 2: ALL CAPS (likely heading)
            elif re.match(r'^[A-Z][A-Z\s]{5,}$', line) and len(line) < 100:
                is_heading = True
                level = 1
            
            # Pattern 3: Title Case with no period - be more strict
            # Only consider it a heading if it's short (less than 50 chars) and looks like a title
            elif (re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line) and 
                  not line.endswith('.') and len(line) < 50 and len(line) > 3 and
                  line.lower() in ['abstract', 'introduction', 'conclusion', 'references', 
                                   'methodology', 'results', 'discussion', 'background',
                                   'related work', 'future work', 'acknowledgments', 'appendix']):
                is_heading = True
                level = 2
            
            if is_heading:
                # Save previous section
                if current_section:
                    current_section["content"] = "\n".join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                heading_text = numbered_match.group(2) if numbered_match else line
                current_section = {
                    "heading": heading_text,
                    "level": level,
                    "content": ""
                }
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Add last section
        if current_section:
            current_section["content"] = "\n".join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def _extract_tables(self, pdf: PDF) -> List[Dict]:
        """Extract tables from PDF."""
        tables = []
        
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables()
            for table_idx, table in enumerate(page_tables):
                if table and len(table) > 1:  # At least header + one row
                    tables.append({
                        "page": page_num,
                        "table_index": table_idx,
                        "data": table,
                        "caption": ""
                    })
        
        return tables
    
    def _extract_table_caption(self, page_text: str, table_index: int) -> str:
        """Extract caption for a table."""
        # Look for "Table X" patterns near the table
        table_pattern = re.compile(r'(?i)table\s+\d+[.:]\s*(.+?)(?=\n|$)', re.DOTALL)
        matches = list(table_pattern.finditer(page_text))
        
        if matches and table_index < len(matches):
            return matches[table_index].group(1).strip()
        
        return ""
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references/citations from text."""
        references = []
        
        # Find References section
        ref_section_match = re.search(
            r'(?i)(?:references|bibliography|works?\s+cited)\s*\n(.+)',
            text,
            re.DOTALL
        )
        
        if ref_section_match:
            ref_text = ref_section_match.group(1)
            
            # Split by common patterns
            # Pattern 1: Numbered references [1], [2], etc.
            numbered_refs = re.split(r'\[\d+\]', ref_text)
            if len(numbered_refs) > 1:
                references = [ref.strip() for ref in numbered_refs[1:] if ref.strip()]
            else:
                # Pattern 2: Line breaks (each line is a reference)
                ref_lines = [line.strip() for line in ref_text.split('\n') if line.strip()]
                references = [ref for ref in ref_lines if len(ref) > 20]  # Filter short lines
        
        return references[:100]  # Limit to 100 references
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text (basic pattern matching)."""
        equations = []
        
        # Pattern 1: LaTeX-style equations
        latex_pattern = r'\$[^$]+\$|\\\[[^\]]+\\\]|\\\([^\)]+\\\)'
        latex_matches = re.findall(latex_pattern, text)
        equations.extend(latex_matches)
        
        # Pattern 2: Numbered equations (Equation 1, Eq. 2, etc.)
        eq_pattern = r'(?i)(?:equation|eq\.?)\s+\d+[.:]\s*(.+?)(?=\n|equation|eq\.)'
        eq_matches = re.findall(eq_pattern, text, re.DOTALL)
        equations.extend([eq.strip() for eq in eq_matches if len(eq.strip()) > 5])
        
        return list(set(equations))  # Remove duplicates
    
    def _extract_figure_captions(self, text: str) -> List[Dict]:
        """Extract figure captions from text."""
        figures = []
        
        # Pattern: "Figure X: Caption" or "Fig. X: Caption"
        fig_pattern = r'(?i)(?:figure|fig\.?)\s+(\d+)[.:]\s*(.+?)(?=\n(?:figure|fig\.?|\d+\.|$))'
        matches = re.finditer(fig_pattern, text, re.DOTALL)
        
        for match in matches:
            figure_num = match.group(1)
            caption = match.group(2).strip()
            figures.append({
                "figure_number": int(figure_num),
                "caption": caption,
                "page": None  # Will be determined by context
            })
        
        return figures
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process multiple PDF documents in batch.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of dictionaries containing structured document data
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
                continue
        
        return results
    
    def chunk_document(self, document: Dict, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Chunk document into smaller pieces for vector storage.
        
        Args:
            document: Document dictionary from process_pdf
            chunk_size: Target chunk size in characters
            overlap: Overlap size between chunks
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        chunk_id = 0
        
        # Chunk by sections first (semantic chunking)
        for section in document.get("sections", []):
            content = f"{section['heading']}\n{section['content']}"
            
            if len(content) <= chunk_size:
                # Section fits in one chunk
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document.get("file_name", "unknown"),
                    "section": section["heading"],
                    "level": section["level"],
                    "content": content,
                    "metadata": {
                        "type": "section",
                        "heading": section["heading"]
                    }
                })
                chunk_id += 1
            else:
                # Split large section into smaller chunks
                words = content.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    
                    if current_length + word_length > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": document.get("file_name", "unknown"),
                            "section": section["heading"],
                            "level": section["level"],
                            "content": chunk_text,
                            "metadata": {
                                "type": "section_part",
                                "heading": section["heading"]
                            }
                        })
                        chunk_id += 1
                        
                        # Start new chunk with overlap
                        overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                        current_chunk = overlap_words + [word]
                        current_length = sum(len(w) + 1 for w in current_chunk)
                    else:
                        current_chunk.append(word)
                        current_length += word_length
                
                # Add remaining chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": document.get("file_name", "unknown"),
                        "section": section["heading"],
                        "level": section["level"],
                        "content": chunk_text,
                        "metadata": {
                            "type": "section_part",
                            "heading": section["heading"]
                        }
                    })
                    chunk_id += 1
        
        # Add abstract as a chunk if it exists
        if document.get("abstract"):
            chunks.insert(0, {
                "chunk_id": chunk_id,
                "document_id": document.get("file_name", "unknown"),
                "section": "Abstract",
                "level": 0,
                "content": document["abstract"],
                "metadata": {
                    "type": "abstract"
                }
            })
            chunk_id += 1
        
        # ALWAYS also chunk raw text to ensure we capture all content
        # This is crucial because section extraction may miss content
        if document.get("raw_text"):
            raw_text = document["raw_text"]
            
            # Check if sections have meaningful content
            total_section_content = sum(len(s.get("content", "")) for s in document.get("sections", []))
            raw_text_length = len(raw_text)
            
            # If sections captured less than 50% of raw text, add raw text chunks
            should_add_raw = (not chunks) or (total_section_content < raw_text_length * 0.5)
            
            if should_add_raw:
                logger.info(f"Adding raw text chunks (sections captured {total_section_content}/{raw_text_length} chars)")
                
                # Split raw text into chunks
                words = raw_text.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    
                    if current_length + word_length > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = " ".join(current_chunk)
                        # Only add if it has meaningful content (not just page markers)
                        if len(chunk_text) > 50 and not chunk_text.startswith("--- Page"):
                            chunks.append({
                                "chunk_id": chunk_id,
                                "document_id": document.get("file_name", "unknown"),
                                "section": "Full Content",
                                "level": 1,
                                "content": chunk_text,
                                "metadata": {
                                    "type": "text_chunk"
                                }
                            })
                            chunk_id += 1
                        
                        # Start new chunk with overlap
                        overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                        current_chunk = overlap_words + [word]
                        current_length = sum(len(w) + 1 for w in current_chunk)
                    else:
                        current_chunk.append(word)
                        current_length += word_length
                
                # Add remaining chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) > 50:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "document_id": document.get("file_name", "unknown"),
                            "section": "Full Content",
                            "level": 1,
                            "content": chunk_text,
                            "metadata": {
                                "type": "text_chunk"
                            }
                        })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def process_with_vision(self, pdf_path: str, page_num: int) -> Optional[Dict]:
        """
        Process a PDF page using Gemini Vision API for multi-modal extraction.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to process (1-indexed)
            
        Returns:
            Dictionary with vision-extracted content or None if API not available
        """
        if not self.vision_model:
            logger.warning("Vision API not available. Skipping vision processing.")
            return None
        
        try:
            import fitz  # PyMuPDF for image extraction
            doc = fitz.open(pdf_path)
            
            if page_num > len(doc):
                logger.error(f"Page {page_num} out of range")
                return None
            
            page = doc[page_num - 1]
            pix = page.get_pixmap()
            
            # Convert to PIL Image
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Use Gemini Vision to analyze
            response = self.vision_model.generate_content([
                "Extract and describe any figures, tables, diagrams, or complex layouts from this page. "
                "Provide structured information about visual elements.",
                img
            ])
            
            return {
                "page": page_num,
                "vision_analysis": response.text,
                "has_figures": "figure" in response.text.lower(),
                "has_tables": "table" in response.text.lower()
            }
            
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            return None
        except Exception as e:
            logger.error(f"Error in vision processing: {str(e)}")
            return None

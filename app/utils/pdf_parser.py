import fitz  # PyMuPDF
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If PDF cannot be opened or processed
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text")
                if page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        doc.close()
        return "\n".join(text_parts)
        
    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")

def extract_text_with_metadata(pdf_path: str) -> dict:
    """
    Extract text and metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing text, page count, and metadata
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Extract text
        text_parts = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text)
        
        # Extract metadata
        metadata = doc.metadata
        page_count = doc.page_count
        
        doc.close()
        
        return {
            "text": "\n".join(text_parts),
            "page_count": page_count,
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", "")
        }
        
    except Exception as e:
        logger.error(f"Error extracting PDF with metadata {pdf_path}: {e}")
        raise Exception(f"Failed to extract PDF with metadata: {e}")

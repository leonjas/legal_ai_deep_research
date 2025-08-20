import re
from typing import List
import logging

logger = logging.getLogger(__name__)

def split_into_clauses(text: str, min_words: int = 4) -> List[str]:
    """
    Split contract text into individual clauses for analysis.
    
    Args:
        text: Raw contract text
        min_words: Minimum number of words required for a clause
        
    Returns:
        List of cleaned clause strings
    """
    if not text or not text.strip():
        return []
    
    # Strategy 1: Try splitting on sentence endings (periods, newlines)
    # This works for most terms of service documents
    
    # First, try splitting on periods followed by whitespace or end of string
    period_splits = re.split(r'\.(?:\s+|$)', text)
    period_clauses = []
    
    for clause in period_splits:
        cleaned = re.sub(r"\s+", " ", clause).strip()
        if cleaned and len(cleaned.split()) >= min_words:
            period_clauses.append(cleaned)
    
    # If period splitting gives us good results, use it
    if len(period_clauses) > 1:
        logger.info(f"Split text using periods: {len(period_clauses)} clauses")
        return period_clauses
    
    # Strategy 2: Try splitting on newlines
    newline_splits = text.split('\n')
    newline_clauses = []
    
    for clause in newline_splits:
        cleaned = re.sub(r"\s+", " ", clause).strip()
        if cleaned and len(cleaned.split()) >= min_words:
            # Remove trailing period if present
            if cleaned.endswith('.'):
                cleaned = cleaned[:-1].strip()
            if cleaned and len(cleaned.split()) >= min_words:
                newline_clauses.append(cleaned)
    
    # If newline splitting gives us good results, use it
    if len(newline_clauses) > 1:
        logger.info(f"Split text using newlines: {len(newline_clauses)} clauses")
        return newline_clauses
    
    # Strategy 3: Fallback to complex patterns (original method)
    patterns = [
        r"\n{2,}",           # Multiple newlines
        r";(?=\s)",          # Semicolon followed by space
        r"\u2022",           # Bullet points
        r"^\s*\d+\.\s+",     # Numbered lists (1. 2. 3.)
        r"^\s*[a-z]\)\s+",   # Lettered lists (a) b) c)
        r"^\s*[A-Z]\.\s+",   # Capital letter lists (A. B. C.)
        r"^\s*\([a-z]\)\s+", # Parenthesized letters ((a) (b) (c))
        r"^\s*\(\d+\)\s+"    # Parenthesized numbers ((1) (2) (3))
    ]
    
    # Combine all patterns with OR
    combined_pattern = "|".join(f"({pattern})" for pattern in patterns)
    
    # Split the text
    parts = re.split(combined_pattern, text, flags=re.MULTILINE)
    
    # Clean and filter clauses
    clauses = []
    for part in parts:
        if part is None:
            continue
            
        # Clean whitespace and normalize
        cleaned = re.sub(r"\s+", " ", part).strip()
        
        # Skip empty parts or separators
        if not cleaned or cleaned in [";", "•"] or re.match(r"^\s*\d+\.\s*$", cleaned):
            continue
            
        # Filter by minimum word count
        word_count = len(cleaned.split())
        if word_count >= min_words:
            clauses.append(cleaned)
    
    logger.info(f"Split text into {len(clauses)} clauses (min {min_words} words each)")
    return clauses

def split_into_sentences(text: str, min_words: int = 4) -> List[str]:
    """
    Alternative approach: split into sentences rather than clauses.
    
    Args:
        text: Raw contract text
        min_words: Minimum number of words required
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean and filter
    cleaned_sentences = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if len(cleaned.split()) >= min_words:
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

def extract_section_headers(text: str) -> List[dict]:
    """
    Extract section headers from contract text for better organization.
    
    Args:
        text: Raw contract text
        
    Returns:
        List of dictionaries with header info
    """
    headers = []
    
    # Common header patterns
    header_patterns = [
        r"^(SECTION|Section)\s+(\d+)\.?\s*(.+)$",
        r"^(\d+)\.\s*([A-Z][^.]+)$",
        r"^([A-Z\s]+)$",  # All caps headers
    ]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for pattern in header_patterns:
            match = re.match(pattern, line)
            if match:
                headers.append({
                    "line_number": i,
                    "header": line,
                    "type": "section" if "section" in line.lower() else "general"
                })
                break
    
    return headers

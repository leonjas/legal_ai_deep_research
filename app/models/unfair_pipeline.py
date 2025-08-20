from typing import List, Dict, Optional, Union
import os
import logging
from pathlib import Path
from dataclasses import dataclass

from app.utils.pdf_parser import extract_text
from app.utils.clause_split import split_into_clauses
from app.models.unfair_model import UnfairClauseModel
from app.models.contract_analyzer import UnfairClause

# Try to import settings, fallback to defaults if not available
try:
    from app.utils.settings import UNFAIR_MIN_CONF, MIN_CLAUSE_WORDS
except ImportError:
    UNFAIR_MIN_CONF = 0.60
    MIN_CLAUSE_WORDS = 4

# Import label normalization utilities
try:
    from app.utils.explain_labels import normalize_label, format_unfair_result, get_explanation_for_label
except ImportError:
    import logging
    logging.getLogger(__name__).warning("Label normalization utilities not available")
    def normalize_label(label): return label.lower()
    def format_unfair_result(pred): return pred
    def get_explanation_for_label(label): return {"explanation": "No explanation available", "severity": "medium"}

logger = logging.getLogger(__name__)

class UnfairDetectionResult:
    """Container for unfair clause detection results."""
    
    def __init__(self, file_path: str, total_clauses: int, unfair_clauses: List[UnfairClause], 
                 model_info: Dict, processing_time: float = 0.0):
        self.file_path = file_path
        self.total_clauses = total_clauses
        self.unfair_clauses = unfair_clauses
        self.model_info = model_info
        self.processing_time = processing_time
        
    def get_summary(self) -> Dict:
        """Get a summary of the detection results."""
        return {
            "file_name": Path(self.file_path).name,
            "total_clauses": self.total_clauses,
            "unfair_count": len(self.unfair_clauses),
            "unfair_percentage": (len(self.unfair_clauses) / max(self.total_clauses, 1)) * 100,
            "model_used": self.model_info.get("model_name", "unknown"),
            "processing_time": self.processing_time
        }
    
    def get_unfair_by_confidence(self, min_confidence: float = 0.8) -> List[Dict]:
        """Get unfair clauses above a specific confidence threshold."""
        return [clause for clause in self.unfair_clauses 
                if clause["confidence"] >= min_confidence]
    
    def get_unfair_by_category(self) -> Dict[str, List[Dict]]:
        """Group unfair clauses by category."""
        categories = {}
        for clause in self.unfair_clauses:
            category = clause.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(clause)
        return categories
    
    def get_processed_results(self) -> List[Dict]:
        """Get unfair clauses with normalized labels and explanations."""
        processed = []
        for clause in self.unfair_clauses:
            try:
                # Apply label normalization and formatting
                formatted = format_unfair_result(clause)
                processed.append(formatted)
            except:
                # Fallback for compatibility
                processed.append(clause)
        return processed

def run_unfair_pipeline(
    file_path: str,
    model_name: Optional[str] = None,
    min_conf: Optional[float] = None,
    min_words: Optional[int] = None
) -> UnfairDetectionResult:
    """
    Run the complete unfair clause detection pipeline on a PDF file.
    
    Args:
        file_path: Path to the PDF file to analyze
        model_name: HuggingFace model name (optional, uses default from settings)
        min_conf: Minimum confidence threshold for unfair classification
        min_words: Minimum words per clause
        
    Returns:
        UnfairDetectionResult containing all analysis results
        
    Raises:
        Exception: If file processing fails
    """
    import time
    start_time = time.time()
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Set defaults
    min_conf = min_conf or UNFAIR_MIN_CONF
    min_words = min_words or MIN_CLAUSE_WORDS
    
    logger.info(f"Starting unfair clause detection pipeline for: {file_path}")
    logger.info(f"Settings - min_conf: {min_conf}, min_words: {min_words}")
    
    try:
        # Step 1: Extract text from PDF
        logger.info("Extracting text from PDF...")
        text = extract_text(file_path)
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        # Step 2: Split into clauses
        logger.info("Splitting text into clauses...")
        clauses = split_into_clauses(text, min_words=min_words)
        
        if not clauses:
            logger.warning("No valid clauses found in document")
            return UnfairDetectionResult(
                file_path=file_path,
                total_clauses=0,
                unfair_clauses=[],  # Empty list of UnfairClause objects
                model_info={"model_name": "none"},
                processing_time=time.time() - start_time
            )
        
        logger.info(f"Found {len(clauses)} clauses for analysis")
        
        # Step 3: Initialize model and predict
        logger.info("Loading unfair clause detection model...")
        model = UnfairClauseModel(model_name=model_name)
        model_info = model.get_model_info()
        
        logger.info("Running unfair clause predictions...")
        predictions = model.predict(clauses)
        
        # Step 4: Filter and process unfair clauses
        unfair_clauses = []
        for i, pred in enumerate(predictions):
            # Use normalized label for filtering
            normalized_label = normalize_label(pred.get("label", ""))
            
            # Skip fair clauses
            if normalized_label == "fair":
                continue
                
            # Check confidence threshold
            confidence = pred.get("confidence", 0.0)
            if confidence >= min_conf:
                try:
                    # Get explanation for the label
                    explanation_info = get_explanation_for_label(normalized_label)
                    
                    # Create UnfairClause object
                    unfair_clause = UnfairClause(
                        text=pred.get("clause", ""),
                        clause_type=normalized_label,
                        confidence=confidence,
                        explanation=explanation_info.get("explanation", "Detected as potentially unfair"),
                        severity=explanation_info.get("severity", "medium"),
                        sentence_index=i,
                        start_position=0,
                        end_position=len(pred.get("clause", ""))
                    )
                    unfair_clauses.append(unfair_clause)
                except Exception as e:
                    logger.warning(f"Error processing clause {i}: {e}")
                    # Fallback UnfairClause
                    unfair_clause = UnfairClause(
                        text=pred.get("clause", ""),
                        clause_type=pred.get("label", "unfair"),
                        confidence=confidence,
                        explanation="Detected as potentially unfair by ML model",
                        severity="medium",
                        sentence_index=i,
                        start_position=0,
                        end_position=len(pred.get("clause", ""))
                    )
                    unfair_clauses.append(unfair_clause)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Pipeline completed in {processing_time:.2f}s")
        logger.info(f"Found {len(unfair_clauses)} unfair clauses out of {len(clauses)} total")
        
        return UnfairDetectionResult(
            file_path=file_path,
            total_clauses=len(clauses),
            unfair_clauses=unfair_clauses,
            model_info=model_info,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in unfair detection pipeline: {e}")
        raise Exception(f"Pipeline failed: {e}")

def run_unfair_pipeline_text(
    text: str,
    model_name: Optional[str] = None,
    min_conf: Optional[float] = None,
    min_words: Optional[int] = None
) -> Dict:
    """
    Run unfair clause detection on raw text instead of PDF.
    
    Args:
        text: Contract text to analyze
        model_name: HuggingFace model name (optional)
        min_conf: Minimum confidence threshold
        min_words: Minimum words per clause
        
    Returns:
        Dictionary with analysis results
    """
    import time
    start_time = time.time()
    
    # Set defaults
    min_conf = min_conf or UNFAIR_MIN_CONF
    min_words = min_words or MIN_CLAUSE_WORDS
    
    logger.info("Starting unfair clause detection on text input")
    
    try:
        # Split into clauses
        clauses = split_into_clauses(text, min_words=min_words)
        
        if not clauses:
            return {
                "total_clauses": 0,
                "unfair_clauses": [],
                "summary": "No valid clauses found in text",
                "processing_time": time.time() - start_time
            }
        
        # Initialize model and predict
        model = UnfairClauseModel(model_name=model_name)
        predictions = model.predict(clauses)
        
        # Filter and process unfair clauses
        unfair_clauses = []
        for pred in predictions:
            normalized_label = normalize_label(pred.get("label", ""))
            
            if normalized_label != "fair" and pred.get("confidence", 0.0) >= min_conf:
                try:
                    formatted_pred = format_unfair_result(pred)
                    unfair_clauses.append(formatted_pred)
                except:
                    unfair_clauses.append(pred)
        
        processing_time = time.time() - start_time
        
        return {
            "total_clauses": len(clauses),
            "unfair_clauses": unfair_clauses,
            "all_predictions": predictions,
            "model_info": model.get_model_info(),
            "processing_time": processing_time,
            "summary": f"Found {len(unfair_clauses)} unfair clauses out of {len(clauses)} total"
        }
        
    except Exception as e:
        logger.error(f"Error in text-based unfair detection: {e}")
        raise Exception(f"Text analysis failed: {e}")

def batch_analyze_contracts(
    file_paths: List[str],
    model_name: Optional[str] = None,
    min_conf: Optional[float] = None
) -> List[UnfairDetectionResult]:
    """
    Analyze multiple contract files in batch.
    
    Args:
        file_paths: List of PDF file paths to analyze
        model_name: HuggingFace model name (optional)
        min_conf: Minimum confidence threshold
        
    Returns:
        List of UnfairDetectionResult objects
    """
    results = []
    
    # Initialize model once for all files
    model = UnfairClauseModel(model_name=model_name)
    min_conf = min_conf or UNFAIR_MIN_CONF
    
    logger.info(f"Starting batch analysis of {len(file_paths)} files")
    
    for i, file_path in enumerate(file_paths, 1):
        logger.info(f"Processing file {i}/{len(file_paths)}: {Path(file_path).name}")
        
        try:
            result = run_unfair_pipeline(file_path, model_name=None, min_conf=min_conf)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            # Add a failed result
            results.append(UnfairDetectionResult(
                file_path=file_path,
                total_clauses=0,
                unfair_clauses=[],
                model_info={"error": str(e)},
                processing_time=0.0
            ))
    
    logger.info(f"Batch analysis completed. {len(results)} files processed")
    return results

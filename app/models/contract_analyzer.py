"""
Contract Analysis Model for Unfair Clause Detection
Implements LexGLUE-based unfair clause detection in contracts
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import spacy
import nltk
from typing import List, Dict, Tuple, Optional
import re
import logging
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

@dataclass
class UnfairClause:
    """Data class for detected unfair clauses"""
    text: str
    clause_type: str
    confidence: float
    explanation: str
    severity: str  # 'high', 'medium', 'low'
    sentence_index: int
    start_position: int
    end_position: int

@dataclass
class ContractAnalysisResult:
    """Data class for contract analysis results"""
    total_sentences: int
    unfair_clauses: List[UnfairClause]
    overall_risk_score: float
    summary: str
    recommendations: List[str]

class ContractAnalyzer:
    """
    Main contract analyzer class for detecting unfair clauses using LexGLUE methodology
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._load_models()
        self._load_unfair_clause_patterns()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
    
    def _load_models(self):
        """Load pre-trained models and tokenizers"""
        try:
            # Use the pre-fine-tuned UNFAIR-ToS model from Hugging Face for LexGLUE unfair clause detection
            # This model is specifically trained for unfair clause detection in Terms of Service
            model_name = "marmolpen3/lexglue-unfair-tos"
            
            self.logger.info(f"Loading UNFAIR-ToS model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Print label mapping to understand the model's output format
            if hasattr(self.model.config, 'id2label'):
                self.logger.info(f"Model label mapping: {self.model.config.id2label}")
                self.label_mapping = self.model.config.id2label
            else:
                # Default binary classification labels if not specified
                self.label_mapping = {0: "fair", 1: "unfair"}
                self.logger.info("Using default binary labels: {0: 'fair', 1: 'unfair'}")
            
            # Move model to device
            self.model.to(self.device)
            
            # Create classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load spaCy model for NLP processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.info("spaCy model not found. Downloading en_core_web_sm...")
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
                    self.logger.info("Successfully downloaded and loaded spaCy model")
                except Exception as e:
                    self.logger.warning(f"Failed to download spaCy model: {e}")
                    self.nlp = None
                
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            # Fallback to a simpler model
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models if primary models fail"""
        try:
            self.logger.info("Loading fallback models...")
            
            # Try alternative UNFAIR-ToS models first
            fallback_models = [
                "CodeHima/TOSRobertaV2",  # Alternative UNFAIR-ToS model
                "nlpaueb/legal-bert-base-uncased",  # General legal BERT
                "bert-base-uncased"  # Basic BERT
            ]
            
            for model_name in fallback_models:
                try:
                    self.logger.info(f"Trying fallback model: {model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    
                    # Check for label mapping
                    if hasattr(self.model.config, 'id2label'):
                        self.label_mapping = self.model.config.id2label
                        self.logger.info(f"Fallback model label mapping: {self.label_mapping}")
                    else:
                        self.label_mapping = {0: "fair", 1: "unfair"}
                    
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # Use CPU
            )
            
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    self.nlp = None
            except:
                self.nlp = None
                
        except Exception as e:
            self.logger.error(f"Error loading fallback models: {e}")
            self.classifier = None
            self.sentence_model = None
            self.nlp = None
    
    def _load_unfair_clause_patterns(self):
        """Load patterns and indicators for different types of unfair clauses"""
        self.unfair_patterns = {
            "unilateral_termination": {
                "keywords": [
                    "terminate at any time", "terminate without cause", "terminate immediately",
                    "sole discretion", "without notice", "terminate for any reason",
                    "cancel unilaterally", "end this agreement at will"
                ],
                "patterns": [
                    r"(?i)\b(?:may|can|shall)\s+terminate\s+(?:this\s+)?(?:agreement|contract)\s+(?:at\s+any\s+time|without\s+cause)",
                    r"(?i)\bsole\s+discretion\b.*\bterminate\b",
                    r"(?i)\bterminate\b.*\bwithout\s+(?:notice|warning|cause)\b"
                ],
                "explanation": "Allows one party to terminate the contract unilaterally without justification",
                "severity": "high"
            },
            "automatic_renewal": {
                "keywords": [
                    "automatically renew", "auto-renewal", "automatically extend",
                    "unless cancelled", "continue indefinitely", "evergreen clause",
                    "self-renewing", "automatic extension"
                ],
                "patterns": [
                    r"(?i)\b(?:automatically|auto)\s+(?:renew|extend|continue)\b",
                    r"(?i)\bevergreen\s+clause\b",
                    r"(?i)\bunless\s+(?:cancelled|terminated)\s+by\b.*\bautomatically\b"
                ],
                "explanation": "Contract automatically renews without explicit consent",
                "severity": "medium"
            },
            "broad_indemnification": {
                "keywords": [
                    "indemnify and hold harmless", "defend and indemnify", "unlimited liability",
                    "all claims", "any and all damages", "regardless of cause",
                    "broad indemnification", "comprehensive indemnity"
                ],
                "patterns": [
                    r"(?i)\bindemnify\s+and\s+hold\s+harmless\b",
                    r"(?i)\bany\s+and\s+all\s+(?:claims|damages|losses)\b",
                    r"(?i)\bunlimited\s+liability\b",
                    r"(?i)\bregardless\s+of\s+(?:cause|fault|negligence)\b"
                ],
                "explanation": "Requires broad indemnification that may be unreasonable",
                "severity": "high"
            },
            "limitation_of_liability": {
                "keywords": [
                    "exclude all warranties", "no warranties", "as is basis",
                    "disclaim all liability", "maximum liability", "in no event",
                    "excluding liability", "limitation of damages"
                ],
                "patterns": [
                    r"(?i)\b(?:exclude|disclaim)\s+all\s+(?:warranties|liability)\b",
                    r"(?i)\bas\s+is\s+basis\b",
                    r"(?i)\bin\s+no\s+event\s+shall\b.*\bliable\b",
                    r"(?i)\bmaximum\s+liability\b.*\bshall\s+not\s+exceed\b"
                ],
                "explanation": "Unfairly limits liability or disclaims warranties",
                "severity": "medium"
            },
            "unreasonable_penalties": {
                "keywords": [
                    "liquidated damages", "penalty clause", "substantial penalty",
                    "forfeit all", "lose all rights", "immediate payment",
                    "punitive damages", "excessive fees"
                ],
                "patterns": [
                    r"(?i)\bliquidated\s+damages\b",
                    r"(?i)\bpenalty\s+(?:of|clause)\b",
                    r"(?i)\bforfeit\s+all\b",
                    r"(?i)\bexcessive\s+(?:fees|charges|penalties)\b"
                ],
                "explanation": "Contains unreasonable penalties or liquidated damages",
                "severity": "high"
            },
            "mandatory_arbitration": {
                "keywords": [
                    "binding arbitration", "waive right to jury trial", "arbitration only",
                    "no class action", "individual arbitration", "waive class action",
                    "exclusive arbitration", "mandatory arbitration"
                ],
                "patterns": [
                    r"(?i)\bbinding\s+arbitration\b",
                    r"(?i)\bwaive\s+(?:right\s+to\s+)?(?:jury\s+trial|class\s+action)\b",
                    r"(?i)\b(?:exclusive|mandatory)\s+arbitration\b",
                    r"(?i)\bno\s+class\s+action\b"
                ],
                "explanation": "Requires mandatory arbitration and may waive important legal rights",
                "severity": "medium"
            }
        }
    
    def segment_contract(self, text: str) -> List[str]:
        """
        Segment contract text into sentences for analysis
        
        Args:
            text: Raw contract text
            
        Returns:
            List of sentences
        """
        try:
            if self.nlp:
                # Use spaCy for better sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Fallback to NLTK
                sentences = nltk.sent_tokenize(text)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            return sentences
            
        except Exception as e:
            self.logger.error(f"Error in sentence segmentation: {e}")
            # Simple fallback
            sentences = text.split('.')
            return [s.strip() + '.' for s in sentences if s.strip()]
    
    def detect_pattern_based_clauses(self, sentences: List[str]) -> List[UnfairClause]:
        """
        Detect unfair clauses using pattern matching
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of detected unfair clauses
        """
        detected_clauses = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            for clause_type, patterns in self.unfair_patterns.items():
                # Check keywords
                keyword_matches = sum(1 for keyword in patterns["keywords"] 
                                    if keyword.lower() in sentence_lower)
                
                # Check regex patterns
                pattern_matches = sum(1 for pattern in patterns["patterns"] 
                                    if re.search(pattern, sentence))
                
                # Calculate confidence based on matches
                total_patterns = len(patterns["keywords"]) + len(patterns["patterns"])
                confidence = (keyword_matches + pattern_matches * 2) / (total_patterns + len(patterns["patterns"]))
                
                if confidence > 0.3:  # Threshold for detection
                    unfair_clause = UnfairClause(
                        text=sentence,
                        clause_type=clause_type,
                        confidence=min(confidence, 1.0),
                        explanation=patterns["explanation"],
                        severity=patterns["severity"],
                        sentence_index=i,
                        start_position=0,  # Would need more sophisticated position tracking
                        end_position=len(sentence)
                    )
                    detected_clauses.append(unfair_clause)
        
        return detected_clauses
    
    def ml_based_detection(self, sentences: List[str]) -> List[UnfairClause]:
        """
        Use machine learning models for unfair clause detection
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of detected unfair clauses
        """
        detected_clauses = []
        
        if not self.classifier:
            self.logger.warning("ML classifier not available, skipping ML-based detection")
            return detected_clauses
        
        try:
            for i, sentence in enumerate(sentences):
                # Skip very short sentences
                if len(sentence.split()) < 5:
                    continue
                
                # Get prediction
                result = self.classifier(sentence)
                
                # Handle the model output based on label mapping
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    predicted_label = prediction.get('label', '').lower()
                    confidence_score = prediction.get('score', 0)
                    
                    # Check if classified as unfair based on the model's label mapping
                    # Common patterns: "unfair", "clearly_unfair", "potentially_unfair", "LABEL_1"
                    is_unfair = (
                        'unfair' in predicted_label or 
                        predicted_label in ['label_1', '1'] or
                        (hasattr(self, 'label_mapping') and 
                         any('unfair' in str(label).lower() for label in self.label_mapping.values() 
                             if prediction.get('label') == f"LABEL_{idx}" for idx in self.label_mapping.keys()))
                    )
                    
                    if is_unfair and confidence_score > 0.7:
                        
                        unfair_clause = UnfairClause(
                            text=sentence,
                            clause_type="ml_detected",
                            confidence=prediction.get('score', 0),
                            explanation="Detected as potentially unfair by ML model",
                            severity="medium",
                            sentence_index=i,
                            start_position=0,
                            end_position=len(sentence)
                        )
                        detected_clauses.append(unfair_clause)
                        
        except Exception as e:
            self.logger.error(f"Error in ML-based detection: {e}")
        
        return detected_clauses
    
    def analyze_contract(self, contract_text: str) -> ContractAnalysisResult:
        """
        Perform comprehensive analysis of a contract
        
        Args:
            contract_text: Raw contract text
            
        Returns:
            ContractAnalysisResult with detected unfair clauses and analysis
        """
        self.logger.info("Starting contract analysis...")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(contract_text)
        
        # Segment into sentences
        sentences = self.segment_contract(cleaned_text)
        self.logger.info(f"Segmented contract into {len(sentences)} sentences")
        
        # Detect unfair clauses using different methods
        pattern_clauses = self.detect_pattern_based_clauses(sentences)
        ml_clauses = self.ml_based_detection(sentences)
        
        # Combine and deduplicate results
        all_clauses = self._merge_detections(pattern_clauses, ml_clauses)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(all_clauses, len(sentences))
        
        # Generate summary and recommendations
        summary = self._generate_summary(all_clauses, risk_score)
        recommendations = self._generate_recommendations(all_clauses)
        
        result = ContractAnalysisResult(
            total_sentences=len(sentences),
            unfair_clauses=all_clauses,
            overall_risk_score=risk_score,
            summary=summary,
            recommendations=recommendations
        )
        
        self.logger.info(f"Analysis complete. Found {len(all_clauses)} potentially unfair clauses")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess contract text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers, headers, footers (basic patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text.strip()
    
    def _merge_detections(self, pattern_clauses: List[UnfairClause], 
                         ml_clauses: List[UnfairClause]) -> List[UnfairClause]:
        """Merge and deduplicate detections from different methods"""
        all_clauses = pattern_clauses + ml_clauses
        
        # Simple deduplication based on sentence similarity
        unique_clauses = []
        for clause in all_clauses:
            is_duplicate = False
            for existing in unique_clauses:
                if (existing.sentence_index == clause.sentence_index or 
                    self._text_similarity(existing.text, clause.text) > 0.9):
                    # Keep the one with higher confidence
                    if clause.confidence > existing.confidence:
                        unique_clauses.remove(existing)
                        unique_clauses.append(clause)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_clauses.append(clause)
        
        return unique_clauses
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not self.sentence_model:
            # Simple fallback
            return 1.0 if text1.lower() == text2.lower() else 0.0
        
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_risk_score(self, clauses: List[UnfairClause], total_sentences: int) -> float:
        """Calculate overall contract risk score"""
        if not clauses:
            return 0.0
        
        # Weight by severity
        severity_weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
        
        total_weight = sum(
            clause.confidence * severity_weights.get(clause.severity, 0.5)
            for clause in clauses
        )
        
        # Normalize by contract length
        risk_score = min(total_weight / max(total_sentences * 0.1, 1), 1.0)
        
        return risk_score
    
    def _generate_summary(self, clauses: List[UnfairClause], risk_score: float) -> str:
        """Generate analysis summary"""
        if not clauses:
            return "No potentially unfair clauses detected in this contract."
        
        clause_types = set(clause.clause_type for clause in clauses)
        high_severity = sum(1 for clause in clauses if clause.severity == "high")
        
        risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
        
        summary = f"""
        Contract Risk Assessment: {risk_level} Risk (Score: {risk_score:.2f})
        
        Found {len(clauses)} potentially unfair clauses across {len(clause_types)} categories.
        {high_severity} clauses are classified as high severity.
        
        Main concerns identified: {', '.join(clause_types)}
        """
        
        return summary.strip()
    
    def _generate_recommendations(self, clauses: List[UnfairClause]) -> List[str]:
        """Generate recommendations based on detected clauses"""
        recommendations = []
        
        clause_types = set(clause.clause_type for clause in clauses)
        
        type_recommendations = {
            "unilateral_termination": "Consider negotiating mutual termination clauses with reasonable notice periods",
            "automatic_renewal": "Request explicit consent requirements for contract renewals",
            "broad_indemnification": "Negotiate limited indemnification with specific exclusions",
            "limitation_of_liability": "Seek balanced liability limitations that protect both parties",
            "unreasonable_penalties": "Review penalty clauses for reasonableness and proportionality",
            "mandatory_arbitration": "Consider implications of waiving jury trial rights"
        }
        
        for clause_type in clause_types:
            if clause_type in type_recommendations:
                recommendations.append(type_recommendations[clause_type])
        
        # General recommendations
        if len(clauses) > 5:
            recommendations.append("Consider comprehensive legal review due to multiple potential issues")
        
        if any(clause.severity == "high" for clause in clauses):
            recommendations.append("Prioritize addressing high-severity clauses before signing")
        
        return recommendations

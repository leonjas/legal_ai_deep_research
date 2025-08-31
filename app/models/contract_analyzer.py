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
from typing import Any
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
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
        
        # Setup NLTK with error handling
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK with proper error handling"""
        if not NLTK_AVAILABLE:
            logging.info("NLTK not available, using fallback methods")
            return
            
        try:
            # Try to find the tokenizer
            try:
                nltk.data.find('tokenizers/punkt_tab')
                logging.info("NLTK punkt_tab tokenizer found")
            except (LookupError, OSError):
                try:
                    logging.info("Downloading punkt tokenizers...")
                    nltk.download('punkt', quiet=True)
                    nltk.download('punkt_tab', quiet=True)
                except Exception as e:
                    logging.warning(f"Could not download NLTK data: {e}")
        except Exception as e:
            logging.warning(f"NLTK setup failed, will use fallback: {e}")

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
            
            # Load spaCy model for NLP processing (optional)
            if SPACY_AVAILABLE and spacy:
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
            else:
                self.logger.info("spaCy not available, will use fallback methods")
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
                if SPACY_AVAILABLE and spacy:
                    self.nlp = spacy.load("en_core_web_sm")
                else:
                    self.nlp = None
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
            if self.nlp and SPACY_AVAILABLE:
                # Use spaCy for better sentence segmentation
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            elif NLTK_AVAILABLE and nltk:
                # Fallback to NLTK
                sentences = nltk.sent_tokenize(text)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                # Simple fallback
                sentences = text.split('.')
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
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
                         any('unfair' in str(self.label_mapping.get(idx, '')).lower() 
                             for idx in self.label_mapping.keys() 
                             if prediction.get('label') == f"LABEL_{idx}"))
                    )
                    
                    # Lower confidence threshold and add pattern-based detection
                    if (is_unfair and confidence_score > 0.5) or self._is_unfair_by_pattern(sentence):
                        
                        # Determine clause type and severity
                        clause_type, severity = self._classify_unfair_clause(sentence)
                        
                        unfair_clause = UnfairClause(
                            text=sentence,
                            clause_type=clause_type,
                            confidence=max(prediction.get('score', 0), 0.6 if self._is_unfair_by_pattern(sentence) else 0),
                            explanation=f"Detected as {clause_type} clause",
                            severity=severity,
                            sentence_index=i,
                            start_position=0,
                            end_position=len(sentence)
                        )
                        detected_clauses.append(unfair_clause)
                        
        except Exception as e:
            self.logger.error(f"Error in ML-based detection: {e}")
        
        return detected_clauses
    
    def _is_unfair_by_pattern(self, sentence: str) -> bool:
        """Pattern-based unfair clause detection"""
        sentence_lower = sentence.lower()
        
        # Unfair termination patterns
        termination_patterns = [
            "terminate.*without cause", "terminate.*without notice", 
            "terminate.*at any time", "terminate.*sole discretion"
        ]
        
        # Liability limitation patterns  
        liability_patterns = [
            "not liable", "disclaim.*warranty", "exclude.*warranty",
            "limitation of liability", "waive.*right"
        ]
        
        # Indemnification patterns
        indemnification_patterns = [
            "indemnify.*hold harmless", "indemnify.*defend",
            "indemnify.*against.*all.*claims"
        ]
        
        # Modification patterns
        modification_patterns = [
            "modify.*at.*discretion", "change.*terms.*time",
            "reserve.*right.*modify"
        ]
        
        # Automatic renewal patterns
        renewal_patterns = [
            "automatically renew", "auto.*renewal",
            "renew.*successive.*period"
        ]
        
        # Data usage patterns
        data_patterns = [
            "perpetual.*license", "irrevocable.*license",
            "worldwide.*license.*use.*modify"
        ]
        
        all_patterns = (termination_patterns + liability_patterns + 
                       indemnification_patterns + modification_patterns +
                       renewal_patterns + data_patterns)
        
        import re
        for pattern in all_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _classify_unfair_clause(self, sentence: str) -> tuple:
        """Classify the type and severity of an unfair clause"""
        sentence_lower = sentence.lower()
        
        # High severity patterns
        if any(pattern in sentence_lower for pattern in [
            "terminate.*without cause", "terminate.*without notice",
            "indemnify.*all.*claims", "waive.*right.*jury",
            "exclude.*all.*warranties"
        ]):
            if "terminate" in sentence_lower:
                return "unfair_termination", "high"
            elif "indemnify" in sentence_lower:
                return "broad_indemnification", "high"
            elif "waive" in sentence_lower:
                return "rights_waiver", "high"
            else:
                return "warranty_exclusion", "high"
        
        # Medium severity patterns
        elif any(pattern in sentence_lower for pattern in [
            "limitation of liability", "automatically renew",
            "modify.*discretion", "perpetual.*license"
        ]):
            if "liability" in sentence_lower:
                return "liability_limitation", "medium"
            elif "renew" in sentence_lower:
                return "automatic_renewal", "medium"
            elif "modify" in sentence_lower:
                return "unilateral_modification", "medium"
            else:
                return "excessive_data_rights", "medium"
        
        # Default classification
        return "potentially_unfair", "medium"
    
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

    def generate_enhanced_contract_summary(self, contract_text: str, unfair_clauses: List[UnfairClause], risk_score: float) -> Dict[str, Any]:
        """
        Generate enhanced analytical contract summary with risk prioritization and cross-references
        
        Args:
            contract_text: Full contract text
            unfair_clauses: List of detected unfair clauses
            risk_score: Overall risk score
            
        Returns:
            Dictionary with comprehensive summary components
        """
        
        # Extract structured sections
        sections = self._extract_contract_sections(contract_text)
        
        # Analyze clause relationships
        clause_relationships = self._analyze_clause_relationships(unfair_clauses)
        
        # Generate risk prioritization
        risk_analysis = self._generate_risk_prioritization(unfair_clauses, risk_score)
        
        # Create actionable insights
        actionable_insights = self._generate_actionable_insights(unfair_clauses, sections)
        
        return {
            "executive_summary": self._create_executive_summary(risk_score, unfair_clauses, sections),
            "sections_analysis": sections,
            "risk_prioritization": risk_analysis,
            "clause_relationships": clause_relationships,
            "actionable_insights": actionable_insights,
            "negotiation_strategy": self._generate_negotiation_strategy(unfair_clauses),
            "compliance_concerns": self._identify_compliance_concerns(unfair_clauses)
        }
    
    def _extract_contract_sections(self, contract_text: str) -> Dict[str, Dict]:
        """Extract and analyze key contract sections"""
        
        sections = {
            "service_terms": {"content": "", "risk_indicators": [], "key_points": []},
            "payment_terms": {"content": "", "risk_indicators": [], "key_points": []},
            "termination": {"content": "", "risk_indicators": [], "key_points": []},
            "liability": {"content": "", "risk_indicators": [], "key_points": []},
            "data_privacy": {"content": "", "risk_indicators": [], "key_points": []},
            "intellectual_property": {"content": "", "risk_indicators": [], "key_points": []},
            "dispute_resolution": {"content": "", "risk_indicators": [], "key_points": []}
        }
        
        # Section identification patterns
        section_patterns = {
            "service_terms": [r"service\s+terms", r"scope\s+of\s+services", r"software\s+license"],
            "payment_terms": [r"payment", r"billing", r"fees", r"subscription"],
            "termination": [r"termination", r"cancellation", r"end\s+of\s+agreement"],
            "liability": [r"liability", r"limitation", r"damages", r"warranties"],
            "data_privacy": [r"data", r"privacy", r"personal\s+information", r"confidential"],
            "intellectual_property": [r"intellectual\s+property", r"copyright", r"trademark", r"ownership"],
            "dispute_resolution": [r"dispute", r"arbitration", r"governing\s+law", r"jurisdiction"]
        }
        
        paragraphs = [p.strip() for p in contract_text.split('\n\n') if len(p.strip()) > 50]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Classify paragraph by section
            for section_name, patterns in section_patterns.items():
                if any(re.search(pattern, paragraph_lower) for pattern in patterns):
                    if not sections[section_name]["content"]:
                        sections[section_name]["content"] = paragraph[:500]
                    
                    # Identify risk indicators
                    risk_indicators = self._identify_section_risks(paragraph, section_name)
                    sections[section_name]["risk_indicators"].extend(risk_indicators)
                    
                    # Extract key points
                    key_points = self._extract_key_points(paragraph, section_name)
                    sections[section_name]["key_points"].extend(key_points)
                    break
        
        return sections
    
    def _identify_section_risks(self, paragraph: str, section_type: str) -> List[str]:
        """Identify specific risks within a contract section"""
        
        risk_patterns = {
            "service_terms": [
                ("unilateral changes", r"(?i)(?:may|can)\s+(?:modify|change|alter)\s+(?:at\s+any\s+time|without\s+notice)"),
                ("service limitations", r"(?i)(?:may|can)\s+(?:suspend|discontinue|limit)\s+(?:at\s+any\s+time|without\s+notice)"),
                ("broad discretion", r"(?i)(?:sole\s+discretion|absolute\s+discretion)")
            ],
            "payment_terms": [
                ("automatic renewal", r"(?i)automatically\s+(?:renew|extend|continue)"),
                ("penalty fees", r"(?i)(?:penalty|fine|charge).*\$[0-9,]+"),
                ("price changes", r"(?i)(?:increase|change|modify)\s+(?:fees|price|cost)")
            ],
            "termination": [
                ("immediate termination", r"(?i)terminate\s+(?:immediately|at\s+any\s+time|without\s+notice)"),
                ("no refund", r"(?i)no\s+refund"),
                ("data deletion", r"(?i)(?:delete|remove|destroy)\s+(?:all\s+)?data")
            ],
            "liability": [
                ("liability exclusion", r"(?i)(?:exclude|disclaim|limit)\s+(?:all\s+)?(?:liability|warranties)"),
                ("consequential damages", r"(?i)no\s+liability\s+for\s+(?:consequential|indirect|incidental)"),
                ("maximum liability", r"(?i)liability\s+(?:limited\s+to|shall\s+not\s+exceed)")
            ]
        }
        
        risks = []
        if section_type in risk_patterns:
            for risk_name, pattern in risk_patterns[section_type]:
                if re.search(pattern, paragraph):
                    risks.append(risk_name)
        
        return risks
    
    def _extract_key_points(self, paragraph: str, section_type: str) -> List[str]:
        """Extract key contractual points from a section"""
        
        # Split into sentences and find important ones
        sentences = [s.strip() for s in paragraph.split('.') if len(s.strip()) > 20]
        
        key_indicators = {
            "service_terms": ["license", "access", "use", "restrictions", "permissions"],
            "payment_terms": ["due", "billing", "subscription", "refund", "cancellation"],
            "termination": ["terminate", "end", "cancel", "notice", "effect"],
            "liability": ["liable", "responsible", "damages", "limit", "exclude"],
            "data_privacy": ["collect", "use", "share", "process", "store"],
            "intellectual_property": ["own", "license", "copyright", "trademark", "rights"]
        }
        
        key_points = []
        indicators = key_indicators.get(section_type, [])
        
        for sentence in sentences[:3]:  # Take top 3 sentences
            if any(indicator in sentence.lower() for indicator in indicators):
                key_points.append(sentence[:150] + "..." if len(sentence) > 150 else sentence)
        
        return key_points
    
    def _analyze_clause_relationships(self, unfair_clauses: List[UnfairClause]) -> Dict[str, List[str]]:
        """Analyze relationships and interactions between unfair clauses"""
        
        relationships = {
            "reinforcing_clauses": [],
            "conflicting_clauses": [],
            "escalating_risks": []
        }
        
        clause_types = [clause.clause_type for clause in unfair_clauses]
        
        # Define clause relationships
        reinforcing_combinations = [
            (["unilateral_termination", "broad_indemnification"], "Termination + indemnification creates one-sided risk"),
            (["limitation_of_liability", "mandatory_arbitration"], "Limited liability + forced arbitration reduces recourse"),
            (["automatic_renewal", "unreasonable_penalties"], "Auto-renewal + penalties trap customers"),
            (["unilateral_modification", "broad_indemnification"], "Modification rights + indemnification shift all risk")
        ]
        
        for combination, description in reinforcing_combinations:
            if all(clause_type in clause_types for clause_type in combination):
                relationships["reinforcing_clauses"].append(description)
        
        # Identify escalating risks
        if len(unfair_clauses) >= 5:
            relationships["escalating_risks"].append("Multiple unfair clauses create compound risk exposure")
        
        high_severity_count = sum(1 for clause in unfair_clauses if clause.severity == "high")
        if high_severity_count >= 3:
            relationships["escalating_risks"].append("Concentration of high-severity clauses indicates systematic unfairness")
        
        return relationships
    
    def _generate_risk_prioritization(self, unfair_clauses: List[UnfairClause], risk_score: float) -> Dict[str, Any]:
        """Generate prioritized risk assessment"""
        
        # Group clauses by severity and impact
        high_priority = [c for c in unfair_clauses if c.severity == "high"]
        medium_priority = [c for c in unfair_clauses if c.severity == "medium"]
        low_priority = [c for c in unfair_clauses if c.severity == "low"]
        
        # Calculate business impact
        business_impact = {
            "financial_risk": 0,
            "operational_risk": 0,
            "legal_risk": 0,
            "reputational_risk": 0
        }
        
        impact_mapping = {
            "unilateral_termination": {"operational_risk": 0.8, "financial_risk": 0.6},
            "broad_indemnification": {"legal_risk": 0.9, "financial_risk": 0.8},
            "limitation_of_liability": {"legal_risk": 0.7, "financial_risk": 0.6},
            "automatic_renewal": {"financial_risk": 0.5, "operational_risk": 0.3},
            "unreasonable_penalties": {"financial_risk": 0.7},
            "mandatory_arbitration": {"legal_risk": 0.6}
        }
        
        for clause in unfair_clauses:
            if clause.clause_type in impact_mapping:
                for risk_type, impact in impact_mapping[clause.clause_type].items():
                    business_impact[risk_type] += impact
        
        # Normalize scores
        max_impact = max(business_impact.values()) if business_impact.values() else 1
        if max_impact > 0:
            business_impact = {k: v/max_impact for k, v in business_impact.items()}
        
        return {
            "overall_risk_level": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low",
            "risk_score": risk_score,
            "high_priority_clauses": len(high_priority),
            "medium_priority_clauses": len(medium_priority),
            "low_priority_clauses": len(low_priority),
            "business_impact": business_impact,
            "immediate_action_required": len(high_priority) > 0 or risk_score > 0.7
        }
    
    def _generate_actionable_insights(self, unfair_clauses: List[UnfairClause], sections: Dict) -> List[Dict[str, str]]:
        """Generate specific, actionable insights for contract negotiation"""
        
        insights = []
        
        # Section-specific insights
        for section_name, section_data in sections.items():
            if section_data["risk_indicators"]:
                insight = {
                    "section": section_name.replace('_', ' ').title(),
                    "risk_level": "High" if len(section_data["risk_indicators"]) >= 2 else "Medium",
                    "specific_concern": f"Found {len(section_data['risk_indicators'])} risk indicators: {', '.join(section_data['risk_indicators'])}",
                    "recommended_action": self._get_section_recommendation(section_name, section_data["risk_indicators"])
                }
                insights.append(insight)
        
        # Clause-specific insights
        clause_groups = {}
        for clause in unfair_clauses:
            if clause.clause_type not in clause_groups:
                clause_groups[clause.clause_type] = []
            clause_groups[clause.clause_type].append(clause)
        
        for clause_type, clauses in clause_groups.items():
            if len(clauses) > 1:
                insight = {
                    "section": "Multiple Instances",
                    "risk_level": "High",
                    "specific_concern": f"Found {len(clauses)} instances of {clause_type.replace('_', ' ')} clauses",
                    "recommended_action": f"Address all {len(clauses)} instances collectively in negotiation"
                }
                insights.append(insight)
        
        return insights
    
    def _get_section_recommendation(self, section_name: str, risk_indicators: List[str]) -> str:
        """Get specific recommendation for a contract section"""
        
        recommendations = {
            "service_terms": "Negotiate mutual modification rights and service level guarantees",
            "payment_terms": "Request transparent pricing and reasonable cancellation terms",
            "termination": "Negotiate mutual termination rights with reasonable notice periods",
            "liability": "Seek balanced liability allocation with appropriate exclusions",
            "data_privacy": "Ensure compliance with data protection regulations and user rights",
            "intellectual_property": "Clarify ownership rights and usage permissions",
            "dispute_resolution": "Consider implications of mandatory arbitration clauses"
        }
        
        return recommendations.get(section_name, "Conduct detailed legal review of this section")
    
    def _generate_negotiation_strategy(self, unfair_clauses: List[UnfairClause]) -> Dict[str, Any]:
        """Generate strategic approach for contract negotiation"""
        
        strategy = {
            "primary_objectives": [],
            "negotiation_priorities": [],
            "fallback_positions": [],
            "deal_breakers": []
        }
        
        high_severity_clauses = [c for c in unfair_clauses if c.severity == "high"]
        
        # Primary objectives
        if high_severity_clauses:
            strategy["primary_objectives"].append("Remove or significantly modify high-severity unfair clauses")
        
        if len(unfair_clauses) >= 5:
            strategy["primary_objectives"].append("Comprehensive clause review and rebalancing")
        
        # Negotiation priorities (ranked)
        clause_priority = {
            "broad_indemnification": 1,
            "unilateral_termination": 2,
            "limitation_of_liability": 3,
            "mandatory_arbitration": 4,
            "automatic_renewal": 5,
            "unreasonable_penalties": 6
        }
        
        priorities = sorted(
            [(clause.clause_type, clause_priority.get(clause.clause_type, 10)) for clause in unfair_clauses],
            key=lambda x: x[1]
        )
        
        strategy["negotiation_priorities"] = [
            f"{i+1}. Address {clause_type.replace('_', ' ')} provisions"
            for i, (clause_type, _) in enumerate(priorities[:5])
        ]
        
        # Deal breakers
        if any(c.clause_type == "broad_indemnification" and c.severity == "high" for c in unfair_clauses):
            strategy["deal_breakers"].append("Unlimited indemnification liability")
        
        if len([c for c in unfair_clauses if c.severity == "high"]) >= 3:
            strategy["deal_breakers"].append("Concentration of high-severity unfair terms")
        
        return strategy
    
    def _identify_compliance_concerns(self, unfair_clauses: List[UnfairClause]) -> List[Dict[str, str]]:
        """Identify potential regulatory compliance concerns"""
        
        concerns = []
        
        compliance_mapping = {
            "automatic_renewal": {
                "regulation": "Consumer Protection Laws",
                "concern": "Auto-renewal clauses may violate state consumer protection requirements",
                "jurisdiction": "State-level (varies by state)"
            },
            "broad_indemnification": {
                "regulation": "Unconscionability Doctrine",
                "concern": "Overly broad indemnification may be deemed unconscionable",
                "jurisdiction": "Federal and State Courts"
            },
            "mandatory_arbitration": {
                "regulation": "Consumer Arbitration Rules",
                "concern": "Arbitration clauses in consumer contracts face increasing scrutiny",
                "jurisdiction": "Federal (CFPB) and State"
            },
            "limitation_of_liability": {
                "regulation": "UCC and Consumer Protection",
                "concern": "Complete liability limitations may violate consumer protection standards",
                "jurisdiction": "State Commercial Law"
            }
        }
        
        for clause in unfair_clauses:
            if clause.clause_type in compliance_mapping:
                concern_info = compliance_mapping[clause.clause_type]
                concerns.append({
                    "clause_type": clause.clause_type.replace('_', ' ').title(),
                    "regulation": concern_info["regulation"],
                    "concern": concern_info["concern"],
                    "jurisdiction": concern_info["jurisdiction"],
                    "severity": clause.severity
                })
        
        return concerns
    
    def _create_executive_summary(self, risk_score: float, unfair_clauses: List[UnfairClause], sections: Dict) -> str:
        """Create executive summary of contract analysis"""
        
        risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"
        high_risk_clauses = len([c for c in unfair_clauses if c.severity == "high"])
        
        # Identify most problematic sections
        problematic_sections = [
            name.replace('_', ' ').title() 
            for name, data in sections.items() 
            if len(data.get("risk_indicators", [])) >= 2
        ]
        
        summary = f"""
EXECUTIVE SUMMARY

Overall Risk Assessment: {risk_level} Risk (Score: {risk_score:.1%})

Key Findings:
• {len(unfair_clauses)} potentially unfair clauses identified
• {high_risk_clauses} high-severity clauses requiring immediate attention
• {len(problematic_sections)} contract sections with significant risk indicators

Most Concerning Areas: {', '.join(problematic_sections[:3]) if problematic_sections else 'No major section-level risks identified'}

Recommendation: {'NEGOTIATE BEFORE SIGNING' if risk_score > 0.7 or high_risk_clauses > 0 else 'PROCEED WITH CAUTION' if risk_score > 0.3 else 'ACCEPTABLE RISK LEVEL'}
        """
        
        return summary.strip()

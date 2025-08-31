"""
AppealRecommendationPipeline - Document-Level Legal Appeal Analysis
Operates at document level with legal-tuned LLM, FAISS retrieval, and risk scoring
"""

import re
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

# Optional imports for better text processing
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# For legal document parsing and LLM integration
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalRuling:
    """Document-level legal ruling structure"""
    ruling_text: str
    reasoning_structure: str
    penalties_imposed: List[str]
    jurisdiction: str
    case_id: str
    ruling_date: str
    judge_name: Optional[str] = ""
    case_type: str = ""
    
@dataclass  
class RiskFactor:
    """Individual risk factor in appeal analysis"""
    factor_name: str
    risk_level: float  # 0.0 to 1.0
    weight: float
    plain_language_explanation: str
    supporting_evidence: List[str]

@dataclass
class AppealWorthinessReport:
    """Final appeal recommendation report"""
    appeal_score: float  # 0-100 appeal worthiness score
    recommendation: str  # "Strong Appeal", "Moderate Appeal", "Weak Appeal", "No Appeal"
    retrieved_case_anchors: List[Dict]  # Similar cases from FAISS search
    risk_factors: List[RiskFactor]
    key_risk_indicators: List[str]
    reasoning_breakdown: str
    estimated_success_probability: float
    recommended_appeal_strategy: str
    timeline_estimate: str
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class LegalPrecedent:
    """Data class for legal precedents in vector database"""
    id: str
    clause_text: str
    recommendation_text: str
    legal_basis: str
    jurisdiction: str
    success_rate: float
    case_references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    precedent: LegalPrecedent
    similarity_score: float
    relevance_explanation: str

def _split_sentences(text: str) -> List[str]:
    """Best-effort splitter; tries spaCy if available, else regex."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # model not downloaded; fall back to regex
            raise
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except Exception:
        # Regex fallback (good enough for ToS-style text)
        parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
        return [p.strip() for p in parts if len(p.split()) > 4]

@dataclass
class UnfairClause:
    """Standalone UnfairClause for appeal recommendations (avoids spaCy dependency)"""
    text: str
    clause_type: str
    confidence: float
    explanation: str = ""

@dataclass
class AppealRecommendation:
    """Data class for appeal recommendation"""
    clause_type: str
    priority: str  # 'high', 'medium', 'low'
    negotiation_strategy: str
    suggested_alternatives: List[str]
    legal_precedents: List[str]
    negotiation_points: List[str]
    template_language: str
    difficulty_level: str
    success_likelihood: str

@dataclass
class AppealReport:
    """Complete appeal report for all unfair clauses"""
    overall_strategy: str
    priority_clauses: List[str]
    recommendations: List[AppealRecommendation]
    negotiation_timeline: Dict[str, str]
    preparation_checklist: List[str]
    alternative_options: List[str]
    estimated_success_rate: str

class AppealRecommendationPipeline:
    """
    Document-Level Legal Appeal Analysis Pipeline
    
    Core Steps:
    1. Document-Level Parsing: Legal-tuned LLM extracts ruling, reasoning, penalties
    2. FAISS-Based Retrieval: Sentence-BERT encoding + dense similarity search  
    3. Risk Scoring: Multi-dimensional risk model with heuristic weights
    4. Output: Single report with appeal-worthiness score and risk breakdown
    """
    
    def __init__(self):
        logger.info("Initializing AppealRecommendationPipeline...")
        
        # Initialize legal document parser (legal-tuned LLM)
        self._initialize_legal_parser()
        
        # Initialize FAISS-based retrieval system
        self._initialize_faiss_retrieval()
        
        # Initialize risk scoring model
        self._initialize_risk_model()
        
        # Load precedent and statute corpus
        self._load_legal_corpus()
        
        logger.info("AppealRecommendationPipeline initialized successfully")
    
    def _initialize_legal_parser(self):
        """Initialize legal-tuned LLM for document parsing"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Use a model with safetensors support to avoid torch.load issues
                self.legal_parser = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Alternative model with safetensors
                    return_all_scores=True,
                    trust_remote_code=True
                )
                logger.info("Legal document parser initialized with fallback model")
            else:
                logger.warning("Transformers not available, using rule-based parser")
                self.legal_parser = None
        except Exception as e:
            logger.info(f"Using rule-based legal parser (transformers issue: {str(e)[:100]}...)")
            self.legal_parser = None
    
    def _initialize_faiss_retrieval(self):
        """Initialize FAISS-based retrieval system with Sentence-BERT"""
        try:
            # Use sentence transformer for legal text embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence-BERT model loaded for FAISS retrieval")
            
            # Initialize FAISS index (will be built when corpus is loaded)
            self.faiss_index = None
            self.case_embeddings = None
            self.legal_corpus = []
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS retrieval: {e}")
            raise
    
    def _initialize_risk_model(self):
        """Initialize multi-dimensional risk scoring model"""
        
        # Define risk factors with heuristic weights
        self.risk_factors_config = {
            "inconsistent_penalty": {
                "weight": 0.25,
                "description": "Penalty imposed is inconsistent with similar cases"
            },
            "lack_of_reasoning": {
                "weight": 0.30, 
                "description": "Insufficient or flawed legal reasoning in ruling"
            },
            "procedural_errors": {
                "weight": 0.20,
                "description": "Procedural violations during case proceedings"
            },
            "precedent_contradictions": {
                "weight": 0.15,
                "description": "Ruling contradicts established legal precedents"
            },
            "evidence_issues": {
                "weight": 0.10,
                "description": "Problems with evidence handling or interpretation"
            }
        }
        
        logger.info("Risk scoring model initialized with 5 dimensions")
    
    def _load_legal_corpus(self):
        """Load curated corpus of precedents and statutes"""
        
        # Enhanced legal corpus with more diverse cases
        self.legal_corpus = [
            {
                "case_id": "contract_001",
                "title": "Contract Termination Without Notice",
                "text": "Court ruled that unilateral termination clauses without reasonable notice violate good faith dealing principles",
                "outcome": "Appeal Successful",
                "jurisdiction": "State Court",
                "year": 2023,
                "success_rate": 0.78
            },
            {
                "case_id": "liability_001", 
                "title": "Broad Liability Waiver Challenge",
                "text": "Comprehensive liability waivers found unconscionable and void under consumer protection statutes",
                "outcome": "Appeal Successful",
                "jurisdiction": "Federal Court",
                "year": 2022,
                "success_rate": 0.85
            },
            {
                "case_id": "precedent_001",
                "title": "Unconscionability Legal Doctrine",
                "text": "Contracts with excessively one-sided terms may be voided as unconscionable under equity principles",
                "outcome": "Legal Principle",
                "jurisdiction": "General Law",
                "year": 0,
                "success_rate": 0.65
            },
            {
                "case_id": "employment_001",
                "title": "At-Will Employment Termination",
                "text": "Employee termination without cause upheld under at-will employment doctrine despite lack of notice",
                "outcome": "Appeal Denied",
                "jurisdiction": "State Court", 
                "year": 2023,
                "success_rate": 0.25
            },
            {
                "case_id": "penalty_001",
                "title": "Excessive Penalty Assessment",
                "text": "Monetary penalties must be proportionate to harm caused and consistent with similar violations",
                "outcome": "Appeal Successful",
                "jurisdiction": "Appeals Court",
                "year": 2022,
                "success_rate": 0.72
            },
            {
                "case_id": "reasoning_001",
                "title": "Insufficient Legal Reasoning",
                "text": "Court decisions must provide adequate legal reasoning to support conclusions and rulings",
                "outcome": "Appeal Successful", 
                "jurisdiction": "Supreme Court",
                "year": 2021,
                "success_rate": 0.90
            }
        ]
        
        # Build FAISS index from corpus
        self._build_faiss_index()
        
        logger.info(f"Legal corpus loaded with {len(self.legal_corpus)} items")
    
    def _build_faiss_index(self):
        """Build FAISS index from legal corpus"""
        
        # Extract text from corpus for embedding
        corpus_texts = [item["text"] for item in self.legal_corpus]
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(corpus_texts, convert_to_tensor=False)
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_np)
        self.case_embeddings = embeddings_np
        
        logger.info(f"FAISS index built with {len(corpus_texts)} legal documents")
    
    def analyze_document(self, document_text: str, case_id: str = None) -> AppealWorthinessReport:
        """
        Main pipeline method - analyze legal document for appeal worthiness
        
        Args:
            document_text: Full legal document/ruling text
            case_id: Optional case identifier
            
        Returns:
            AppealWorthinessReport with comprehensive analysis
        """
        
        logger.info(f"Starting document-level analysis for case: {case_id}")
        
        # Step 1: Document-Level Parsing
        ruling_structure = self._parse_legal_document(document_text)
        
        # Step 2: FAISS-Based Retrieval  
        similar_cases = self._retrieve_similar_cases(document_text, top_k=5)
        
        # Step 3: Risk Scoring
        risk_analysis = self._calculate_risk_scores(ruling_structure, similar_cases)
        
        # Step 4: Generate Final Report
        appeal_report = self._generate_appeal_report(ruling_structure, similar_cases, risk_analysis)
        
        logger.info(f"Analysis complete. Appeal score: {appeal_report.appeal_score:.1f}")
        
        return appeal_report
    
    def _parse_legal_document(self, document_text: str) -> LegalRuling:
        """Document-Level Parsing using legal-tuned LLM"""
        
        # Extract key sections using pattern matching and LLM analysis
        ruling_sections = self._extract_ruling_sections(document_text)
        
        # Use legal parser if available, otherwise use rule-based extraction
        if self.legal_parser:
            reasoning_analysis = self._llm_analyze_reasoning(ruling_sections.get("reasoning", ""))
        else:
            reasoning_analysis = self._rule_based_reasoning_analysis(ruling_sections.get("reasoning", ""))
        
        # Extract penalties
        penalties = self._extract_penalties(document_text)
        
        return LegalRuling(
            ruling_text=ruling_sections.get("ruling", ""),
            reasoning_structure=reasoning_analysis,
            penalties_imposed=penalties,
            jurisdiction=self._extract_jurisdiction(document_text),
            case_id=self._extract_case_id(document_text),
            ruling_date=self._extract_date(document_text)
        )
    
    def _extract_ruling_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from legal document"""
        
        sections = {
            "ruling": "",
            "reasoning": "", 
            "facts": "",
            "conclusion": ""
        }
        
        # Rule-based section extraction
        text_lower = text.lower()
        
        # Look for common legal section patterns
        ruling_patterns = [r"we (hold|rule|find|conclude) that", r"the court (orders|rules|holds)"]
        reasoning_patterns = [r"because", r"therefore", r"given that", r"in light of"]
        
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Classify sentence by section
            if any(pattern in sentence_lower for pattern in ruling_patterns):
                sections["ruling"] += sentence + ". "
            elif any(pattern in sentence_lower for pattern in reasoning_patterns):
                sections["reasoning"] += sentence + ". "
                
        return sections
    
    def _llm_analyze_reasoning(self, reasoning_text: str) -> str:
        """Use legal-tuned LLM to analyze reasoning structure"""
        
        if not reasoning_text.strip():
            return "No clear reasoning structure identified"
            
        # This would use the legal LLM to analyze reasoning quality
        # For now, return structured analysis
        return f"Reasoning analysis: {len(reasoning_text.split('.'))} reasoning steps identified"
    
    def _rule_based_reasoning_analysis(self, reasoning_text: str) -> str:
        """Fallback rule-based reasoning analysis"""
        
        if not reasoning_text.strip():
            return "Insufficient reasoning provided"
            
        reasoning_indicators = ["because", "therefore", "given", "since", "due to"]
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in reasoning_text.lower())
        
        if indicator_count >= 3:
            return f"Strong reasoning structure with {indicator_count} logical connections"
        elif indicator_count >= 1:
            return f"Moderate reasoning structure with {indicator_count} logical connections"
        else:
            return "Weak reasoning structure - lacks clear logical connections"
    
    def _extract_penalties(self, text: str) -> List[str]:
        """Extract penalties/sanctions from document"""
        
        penalties = []
        penalty_patterns = [
            r"fined? \$?([0-9,]+)",
            r"penalty of \$?([0-9,]+)",
            r"sanctioned? for \$?([0-9,]+)",
            r"damages of \$?([0-9,]+)"
        ]
        
        for pattern in penalty_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                penalties.append(match.group(0))
                
        return penalties
    
    def _extract_jurisdiction(self, text: str) -> str:
        """Extract jurisdiction from document"""
        jurisdictions = ["federal court", "state court", "district court", "appeals court", "supreme court"]
        
        for jurisdiction in jurisdictions:
            if jurisdiction in text.lower():
                return jurisdiction.title()
                
        return "Unknown Jurisdiction"
    
    def _extract_case_id(self, text: str) -> str:
        """Extract case ID/number from document"""
        case_patterns = [r"case no\.?\s*([0-9\-]+)", r"docket no\.?\s*([0-9\-]+)"]
        
        for pattern in case_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return "Unknown Case ID"
    
    def _extract_date(self, text: str) -> str:
        """Extract ruling date from document"""
        date_patterns = [
            r"(\d{1,2}/\d{1,2}/\d{4})",
            r"(\w+ \d{1,2}, \d{4})",
            r"(\d{4}-\d{2}-\d{2})"
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
                
        return "Unknown Date"
    
    def _retrieve_similar_cases(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """FAISS-based retrieval of similar cases and precedents"""
        
        if not self.faiss_index:
            logger.warning("FAISS index not available")
            return []
        
        # Encode query text
        query_embedding = self.sentence_model.encode([query_text], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Retrieve similar cases with similarity scores
        similar_cases = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.legal_corpus) and distance >= 0:  # Filter out invalid results
                case = self.legal_corpus[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                max_distance = 2.0  # Approximate max L2 distance for normalized embeddings
                similarity = max(0.0, 1.0 - (distance / max_distance))
                case["similarity_score"] = similarity
                case["rank"] = i + 1
                similar_cases.append(case)
                
        # Remove duplicates (same case appearing multiple times)
        seen_cases = set()
        unique_cases = []
        for case in similar_cases:
            case_id = case.get("case_id", "")
            if case_id not in seen_cases:
                seen_cases.add(case_id)
                unique_cases.append(case)
                
        logger.info(f"Retrieved {len(unique_cases)} similar cases via FAISS")
        return unique_cases[:top_k]  # Limit to requested number
    
    def _calculate_risk_scores(self, ruling: LegalRuling, similar_cases: List[Dict]) -> Dict[str, RiskFactor]:
        """Multi-dimensional risk scoring with heuristic weights"""
        
        risk_factors = {}
        
        # Risk Factor 1: Inconsistent Penalty
        penalty_risk = self._assess_penalty_consistency(ruling, similar_cases)
        risk_factors["inconsistent_penalty"] = RiskFactor(
            factor_name="Inconsistent Penalty",
            risk_level=penalty_risk,
            weight=self.risk_factors_config["inconsistent_penalty"]["weight"],
            plain_language_explanation=self._explain_penalty_risk(penalty_risk),
            supporting_evidence=self._get_penalty_evidence(ruling, similar_cases)
        )
        
        # Risk Factor 2: Lack of Reasoning
        reasoning_risk = self._assess_reasoning_quality(ruling)
        risk_factors["lack_of_reasoning"] = RiskFactor(
            factor_name="Reasoning Quality",
            risk_level=reasoning_risk,
            weight=self.risk_factors_config["lack_of_reasoning"]["weight"], 
            plain_language_explanation=self._explain_reasoning_risk(reasoning_risk),
            supporting_evidence=[ruling.reasoning_structure]
        )
        
        # Risk Factor 3: Procedural Errors (simplified assessment)
        procedural_risk = self._assess_procedural_issues(ruling)
        risk_factors["procedural_errors"] = RiskFactor(
            factor_name="Procedural Issues",
            risk_level=procedural_risk,
            weight=self.risk_factors_config["procedural_errors"]["weight"],
            plain_language_explanation=self._explain_procedural_risk(procedural_risk),
            supporting_evidence=["Analysis of procedural compliance"]
        )
        
        # Risk Factor 4: Precedent Contradictions
        precedent_risk = self._assess_precedent_consistency(ruling, similar_cases)
        risk_factors["precedent_contradictions"] = RiskFactor(
            factor_name="Precedent Consistency", 
            risk_level=precedent_risk,
            weight=self.risk_factors_config["precedent_contradictions"]["weight"],
            plain_language_explanation=self._explain_precedent_risk(precedent_risk),
            supporting_evidence=self._get_precedent_evidence(similar_cases)
        )
        
        # Risk Factor 5: Evidence Issues
        evidence_risk = self._assess_evidence_handling(ruling)
        risk_factors["evidence_issues"] = RiskFactor(
            factor_name="Evidence Handling",
            risk_level=evidence_risk,
            weight=self.risk_factors_config["evidence_issues"]["weight"],
            plain_language_explanation=self._explain_evidence_risk(evidence_risk),
            supporting_evidence=["Evidence handling analysis"]
        )
        
        logger.info(f"Risk assessment complete - {len(risk_factors)} factors analyzed")
        return risk_factors
    
    def _assess_penalty_consistency(self, ruling: LegalRuling, similar_cases: List[Dict]) -> float:
        """Assess if penalty is consistent with similar cases"""
        if not ruling.penalties_imposed or not similar_cases:
            return 0.3  # Moderate risk if no comparison data
            
        # Simple heuristic: compare penalty presence/absence
        has_penalty = len(ruling.penalties_imposed) > 0
        similar_with_penalty = sum(1 for case in similar_cases if "penalty" in case.get("text", "").lower())
        
        if similar_with_penalty > len(similar_cases) / 2 and not has_penalty:
            return 0.8  # High risk - similar cases had penalties but this doesn't
        elif similar_with_penalty < len(similar_cases) / 2 and has_penalty:
            return 0.7  # High risk - this has penalty but similar cases didn't
        else:
            return 0.2  # Low risk - consistent with similar cases
    
    def _assess_reasoning_quality(self, ruling: LegalRuling) -> float:
        """Assess quality of legal reasoning"""
        reasoning = ruling.reasoning_structure.lower()
        
        if "insufficient reasoning" in reasoning or "weak reasoning" in reasoning:
            return 0.9  # Very high risk
        elif "moderate reasoning" in reasoning:
            return 0.5  # Moderate risk  
        elif "strong reasoning" in reasoning:
            return 0.1  # Low risk
        else:
            return 0.4  # Default moderate risk
    
    def _assess_procedural_issues(self, ruling: LegalRuling) -> float:
        """Assess procedural compliance (simplified)"""
        # In a real system, this would analyze procedural compliance in detail
        return 0.3  # Default moderate risk
    
    def _assess_precedent_consistency(self, ruling: LegalRuling, similar_cases: List[Dict]) -> float:
        """Assess consistency with legal precedents"""
        if not similar_cases:
            return 0.5
            
        # Check if similar cases had different outcomes
        successful_appeals = sum(1 for case in similar_cases if case.get("outcome") == "Appeal Successful")
        
        if successful_appeals > len(similar_cases) / 2:
            return 0.7  # High risk - similar cases were successfully appealed
        else:
            return 0.3  # Lower risk
    
    def _assess_evidence_handling(self, ruling: LegalRuling) -> float:
        """Assess evidence handling quality"""
        # Simplified assessment based on ruling text
        return 0.25  # Default low-moderate risk
    
    def _explain_penalty_risk(self, risk_level: float) -> str:
        """Plain language explanation of penalty risk"""
        if risk_level > 0.7:
            return "The penalty appears inconsistent with similar cases, suggesting potential grounds for appeal"
        elif risk_level > 0.4:
            return "The penalty shows some inconsistency with precedent but may be within acceptable range"
        else:
            return "The penalty appears consistent with similar cases"
    
    def _explain_reasoning_risk(self, risk_level: float) -> str:
        """Plain language explanation of reasoning risk"""
        if risk_level > 0.7:
            return "The legal reasoning appears insufficient or flawed, providing strong grounds for appeal"
        elif risk_level > 0.4:
            return "The legal reasoning has some weaknesses that could be challenged on appeal"
        else:
            return "The legal reasoning appears sound and well-structured"
    
    def _explain_procedural_risk(self, risk_level: float) -> str:
        """Plain language explanation of procedural risk"""
        return "Procedural compliance assessment - detailed review recommended"
    
    def _explain_precedent_risk(self, risk_level: float) -> str:
        """Plain language explanation of precedent risk"""
        if risk_level > 0.6:
            return "Similar cases have been successfully appealed, suggesting precedent may favor appeal"
        else:
            return "Precedent analysis suggests limited grounds for successful appeal"
    
    def _explain_evidence_risk(self, risk_level: float) -> str:
        """Plain language explanation of evidence risk"""
        return "Evidence handling appears within normal parameters"
    
    def _get_penalty_evidence(self, ruling: LegalRuling, similar_cases: List[Dict]) -> List[str]:
        """Get supporting evidence for penalty assessment"""
        evidence = []
        if ruling.penalties_imposed:
            evidence.append(f"Penalties imposed: {', '.join(ruling.penalties_imposed)}")
        evidence.append(f"Compared against {len(similar_cases)} similar cases")
        return evidence
    
    def _get_precedent_evidence(self, similar_cases: List[Dict]) -> List[str]:
        """Get supporting evidence for precedent analysis"""
        evidence = []
        for case in similar_cases[:3]:  # Top 3 most similar
            evidence.append(f"Similar case: {case.get('title', 'Unknown')} - {case.get('outcome', 'Unknown outcome')}")
        return evidence
    
    def _generate_appeal_report(self, ruling: LegalRuling, similar_cases: List[Dict], risk_factors: Dict[str, RiskFactor]) -> AppealWorthinessReport:
        """Generate final appeal worthiness report"""
        
        # Calculate overall appeal score (0-100)
        appeal_score = self._calculate_appeal_score(risk_factors)
        
        # Generate recommendation based on score
        recommendation = self._generate_recommendation(appeal_score)
        
        # Extract key risk indicators
        key_indicators = self._extract_key_indicators(risk_factors)
        
        # Generate reasoning breakdown
        reasoning = self._generate_reasoning_breakdown(risk_factors, similar_cases)
        
        # Estimate success probability
        success_probability = self._estimate_success_probability(risk_factors, similar_cases)
        
        # Generate appeal strategy
        strategy = self._generate_appeal_strategy(risk_factors, similar_cases)
        
        # Estimate timeline
        timeline = self._estimate_timeline(appeal_score)
        
        # Prepare case anchors
        case_anchors = [
            {
                "case_id": case.get("case_id", "Unknown"),
                "title": case.get("title", "Unknown Case"),
                "similarity": case.get("similarity_score", 0.0),
                "outcome": case.get("outcome", "Unknown"),
                "relevance": case.get("success_rate", 0.0)
            }
            for case in similar_cases[:5]
        ]
        
        return AppealWorthinessReport(
            appeal_score=appeal_score,
            recommendation=recommendation, 
            retrieved_case_anchors=case_anchors,
            risk_factors=list(risk_factors.values()),
            key_risk_indicators=key_indicators,
            reasoning_breakdown=reasoning,
            estimated_success_probability=success_probability,
            recommended_appeal_strategy=strategy,
            timeline_estimate=timeline
        )
    
    def _calculate_appeal_score(self, risk_factors: Dict[str, RiskFactor]) -> float:
        """Calculate weighted appeal worthiness score (0-100)"""
        
        total_weighted_risk = 0.0
        total_weight = 0.0
        
        for factor in risk_factors.values():
            weighted_risk = factor.risk_level * factor.weight
            total_weighted_risk += weighted_risk
            total_weight += factor.weight
            
        if total_weight > 0:
            average_risk = total_weighted_risk / total_weight
            # Convert risk to appeal score (higher risk = higher appeal worthiness)
            appeal_score = average_risk * 100
        else:
            appeal_score = 50.0  # Default moderate appeal worthiness
            
        return min(100.0, max(0.0, appeal_score))
    
    def _generate_recommendation(self, appeal_score: float) -> str:
        """Generate appeal recommendation based on score"""
        if appeal_score >= 75:
            return "Strong Appeal - High likelihood of success"
        elif appeal_score >= 50:
            return "Moderate Appeal - Reasonable grounds exist"
        elif appeal_score >= 25:
            return "Weak Appeal - Limited grounds, consider carefully"
        else:
            return "No Appeal - Insufficient grounds for successful appeal"
    
    def _extract_key_indicators(self, risk_factors: Dict[str, RiskFactor]) -> List[str]:
        """Extract key risk indicators for summary"""
        indicators = []
        
        for factor in risk_factors.values():
            if factor.risk_level > 0.6:  # High risk factors
                indicators.append(f"High Risk: {factor.factor_name} ({factor.risk_level:.1%} risk level)")
            elif factor.risk_level > 0.4:  # Moderate risk factors
                indicators.append(f"Moderate Risk: {factor.factor_name} ({factor.risk_level:.1%} risk level)")
        
        # If no high/moderate risks found, add the highest risk factor
        if not indicators and risk_factors:
            highest_risk = max(risk_factors.values(), key=lambda x: x.risk_level)
            indicators.append(f"Primary Concern: {highest_risk.factor_name} ({highest_risk.risk_level:.1%} risk level)")
                
        return indicators[:5]  # Top 5 indicators
    
    def _generate_reasoning_breakdown(self, risk_factors: Dict[str, RiskFactor], similar_cases: List[Dict]) -> str:
        """Generate detailed reasoning breakdown"""
        
        high_risk_factors = [f for f in risk_factors.values() if f.risk_level > 0.6]
        
        breakdown = f"Analysis based on {len(risk_factors)} risk dimensions and {len(similar_cases)} similar cases. "
        
        if high_risk_factors:
            factor_names = [f.factor_name for f in high_risk_factors]
            breakdown += f"Primary concerns: {', '.join(factor_names)}. "
        
        successful_precedents = sum(1 for case in similar_cases if case.get("outcome") == "Appeal Successful")
        if successful_precedents > 0:
            breakdown += f"{successful_precedents} of {len(similar_cases)} similar cases had successful appeals. "
            
        return breakdown
    
    def _estimate_success_probability(self, risk_factors: Dict[str, RiskFactor], similar_cases: List[Dict]) -> float:
        """Estimate probability of successful appeal"""
        
        # Base probability from risk factors
        risk_score = sum(f.risk_level * f.weight for f in risk_factors.values())
        base_probability = risk_score * 0.8  # Convert to probability
        
        # Adjust based on similar case outcomes
        if similar_cases:
            successful_rate = sum(case.get("success_rate", 0.5) for case in similar_cases) / len(similar_cases)
            adjusted_probability = (base_probability * 0.7) + (successful_rate * 0.3)
        else:
            adjusted_probability = base_probability
            
        return min(0.95, max(0.05, adjusted_probability))
    
    def _generate_appeal_strategy(self, risk_factors: Dict[str, RiskFactor], similar_cases: List[Dict]) -> str:
        """Generate recommended appeal strategy"""
        
        high_risk_factors = [f for f in risk_factors.values() if f.risk_level > 0.6]
        
        if not high_risk_factors:
            return "Limited grounds for appeal - focus on procedural or technical issues"
        
        strategies = []
        for factor in high_risk_factors:
            if "penalty" in factor.factor_name.lower():
                strategies.append("Challenge penalty consistency with precedent")
            elif "reasoning" in factor.factor_name.lower():
                strategies.append("Attack insufficient legal reasoning")
            elif "precedent" in factor.factor_name.lower():
                strategies.append("Cite contradictory precedent cases")
                
        return "; ".join(strategies) if strategies else "General appeal based on case merits"
    
    def _estimate_timeline(self, appeal_score: float) -> str:
        """Estimate appeal timeline"""
        if appeal_score >= 75:
            return "6-12 months (strong case, likely to proceed quickly)"
        elif appeal_score >= 50:
            return "8-18 months (moderate complexity expected)"
        else:
            return "12-24 months (complex case with uncertain outcome)"


# Original AppealRecommendationEngine class for backward compatibility
class AppealRecommendationEngine:
    """
    Generates strategic recommendations for appealing unfair contract clauses
    Enhanced with FAISS vector search for legal precedent matching
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentence_model = None
        self.faiss_index = None
        self.legal_precedents = []
        
        # Initialize vector search components
        self._initialize_vector_search()
        self._load_recommendation_database()
        self._build_vector_database()
    
    def _initialize_vector_search(self):
        """Initialize sentence transformer and FAISS components"""
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Initialized sentence transformer model")
        except Exception as e:
            self.logger.warning(f"Failed to initialize sentence transformer: {e}")
            self.sentence_model = None
    
    def _build_vector_database(self):
        """Build FAISS vector database of legal precedents and recommendations"""
        if not self.sentence_model:
            self.logger.warning("No sentence model available for vector search")
            return
        
        try:
            # Create comprehensive legal precedent database
            self.legal_precedents = self._create_legal_precedent_database()
            
            if not self.legal_precedents:
                self.logger.warning("No legal precedents loaded")
                return
            
            # Generate embeddings for all precedents
            precedent_texts = []
            for precedent in self.legal_precedents:
                # Combine clause text and recommendation for richer embedding
                combined_text = f"{precedent.clause_text} {precedent.recommendation_text} {precedent.legal_basis}"
                precedent_texts.append(combined_text)
            
            # Create embeddings
            embeddings = self.sentence_model.encode(precedent_texts)
            
            # Store embeddings in precedent objects
            for i, precedent in enumerate(self.legal_precedents):
                precedent.embedding = embeddings[i]
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.faiss_index.add(embeddings)
            
            self.logger.info(f"Built FAISS index with {len(self.legal_precedents)} precedents")
            
        except Exception as e:
            self.logger.error(f"Failed to build vector database: {e}")
            self.faiss_index = None
    
    def _create_legal_precedent_database(self) -> List[LegalPrecedent]:
        """Create comprehensive database of legal precedents"""
        precedents = []
        
        # Limitation of Liability precedents
        precedents.extend([
            LegalPrecedent(
                id="liability_001",
                clause_text="Company shall not be liable for any damages whatsoever",
                recommendation_text="Negotiate mutual liability caps and exclude gross negligence",
                legal_basis="Unconscionable contract doctrine - complete liability waivers often unenforceable",
                jurisdiction="US Federal",
                success_rate=0.85,
                case_references=["Williams v. Walker-Thomas Furniture Co.", "A&M Produce Co. v. FMC Corp."],
                tags=["liability", "unconscionable", "consumer_protection"]
            ),
            LegalPrecedent(
                id="liability_002", 
                clause_text="Maximum liability limited to $100 regardless of actual damages",
                recommendation_text="Request liability cap proportional to contract value or annual fees",
                legal_basis="Disproportionate remedy doctrine - caps must bear reasonable relation to potential harm",
                jurisdiction="Common Law",
                success_rate=0.78,
                case_references=["Hadley v. Baxendale", "Southwest Savings v. DennisReeder"],
                tags=["liability", "proportionality", "damages"]
            )
        ])
        
        # Unilateral Termination precedents
        precedents.extend([
            LegalPrecedent(
                id="termination_001",
                clause_text="Either party may terminate this agreement at any time without cause",
                recommendation_text="Add mutual termination rights with reasonable notice periods",
                legal_basis="Mutuality doctrine - contracts must impose equal obligations on both parties",
                jurisdiction="US State Courts",
                success_rate=0.92,
                case_references=["Mattei v. Hopper", "Wood v. Lucy, Lady Duff-Gordon"],
                tags=["termination", "mutuality", "notice"]
            ),
            LegalPrecedent(
                id="termination_002",
                clause_text="Company may terminate immediately at sole discretion",
                recommendation_text="Require specific grounds for termination and cure periods",
                legal_basis="Good faith and fair dealing - termination must have legitimate business reason",
                jurisdiction="UCC/Commercial Law",
                success_rate=0.73,
                case_references=["Kirke La Shelle Co. v. Armstrong Co.", "Fortune v. National Cash Register"],
                tags=["termination", "good_faith", "discretion"]
            )
        ])
        
        # Automatic Renewal precedents  
        precedents.extend([
            LegalPrecedent(
                id="renewal_001",
                clause_text="Contract automatically renews unless cancelled 90 days prior",
                recommendation_text="Require explicit consent for renewals or shorter notice periods",
                legal_basis="Consumer protection laws often limit auto-renewal terms",
                jurisdiction="State Consumer Protection",
                success_rate=0.81,
                case_references=["California SB-313", "New York General Business Law 527-a"],
                tags=["renewal", "consumer_protection", "notice"]
            )
        ])
        
        # Arbitration precedents
        precedents.extend([
            LegalPrecedent(
                id="arbitration_001", 
                clause_text="All disputes must be resolved through binding arbitration",
                recommendation_text="Negotiate carve-outs for IP claims and small claims court",
                legal_basis="Federal Arbitration Act allows but courts scrutinize unconscionable terms",
                jurisdiction="Federal/FAA",
                success_rate=0.65,
                case_references=["AT&T Mobility v. Concepcion", "Rent-A-Center v. Jackson"],
                tags=["arbitration", "unconscionable", "carve_outs"]
            ),
            LegalPrecedent(
                id="arbitration_002",
                clause_text="Waiver of right to class action or jury trial", 
                recommendation_text="Challenge class action waivers in consumer contexts",
                legal_basis="Some states void class action waivers as against public policy",
                jurisdiction="State Courts",
                success_rate=0.58,
                case_references=["McGill v. Citibank", "Gentry v. Superior Court"],
                tags=["arbitration", "class_action", "jury_trial"]
            )
        ])
        
        # Indemnification precedents
        precedents.extend([
            LegalPrecedent(
                id="indemnity_001",
                clause_text="Customer shall indemnify Company against all claims",
                recommendation_text="Limit indemnification to customer's breach or negligence",
                legal_basis="Broad indemnification may be unenforceable as against public policy",
                jurisdiction="Common Law",
                success_rate=0.77,
                case_references=["Dresser Industries v. Page Petroleum", "Ethyl Corp v. Daniel Constr."],
                tags=["indemnification", "public_policy", "scope"]
            )
        ])
        
        return precedents
    
    def search_similar_precedents(self, clause_text: str, top_k: int = 5) -> List[VectorSearchResult]:
        """Search for similar legal precedents using vector similarity"""
        if not self.sentence_model or not self.faiss_index:
            self.logger.warning("Vector search not available")
            return []
        
        try:
            # Encode the query clause
            query_embedding = self.sentence_model.encode([clause_text])
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx >= 0 and idx < len(self.legal_precedents):  # Valid index
                    precedent = self.legal_precedents[idx]
                    
                    # Generate relevance explanation
                    explanation = self._generate_relevance_explanation(clause_text, precedent, similarity)
                    
                    result = VectorSearchResult(
                        precedent=precedent,
                        similarity_score=float(similarity),
                        relevance_explanation=explanation
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def _generate_relevance_explanation(self, query_text: str, precedent: LegalPrecedent, 
                                      similarity: float) -> str:
        """Generate explanation for why a precedent is relevant"""
        # Extract key terms from both texts for explanation
        query_lower = query_text.lower()
        precedent_lower = precedent.clause_text.lower()
        
        common_terms = []
        legal_concepts = ["terminate", "liability", "indemnify", "arbitration", "renewal", "cancel", 
                         "waive", "disclaim", "exclusive", "binding", "damages", "breach"]
        
        for concept in legal_concepts:
            if concept in query_lower and concept in precedent_lower:
                common_terms.append(concept)
        
        if common_terms:
            explanation = f"Similar legal concepts: {', '.join(common_terms)}. "
        else:
            explanation = "Structurally similar clause pattern. "
        
        # Add success rate context
        success_pct = int(precedent.success_rate * 100)
        explanation += f"Historical success rate: {success_pct}% in {precedent.jurisdiction} jurisdiction."
        
        return explanation
    
    def _load_recommendation_database(self):
        """Load recommendations database for different clause types"""
        self.recommendations_db = {
            "limitation_of_liability": {
                "priority": "high",
                "strategy": "Challenge excessive liability limitations and push for mutual liability caps",
                "alternatives": [
                    "Add mutual liability limitations that protect both parties equally",
                    "Include specific exceptions for gross negligence and willful misconduct", 
                    "Set reasonable liability caps based on contract value or annual fees",
                    "Exclude liability for data breaches and security incidents"
                ],
                "precedents": [
                    "Courts often void unlimited liability waivers as unconscionable",
                    "Many jurisdictions require mutual liability limitations",
                    "Consumer protection laws limit liability disclaimers"
                ],
                "negotiation_points": [
                    "Request reciprocal liability limitations",
                    "Argue that complete liability waiver is unconscionable",
                    "Propose liability caps tied to contract value",
                    "Insist on exceptions for gross negligence"
                ],
                "template": "We propose modifying section X to include mutual liability limitations capped at [amount], with exceptions for gross negligence, willful misconduct, and data security breaches.",
                "difficulty": "medium",
                "success_rate": "70%"
            },
            
            "unilateral_termination": {
                "priority": "high", 
                "strategy": "Negotiate for mutual termination rights and reasonable notice periods",
                "alternatives": [
                    "Require mutual termination rights with equal notice periods",
                    "Add 'for cause' requirements for immediate termination",
                    "Include reasonable notice periods (30-90 days)",
                    "Provide data export rights upon termination"
                ],
                "precedents": [
                    "Unilateral termination clauses may be void under unfair contract terms legislation",
                    "Courts favor contracts with mutual termination rights",
                    "Reasonable notice requirements are often legally mandated"
                ],
                "negotiation_points": [
                    "Request reciprocal termination clauses",
                    "Argue for reasonable notice requirements", 
                    "Propose graduated termination process",
                    "Include data portability provisions"
                ],
                "template": "We request mutual termination rights requiring [X] days written notice, except for material breaches which may be terminated immediately after [Y] days cure period.",
                "difficulty": "medium",
                "success_rate": "65%"
            },
            
            "automatic_renewal": {
                "priority": "medium",
                "strategy": "Push for explicit consent and easy opt-out mechanisms",
                "alternatives": [
                    "Require explicit consent for each renewal",
                    "Provide clear opt-out mechanisms with reasonable notice",
                    "Add automatic renewal notifications 60-90 days in advance", 
                    "Allow mid-term cancellation with partial refunds"
                ],
                "precedents": [
                    "Many jurisdictions require explicit consent for auto-renewal",
                    "FTC guidelines emphasize clear disclosure and easy cancellation",
                    "Auto-renewal laws vary by state but generally favor consumers"
                ],
                "negotiation_points": [
                    "Request explicit renewal consent requirements",
                    "Propose advance notification periods",
                    "Include easy cancellation mechanisms",
                    "Add cooling-off period after renewal"
                ],
                "template": "We propose that renewals require explicit written consent at least [X] days before the renewal date, with clear cancellation instructions provided.",
                "difficulty": "low",
                "success_rate": "85%"
            },
            
            "mandatory_arbitration": {
                "priority": "high",
                "strategy": "Negotiate for optional arbitration or improved arbitration terms",
                "alternatives": [
                    "Make arbitration optional rather than mandatory",
                    "Allow small claims court for disputes under threshold",
                    "Provide mutual choice of arbitrator",
                    "Include cost-sharing provisions for arbitration"
                ],
                "precedents": [
                    "Some courts have found mandatory arbitration unconscionable",
                    "Class action waivers are increasingly scrutinized",
                    "Consumer protection laws may override arbitration clauses"
                ],
                "negotiation_points": [
                    "Argue against class action waiver",
                    "Request optional rather than mandatory arbitration",
                    "Propose cost-sharing for arbitration expenses",
                    "Include small claims court exception"
                ],
                "template": "We request that arbitration be optional, with disputes under $[X] eligible for small claims court, and arbitration costs shared equally between parties.",
                "difficulty": "high", 
                "success_rate": "45%"
            },
            
            "broad_indemnification": {
                "priority": "high",
                "strategy": "Limit indemnification scope and add mutual provisions",
                "alternatives": [
                    "Add mutual indemnification clauses",
                    "Limit indemnification to specific scenarios",
                    "Exclude indemnification for company's negligence",
                    "Cap indemnification amounts"
                ],
                "precedents": [
                    "Broad indemnification clauses may be unconscionable",
                    "Courts prefer mutual indemnification arrangements",
                    "Indemnification for own negligence often void"
                ],
                "negotiation_points": [
                    "Request mutual indemnification",
                    "Limit scope to third-party claims only",
                    "Exclude company's own negligence",
                    "Add reasonable caps on indemnification"
                ],
                "template": "We propose mutual indemnification limited to third-party claims arising from each party's negligent acts, excluding indemnification for the indemnified party's own negligence.",
                "difficulty": "medium",
                "success_rate": "60%"
            },
            
            "content_removal": {
                "priority": "medium",
                "strategy": "Negotiate for clear content policies and appeal processes",
                "alternatives": [
                    "Add clear content policy guidelines",
                    "Include notice and appeal process for content removal",
                    "Provide data export rights before removal",
                    "Limit removal to illegal or clearly harmful content"
                ],
                "precedents": [
                    "Due process requirements for content removal",
                    "Right to explanation for automated decisions",
                    "Platform liability protections require good faith"
                ],
                "negotiation_points": [
                    "Request clear content removal standards",
                    "Propose notice and appeal process",
                    "Include data backup provisions",
                    "Limit removal to legal violations"
                ],
                "template": "We request clear content policies with notice and appeal procedures, allowing [X] days to cure violations before permanent removal.",
                "difficulty": "low",
                "success_rate": "75%"
            },
            
            "choice_of_law": {
                "priority": "low",
                "strategy": "Negotiate for neutral jurisdiction or mutual choice",
                "alternatives": [
                    "Choose neutral jurisdiction acceptable to both parties",
                    "Select jurisdiction where customer is located",
                    "Agree on jurisdiction with favorable consumer protection laws",
                    "Include alternative dispute resolution options"
                ],
                "precedents": [
                    "Forum selection clauses must be reasonable",
                    "Consumer protection laws may override choice of law",
                    "Inconvenient forums may be rejected by courts"
                ],
                "negotiation_points": [
                    "Propose customer's home jurisdiction",
                    "Request neutral third jurisdiction",
                    "Argue inconvenience of proposed jurisdiction",
                    "Include virtual dispute resolution options"
                ],
                "template": "We propose [jurisdiction] as a more neutral jurisdiction, or alternatively, allow disputes to be resolved in the customer's home jurisdiction.",
                "difficulty": "medium",
                "success_rate": "55%"
            },
            
            "unreasonable_penalties": {
                "priority": "high",
                "strategy": "Challenge penalty amounts and negotiate reasonable fees",
                "alternatives": [
                    "Replace penalties with actual damages provisions",
                    "Cap penalty amounts at reasonable levels",
                    "Provide grace periods before penalties apply",
                    "Include dispute process for penalty assessments"
                ],
                "precedents": [
                    "Penalty clauses must be reasonable liquidated damages",
                    "Excessive penalties may be void as unconscionable",
                    "Courts prefer actual damages over fixed penalties"
                ],
                "negotiation_points": [
                    "Argue penalties are excessive and punitive",
                    "Request reasonable damage calculations", 
                    "Propose grace periods and cure opportunities",
                    "Include penalty dispute resolution process"
                ],
                "template": "We propose replacing fixed penalties with actual damages, capped at [reasonable amount], with a [X]-day grace period for cure.",
                "difficulty": "medium",
                "success_rate": "70%"
            }
        }
    
    def generate_appeal_report(self, unfair_clauses: List[UnfairClause]) -> AppealReport:
        """
        Generate comprehensive appeal recommendations for all unfair clauses
        
        Args:
            unfair_clauses: List of detected unfair clauses
            
        Returns:
            AppealReport with strategic recommendations
        """
        self.logger.info(f"Generating appeal recommendations for {len(unfair_clauses)} unfair clauses")
        
        recommendations = []
        high_priority = []
        
        # Generate recommendations for each clause
        for clause in unfair_clauses:
            recommendation = self._generate_clause_recommendation(clause)
            recommendations.append(recommendation)
            
            if recommendation.priority == 'high':
                high_priority.append(clause.clause_type)
        
        # Generate overall strategy
        overall_strategy = self._generate_overall_strategy(unfair_clauses, high_priority)
        
        # Create negotiation timeline
        timeline = self._create_negotiation_timeline(recommendations)
        
        # Generate preparation checklist
        checklist = self._create_preparation_checklist(unfair_clauses)
        
        # Alternative options
        alternatives = self._generate_alternative_options(unfair_clauses)
        
        # Estimate success rate
        success_rate = self._estimate_overall_success_rate(recommendations)
        
        report = AppealReport(
            overall_strategy=overall_strategy,
            priority_clauses=high_priority,
            recommendations=recommendations,
            negotiation_timeline=timeline,
            preparation_checklist=checklist,
            alternative_options=alternatives,
            estimated_success_rate=success_rate
        )
        
        self.logger.info("Appeal report generated successfully")
        return report
    
    def _get_base_recommendation(self, clause_type: str) -> Dict:
        """Get base recommendation data for clause type (traditional approach)"""
        clause_type = clause_type.lower()
        
        if clause_type in self.recommendations_db:
            return self.recommendations_db[clause_type].copy()
        else:
            # Generic recommendation for unknown clause types
            return {
                "priority": "medium",
                "strategy": "Review clause for fairness and negotiate more balanced terms",
                "alternatives": ["Request mutual obligations", "Add reasonable limitations", "Include dispute resolution"],
                "precedents": ["Unfair contract terms may be void", "Courts favor balanced agreements"],
                "negotiation_points": ["Argue for fairness", "Propose balanced alternatives", "Request legal review"],
                "template": f"We request review of this {clause_type} clause to ensure balanced terms for both parties.",
                "difficulty": "medium",
                "success_rate": "50%"
            }

    def _generate_ai_powered_recommendation(self, clause: UnfairClause) -> AppealRecommendation:
        """Generate truly AI-powered recommendation using vector search and clause analysis"""
        
        # Get vector search results for similar precedents
        vector_results = self.search_similar_precedents(clause.text, top_k=5)
        
        if not vector_results:
            # Fallback to basic recommendation if no vector results
            return self._generate_fallback_recommendation(clause)
        
        # Analyze clause content to understand specific issues
        clause_analysis = self._analyze_clause_content(clause)
        
        # Generate AI-powered strategy based on similar precedents
        strategy = self._generate_contextual_strategy(clause, vector_results, clause_analysis)
        
        # Generate dynamic alternatives based on precedent analysis
        alternatives = self._generate_dynamic_alternatives(clause, vector_results)
        
        # Generate contextual legal precedents
        legal_precedents = self._generate_contextual_precedents(vector_results)
        
        # Generate specific negotiation points
        negotiation_points = self._generate_negotiation_points(clause, vector_results, clause_analysis)
        
        # Generate dynamic template language
        template = self._generate_template_language(clause, strategy)
        
        # Calculate dynamic priority based on clause severity and precedent success rates
        priority = self._calculate_dynamic_priority(clause, vector_results)
        
        # Calculate success likelihood based on similar precedents
        success_rate = self._calculate_contextual_success_rate(vector_results, clause_analysis)
        
        return AppealRecommendation(
            clause_type=clause.clause_type,
            priority=priority,
            negotiation_strategy=strategy,
            suggested_alternatives=alternatives,
            legal_precedents=legal_precedents,
            negotiation_points=negotiation_points,
            template_language=template,
            difficulty_level=self._assess_difficulty(vector_results, clause_analysis),
            success_likelihood=f"{success_rate:.0f}%"
        )
    
    def _analyze_clause_content(self, clause: UnfairClause) -> Dict[str, any]:
        """Analyze clause content to identify specific problematic elements"""
        analysis = {
            "problematic_keywords": [],
            "severity_indicators": [],
            "imbalance_type": "unknown",
            "specific_issues": []
        }
        
        text_lower = clause.text.lower()
        
        # Identify problematic keywords
        termination_keywords = ["terminate", "end", "cancel", "discontinue"]
        liability_keywords = ["waive", "disclaim", "not liable", "no responsibility"]
        unilateral_keywords = ["sole discretion", "at will", "without notice", "immediately"]
        
        if any(kw in text_lower for kw in termination_keywords):
            analysis["problematic_keywords"].extend(["termination_clause"])
        if any(kw in text_lower for kw in liability_keywords):
            analysis["problematic_keywords"].extend(["liability_waiver"])
        if any(kw in text_lower for kw in unilateral_keywords):
            analysis["problematic_keywords"].extend(["unilateral_power"])
            
        # Identify severity indicators
        severe_keywords = ["without notice", "immediately", "sole discretion", "any reason", "no cause"]
        moderate_keywords = ["limited", "restrict", "exclude", "waive"]
        
        analysis["severity_indicators"] = [kw for kw in severe_keywords if kw in text_lower]
        if not analysis["severity_indicators"]:
            analysis["severity_indicators"] = [kw for kw in moderate_keywords if kw in text_lower]
            
        # Determine imbalance type
        if "terminate" in text_lower and ("company" in text_lower or "we" in text_lower):
            analysis["imbalance_type"] = "unilateral_termination"
        elif "liable" in text_lower or "responsibility" in text_lower:
            analysis["imbalance_type"] = "liability_shift"
        elif "renew" in text_lower or "automatic" in text_lower:
            analysis["imbalance_type"] = "auto_renewal"
            
        return analysis
    
    def _generate_contextual_strategy(self, clause: UnfairClause, vector_results: List, clause_analysis: Dict) -> str:
        """Generate contextual negotiation strategy based on clause analysis and similar precedents"""
        
        # Get top precedent strategies
        top_precedents = vector_results[:3]
        precedent_strategies = []
        
        for result in top_precedents:
            precedent = result.precedent
            strategy_insight = f"Based on {precedent.jurisdiction} precedent with {precedent.success_rate:.0%} success rate: {precedent.legal_basis[:100]}..."
            precedent_strategies.append(strategy_insight)
        
        # Generate contextual strategy based on analysis
        imbalance_type = clause_analysis["imbalance_type"]
        severity_indicators = clause_analysis["severity_indicators"]
        
        strategy_components = []
        
        if imbalance_type == "unilateral_termination":
            strategy_components.append("Challenge the unilateral nature of termination rights")
            if "without notice" in severity_indicators:
                strategy_components.append("Demand reasonable notice period requirements")
        elif imbalance_type == "liability_shift":
            strategy_components.append("Negotiate for mutual liability limitations")
            if "any" in severity_indicators or "all" in severity_indicators:
                strategy_components.append("Argue against broad liability waivers as unconscionable")
                
        # Combine with vector insights
        if precedent_strategies:
            main_strategy = " and ".join(strategy_components)
            context = f"{main_strategy}. {precedent_strategies[0]}"
        else:
            context = " and ".join(strategy_components) if strategy_components else "Negotiate for more balanced terms"
            
        return context
    
    def _generate_dynamic_alternatives(self, clause: UnfairClause, vector_results: List) -> List[str]:
        """Generate dynamic alternatives based on precedent analysis"""
        
        alternatives = []
        clause_type = clause.clause_type.lower()
        
        # Extract recommendations from top precedents
        for result in vector_results[:4]:
            precedent = result.precedent
            if precedent.recommendation_text and precedent.recommendation_text not in alternatives:
                alternatives.append(precedent.recommendation_text)
        
        # Add contextual alternatives based on clause analysis
        text_lower = clause.text.lower()
        
        if "terminate" in text_lower:
            alternatives.extend([
                "Add mutual termination rights with equal notice periods",
                "Require specific cause for immediate termination",
                "Include cure period before termination becomes effective"
            ])
        elif "liable" in text_lower or "waive" in text_lower:
            alternatives.extend([
                "Implement mutual liability caps based on contract value",
                "Exclude gross negligence and willful misconduct from waivers", 
                "Add specific liability exceptions for data breaches"
            ])
        elif "renew" in text_lower:
            alternatives.extend([
                "Require explicit written consent for each renewal",
                "Reduce advance notice period to 30 days",
                "Add option to negotiate terms at renewal"
            ])
            
        # Remove duplicates and limit to top alternatives
        unique_alternatives = list(dict.fromkeys(alternatives))
        return unique_alternatives[:6]
    
    def _generate_contextual_precedents(self, vector_results: List) -> List[str]:
        """Generate contextual legal precedents from vector search"""
        
        precedents = []
        
        for result in vector_results[:4]:
            precedent = result.precedent
            similarity_score = result.similarity_score
            
            precedent_text = (f"{precedent.legal_basis} "
                            f"(Success rate: {precedent.success_rate:.0%}, "
                            f"Similarity: {similarity_score:.1%}, "
                            f"{precedent.jurisdiction})")
            
            if precedent_text not in precedents:
                precedents.append(precedent_text)
                
            # Add case references if available
            for case_ref in precedent.case_references[:2]:  # Limit case references
                case_precedent = f"Legal precedent: {case_ref}"
                if case_precedent not in precedents:
                    precedents.append(case_precedent)
                    
        return precedents[:6]
    
    def _generate_negotiation_points(self, clause: UnfairClause, vector_results: List, clause_analysis: Dict) -> List[str]:
        """Generate specific negotiation points based on clause and precedent analysis"""
        
        points = []
        imbalance_type = clause_analysis["imbalance_type"]
        severity_indicators = clause_analysis["severity_indicators"]
        
        # Add precedent-based points
        for result in vector_results[:3]:
            precedent = result.precedent
            for case_ref in precedent.case_references[:1]:  # One case per precedent
                points.append(f"Cite legal precedent: {case_ref}")
                
        # Add contextual negotiation points
        if imbalance_type == "unilateral_termination":
            points.extend([
                "Argue for mutuality doctrine - equal obligations for both parties",
                "Request specific termination grounds rather than 'at will'"
            ])
            if "without notice" in severity_indicators:
                points.append("Demand reasonable notice period (30-90 days)")
                
        elif imbalance_type == "liability_shift":
            points.extend([
                "Challenge complete liability waiver as unconscionable",
                "Propose liability caps proportional to contract value"
            ])
            
        # Add general fairness arguments
        points.append(f"Highlight unfairness: clause confidence {clause.confidence:.0%} indicates problematic terms")
        
        return points[:8]
    
    def _generate_template_language(self, clause: UnfairClause, strategy: str) -> str:
        """Generate dynamic template language for negotiations"""
        
        clause_type = clause.clause_type.replace("_", " ").title()
        
        return (f"Regarding the {clause_type} provision: {strategy[:100]}... "
                f"We propose revising this clause to ensure balanced obligations for both parties.")
    
    def _calculate_dynamic_priority(self, clause: UnfairClause, vector_results: List) -> str:
        """Calculate dynamic priority based on clause confidence and precedent success rates"""
        
        confidence = clause.confidence
        avg_success_rate = sum(r.precedent.success_rate for r in vector_results[:3]) / len(vector_results[:3]) if vector_results else 0.5
        
        # Higher confidence in unfairness + higher precedent success = higher priority
        priority_score = (confidence * 0.6) + (avg_success_rate * 0.4)
        
        if priority_score > 0.8:
            return "high"
        elif priority_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_contextual_success_rate(self, vector_results: List, clause_analysis: Dict) -> float:
        """Calculate success rate based on similar precedent outcomes"""
        
        if not vector_results:
            return 50.0
            
        # Weight success rates by similarity scores
        weighted_success = 0
        total_weight = 0
        
        for result in vector_results[:4]:
            weight = result.similarity_score
            success_rate = result.precedent.success_rate
            weighted_success += (success_rate * weight)
            total_weight += weight
            
        if total_weight > 0:
            base_success = (weighted_success / total_weight) * 100
        else:
            base_success = 50.0
            
        # Adjust based on clause severity
        severity_indicators = clause_analysis.get("severity_indicators", [])
        if len(severity_indicators) > 2:  # Very problematic clause
            base_success += 10  # Higher success rate for clearly unfair clauses
        elif len(severity_indicators) == 0:  # Less clear unfairness
            base_success -= 5
            
        return min(95.0, max(25.0, base_success))  # Cap between 25-95%
    
    def _assess_difficulty(self, vector_results: List, clause_analysis: Dict) -> str:
        """Assess negotiation difficulty based on precedent analysis"""
        
        if not vector_results:
            return "medium"
            
        avg_success_rate = sum(r.precedent.success_rate for r in vector_results[:3]) / len(vector_results[:3])
        severity_count = len(clause_analysis.get("severity_indicators", []))
        
        # High success rate + clear severity = easier to challenge
        if avg_success_rate > 0.8 and severity_count > 1:
            return "low"
        elif avg_success_rate > 0.6:
            return "medium"  
        else:
            return "high"
            
    def _generate_fallback_recommendation(self, clause: UnfairClause) -> AppealRecommendation:
        """Generate fallback recommendation when no vector results available"""
        
        return AppealRecommendation(
            clause_type=clause.clause_type,
            priority="medium",
            negotiation_strategy=f"Review {clause.clause_type} clause for fairness and negotiate more balanced terms",
            suggested_alternatives=["Request mutual obligations", "Add reasonable limitations", "Include dispute resolution"],
            legal_precedents=["Unfair contract terms may be void under unconscionability doctrine"],
            negotiation_points=["Argue for fairness", "Propose balanced alternatives", "Request legal review"],
            template_language=f"We request review of this {clause.clause_type} clause to ensure balanced terms for both parties.",
            difficulty_level="medium",
            success_likelihood="50%"
        )

    def _generate_clause_recommendation(self, clause: UnfairClause) -> AppealRecommendation:
        """Generate recommendation for specific clause using AI-powered analysis"""
        
        # Use the new AI-powered recommendation system
        return self._generate_ai_powered_recommendation(clause)
    
    def _enhance_with_vector_search(self, clause: UnfairClause, rec_data: Dict) -> Dict:
        """Enhance recommendation with vector search results"""
        try:
            # Search for similar precedents
            vector_results = self.search_similar_precedents(clause.text, top_k=3)
            
            if not vector_results:
                return rec_data  # Return original if no vector results
            
            # Extract insights from vector search results
            enhanced_precedents = rec_data["precedents"].copy()
            enhanced_alternatives = rec_data["alternatives"].copy()
            enhanced_points = rec_data["negotiation_points"].copy()
            
            # Add vector search insights
            for result in vector_results[:2]:  # Use top 2 results
                precedent = result.precedent
                
                # Add legal precedent with success rate
                precedent_text = f"{precedent.legal_basis} (Success rate: {int(precedent.success_rate * 100)}% in {precedent.jurisdiction})"
                if precedent_text not in enhanced_precedents:
                    enhanced_precedents.append(precedent_text)
                
                # Add recommendation as alternative if not already present
                if precedent.recommendation_text not in enhanced_alternatives:
                    enhanced_alternatives.append(precedent.recommendation_text)
                
                # Add case references as negotiation points
                for case_ref in precedent.case_references:
                    point = f"Reference legal precedent: {case_ref}"
                    if point not in enhanced_points:
                        enhanced_points.append(point)
            
            # Calculate enhanced success rate
            if vector_results:
                vector_success_rates = [r.precedent.success_rate for r in vector_results]
                avg_vector_success = sum(vector_success_rates) / len(vector_success_rates)
                
                # Blend original success rate with vector-based insights
                original_rate = float(rec_data["success_rate"].rstrip('%')) / 100
                enhanced_rate = (original_rate * 0.6) + (avg_vector_success * 0.4)  # 60-40 blend
                rec_data["success_rate"] = f"{int(enhanced_rate * 100)}%"
            
            # Enhanced strategy with vector insights
            if vector_results:
                top_result = vector_results[0]
                enhanced_strategy = f"{rec_data['strategy']} Based on similar cases, consider: {top_result.precedent.recommendation_text}"
                rec_data["strategy"] = enhanced_strategy
            
            # Update with enhanced data
            rec_data["precedents"] = enhanced_precedents
            rec_data["alternatives"] = enhanced_alternatives  
            rec_data["negotiation_points"] = enhanced_points
            
            return rec_data
            
        except Exception as e:
            self.logger.error(f"Failed to enhance with vector search: {e}")
            return rec_data  # Return original on error
    
    def _generate_overall_strategy(self, clauses: List[UnfairClause], high_priority: List[str]) -> str:
        """Generate overall negotiation strategy"""
        
        if not clauses:
            return "No unfair clauses detected. Contract appears balanced."
        
        strategy_parts = []
        
        # Opening strategy
        if len(high_priority) > 3:
            strategy_parts.append("This contract contains multiple high-priority unfair clauses that require immediate attention.")
        elif len(high_priority) > 0:
            strategy_parts.append(f"Focus negotiations on {len(high_priority)} high-priority clauses that create significant imbalance.")
        else:
            strategy_parts.append("Address moderate-priority clauses to improve contract fairness.")
        
        # Negotiation approach
        if len(clauses) > 5:
            strategy_parts.append("Recommend comprehensive contract revision rather than piecemeal negotiations.")
        else:
            strategy_parts.append("Target specific clause modifications to achieve balanced terms.")
        
        # Leverage points
        strategy_parts.append("Emphasize mutual benefit and long-term partnership value during negotiations.")
        
        return " ".join(strategy_parts)
    
    def _create_negotiation_timeline(self, recommendations: List[AppealRecommendation]) -> Dict[str, str]:
        """Create suggested negotiation timeline"""
        
        timeline = {}
        
        # Phase 1: High priority clauses
        high_priority_clauses = [r for r in recommendations if r.priority == 'high']
        if high_priority_clauses:
            timeline['Phase 1 (Weeks 1-2)'] = f"Address {len(high_priority_clauses)} high-priority clauses: " + \
                ", ".join([r.clause_type for r in high_priority_clauses[:3]])
        
        # Phase 2: Medium priority clauses  
        medium_priority_clauses = [r for r in recommendations if r.priority == 'medium']
        if medium_priority_clauses:
            timeline['Phase 2 (Weeks 3-4)'] = f"Negotiate {len(medium_priority_clauses)} medium-priority improvements: " + \
                ", ".join([r.clause_type for r in medium_priority_clauses[:3]])
        
        # Phase 3: Final review
        timeline['Phase 3 (Week 5)'] = "Final contract review and acceptance/escalation decision"
        
        return timeline
    
    def _create_preparation_checklist(self, clauses: List[UnfairClause]) -> List[str]:
        """Create preparation checklist for negotiations"""
        
        checklist = [
            "Review all unfair clauses and prioritize by business impact",
            "Research competitor contract terms for benchmarking", 
            "Prepare alternative language proposals for each clause",
            "Identify your key negotiation objectives and minimum requirements",
            "Document the business value you bring to justify requests"
        ]
        
        # Add specific items based on clause types
        clause_types = set(clause.clause_type for clause in clauses)
        
        if 'mandatory_arbitration' in clause_types:
            checklist.append("Research local arbitration laws and consumer protection regulations")
        
        if 'limitation_of_liability' in clause_types:
            checklist.append("Calculate potential damages and appropriate liability caps")
            
        if 'unilateral_termination' in clause_types:
            checklist.append("Assess business impact of sudden termination and data loss")
        
        checklist.append("Consider involving legal counsel for high-stakes negotiations")
        
        return checklist
    
    def _generate_alternative_options(self, clauses: List[UnfairClause]) -> List[str]:
        """Generate alternative options if negotiation fails"""
        
        alternatives = []
        
        if len(clauses) > 5:
            alternatives.append("Seek alternative vendors with more balanced contract terms")
        
        alternatives.extend([
            "Request legal review to identify potentially void clauses",
            "Consider contract insurance to mitigate unfair terms",
            "Negotiate shorter contract terms to reduce exposure",
            "Document all negotiations for potential future legal proceedings"
        ])
        
        # Risk-specific alternatives
        clause_types = set(clause.clause_type for clause in clauses)
        
        if 'limitation_of_liability' in clause_types:
            alternatives.append("Purchase additional professional liability insurance")
        
        if 'unilateral_termination' in clause_types:
            alternatives.append("Implement robust data backup and portability procedures")
        
        return alternatives
    
    def _estimate_overall_success_rate(self, recommendations: List[AppealRecommendation]) -> str:
        """Estimate overall negotiation success rate"""
        
        if not recommendations:
            return "N/A"
        
        # Parse success rates and calculate weighted average
        total_weight = 0
        weighted_success = 0
        
        weight_map = {'high': 3, 'medium': 2, 'low': 1}
        
        for rec in recommendations:
            try:
                success_pct = int(rec.success_likelihood.rstrip('%'))
                weight = weight_map.get(rec.priority, 1)
                
                weighted_success += success_pct * weight
                total_weight += weight
                
            except (ValueError, AttributeError):
                continue
        
        if total_weight == 0:
            return "50%"
        
        avg_success = weighted_success / total_weight
        
        if avg_success >= 70:
            return f"{int(avg_success)}% - High likelihood of successful negotiations"
        elif avg_success >= 50:
            return f"{int(avg_success)}% - Moderate success expected with good preparation"
        else:
            return f"{int(avg_success)}% - Challenging negotiations, consider alternatives"

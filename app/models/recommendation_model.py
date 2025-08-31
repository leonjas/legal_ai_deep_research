"""
Legal Recommendation Model
Core model for generating legal recommendations using FAISS vector search
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer

# Try to import settings
try:
    from app.utils.settings import RECOMMENDATION_MODEL, FAISS_INDEX_SIZE
except ImportError:
    RECOMMENDATION_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_SIZE = 384

logger = logging.getLogger(__name__)

@dataclass
class LegalPrecedent:
    """Data class for legal precedent information"""
    case_id: str
    title: str
    text: str
    outcome: str
    jurisdiction: str
    year: int
    success_rate: float
    category: str = "general"

@dataclass
class RecommendationResult:
    """Data class for recommendation results"""
    priority: str
    success_likelihood: float
    difficulty: str
    strategy: str
    alternatives: List[str]
    legal_precedents: List[Dict[str, any]]
    confidence_score: float = 0.0

class RecommendationModel:
    """
    Core model for generating legal recommendations using vector similarity search
    """
    
    def __init__(self, model_name: str = RECOMMENDATION_MODEL):
        self.model_name = model_name
        self.sentence_model = None
        self.faiss_index = None
        self.case_embeddings = None
        self.legal_corpus = []
        self.logger = logging.getLogger(__name__)
        
        self._initialize_model()
        self._load_legal_corpus()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.sentence_model = SentenceTransformer(self.model_name)
            self.logger.info(f"Initialized sentence transformer: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentence transformer: {e}")
            raise
    
    def _load_legal_corpus(self):
        """Load legal precedents and build FAISS index"""
        
        # Enhanced legal corpus with categorized precedents
        self.legal_corpus = [
            LegalPrecedent(
                case_id="termination_001",
                title="Unilateral Termination Challenge",
                text="Good faith and fair dealing - termination must have legitimate business reason",
                outcome="Appeal Successful",
                jurisdiction="UCC/Commercial Law",
                year=2022,
                success_rate=0.73,
                category="termination"
            ),
            LegalPrecedent(
                case_id="liability_001", 
                title="Unconscionable Liability Waiver",
                text="Unconscionable contract doctrine - complete liability waivers often unenforceable",
                outcome="Appeal Successful",
                jurisdiction="US Federal",
                year=2021,
                success_rate=0.85,
                category="liability"
            ),
            LegalPrecedent(
                case_id="indemnification_001",
                title="Broad Indemnification Challenge", 
                text="Broad indemnification may be unenforceable as against public policy",
                outcome="Appeal Successful",
                jurisdiction="Common Law",
                year=2023,
                success_rate=0.77,
                category="indemnification"
            ),
            LegalPrecedent(
                case_id="arbitration_001",
                title="Class Action Waiver Challenge",
                text="Some states void class action waivers as against public policy",
                outcome="Mixed Results",
                jurisdiction="State Courts", 
                year=2023,
                success_rate=0.58,
                category="arbitration"
            ),
            LegalPrecedent(
                case_id="auto_renewal_001",
                title="Auto-Renewal Consumer Protection",
                text="Consumer protection laws often limit auto-renewal terms",
                outcome="Appeal Successful",
                jurisdiction="State Consumer Protection",
                year=2024,
                success_rate=0.81,
                category="automatic_renewal"
            ),
            LegalPrecedent(
                case_id="modification_001",
                title="Unilateral Modification Rights",
                text="Unilateral contract modification clauses subject to good faith requirements",
                outcome="Appeal Successful", 
                jurisdiction="Contract Law",
                year=2022,
                success_rate=0.68,
                category="modification"
            ),
            LegalPrecedent(
                case_id="penalty_001",
                title="Unreasonable Penalty Challenge",
                text="Penalty clauses must be reasonable and not punitive in nature",
                outcome="Appeal Successful",
                jurisdiction="Contract Law",
                year=2023,
                success_rate=0.72,
                category="penalties"
            ),
            LegalPrecedent(
                case_id="jurisdiction_001",
                title="Forum Selection Clause Challenge",
                text="Forum selection clauses may be unenforceable if fundamentally unfair",
                outcome="Appeal Successful",
                jurisdiction="Federal Courts",
                year=2022,
                success_rate=0.64,
                category="jurisdiction"
            )
        ]
        
        self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index from legal corpus"""
        try:
            # Extract texts for embedding
            corpus_texts = [precedent.text for precedent in self.legal_corpus]
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(corpus_texts, show_progress_bar=False)
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Build FAISS index
            dimension = embeddings_np.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings_np)
            self.case_embeddings = embeddings_np
            
            self.logger.info(f"Built FAISS index with {len(corpus_texts)} legal precedents")
            
        except Exception as e:
            self.logger.error(f"Failed to build FAISS index: {e}")
            raise
    
    def find_similar_precedents(self, clause_text: str, top_k: int = 3) -> List[Tuple[LegalPrecedent, float]]:
        """
        Find similar legal precedents using FAISS vector search
        
        Args:
            clause_text: The unfair clause text to find precedents for
            top_k: Number of similar precedents to return
            
        Returns:
            List of tuples (precedent, similarity_score)
        """
        try:
            # Encode the clause text
            query_embedding = self.sentence_model.encode([clause_text])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Convert distances to similarity scores (L2 distance -> similarity)
            similarities = 1 / (1 + distances[0])  # Convert distance to similarity
            
            # Get matching precedents
            similar_precedents = []
            for idx, similarity in zip(indices[0], similarities):
                if idx < len(self.legal_corpus):
                    precedent = self.legal_corpus[idx]
                    similar_precedents.append((precedent, similarity))
            
            return similar_precedents
            
        except Exception as e:
            self.logger.error(f"Error finding similar precedents: {e}")
            return []
    
    def generate_recommendation(self, clause_text: str, clause_type: str, 
                              severity: str = "medium") -> RecommendationResult:
        """
        Generate legal recommendation for an unfair clause
        
        Args:
            clause_text: The unfair clause text
            clause_type: Type/category of the unfair clause
            severity: Severity level (low/medium/high)
            
        Returns:
            RecommendationResult with strategy and precedents
        """
        try:
            # Find similar precedents
            similar_precedents = self.find_similar_precedents(clause_text, top_k=3)
            
            if not similar_precedents:
                return self._generate_fallback_recommendation(clause_type, severity)
            
            # Get the best matching precedent
            best_precedent, best_similarity = similar_precedents[0]
            
            # Calculate recommendation metrics
            priority = self._determine_priority(severity, best_precedent.success_rate)
            success_likelihood = self._calculate_success_likelihood(similar_precedents)
            difficulty = self._assess_difficulty(clause_type, best_precedent.jurisdiction)
            
            # Generate strategy based on best precedent
            strategy = self._generate_strategy(best_precedent, clause_type, best_similarity)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(clause_type, similar_precedents)
            
            # Format precedents for output
            formatted_precedents = []
            for precedent, similarity in similar_precedents:
                formatted_precedents.append({
                    "description": precedent.text,
                    "success_rate": precedent.success_rate,
                    "similarity": similarity,
                    "jurisdiction": precedent.jurisdiction,
                    "case_reference": precedent.title
                })
            
            return RecommendationResult(
                priority=priority,
                success_likelihood=success_likelihood,
                difficulty=difficulty,
                strategy=strategy,
                alternatives=alternatives,
                legal_precedents=formatted_precedents,
                confidence_score=best_similarity
            )
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return self._generate_fallback_recommendation(clause_type, severity)
    
    def _determine_priority(self, severity: str, success_rate: float) -> str:
        """Determine priority level based on severity and success rate"""
        if severity == "high" or success_rate > 0.8:
            return "high"
        elif severity == "medium" or success_rate > 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_success_likelihood(self, precedents: List[Tuple[LegalPrecedent, float]]) -> float:
        """Calculate weighted success likelihood from similar precedents"""
        if not precedents:
            return 0.5
        
        total_weight = 0
        weighted_success = 0
        
        for precedent, similarity in precedents:
            weight = similarity  # Use similarity as weight
            weighted_success += precedent.success_rate * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return min(weighted_success / total_weight, 0.95)  # Cap at 95%
    
    def _assess_difficulty(self, clause_type: str, jurisdiction: str) -> str:
        """Assess difficulty based on clause type and jurisdiction"""
        
        # Difficulty mapping
        difficult_types = ["broad_indemnification", "liability_limitation"]
        easy_types = ["automatic_renewal", "unreasonable_penalties"]
        
        difficult_jurisdictions = ["US Federal", "Supreme Court"]
        easy_jurisdictions = ["State Consumer Protection", "State Courts"]
        
        difficulty_score = 0
        
        if clause_type in difficult_types:
            difficulty_score += 2
        elif clause_type in easy_types:
            difficulty_score += 0
        else:
            difficulty_score += 1
        
        if jurisdiction in difficult_jurisdictions:
            difficulty_score += 2
        elif jurisdiction in easy_jurisdictions:
            difficulty_score += 0
        else:
            difficulty_score += 1
        
        if difficulty_score >= 3:
            return "high"
        elif difficulty_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_strategy(self, precedent: LegalPrecedent, clause_type: str, similarity: float) -> str:
        """Generate legal strategy based on precedent"""
        
        base_strategies = {
            "unilateral_termination": "Challenge the unilateral nature of termination rights.",
            "broad_indemnification": "Argue that broad indemnification violates public policy.",
            "liability_limitation": "Challenge complete liability exclusions as unconscionable.",
            "automatic_renewal": "Invoke consumer protection laws regarding auto-renewal.",
            "mandatory_arbitration": "Challenge arbitration clauses in consumer contexts.",
            "unreasonable_penalties": "Argue penalties are punitive rather than compensatory."
        }
        
        base_strategy = base_strategies.get(clause_type, "Challenge the clause as unfair.")
        
        # Add precedent-specific guidance
        if similarity > 0.7:
            strategy = f"{base_strategy} Based on {precedent.jurisdiction} precedent with {precedent.success_rate:.0%} success rate: {precedent.text}"
        else:
            strategy = f"{base_strategy} Consider precedent from {precedent.jurisdiction}: {precedent.text}"
        
        return strategy
    
    def _generate_alternatives(self, clause_type: str, precedents: List[Tuple[LegalPrecedent, float]]) -> List[str]:
        """Generate alternative approaches based on clause type and precedents"""
        
        alternatives_map = {
            "unilateral_termination": [
                "Require specific grounds for termination and cure periods",
                "Add mutual termination rights with reasonable notice periods",
                "Negotiate termination for cause provisions only"
            ],
            "broad_indemnification": [
                "Limit indemnification to customer's breach or negligence",
                "Negotiate mutual indemnification clauses",
                "Exclude indemnification for company's gross negligence"
            ],
            "liability_limitation": [
                "Negotiate mutual liability caps and exclude gross negligence",
                "Request liability cap proportional to contract value or annual fees",
                "Preserve liability for data breaches and IP infringement"
            ],
            "automatic_renewal": [
                "Require explicit consent for renewals or shorter notice periods",
                "Add opt-out reminders before renewal dates",
                "Negotiate month-to-month terms instead of annual"
            ],
            "mandatory_arbitration": [
                "Challenge class action waivers in consumer contexts",
                "Negotiate mutual arbitration with consumer-friendly rules",
                "Preserve right to small claims court"
            ]
        }
        
        base_alternatives = alternatives_map.get(clause_type, [
            "Negotiate more balanced terms",
            "Seek legal counsel for specific alternatives",
            "Consider contract rejection if terms are unacceptable"
        ])
        
        # Add precedent-based alternatives if available
        precedent_alternatives = []
        for precedent, similarity in precedents[:2]:  # Use top 2 precedents
            if precedent.category in alternatives_map:
                category_alternatives = alternatives_map[precedent.category]
                for alt in category_alternatives[:1]:  # Take first alternative from each category
                    if alt not in base_alternatives and alt not in precedent_alternatives:
                        precedent_alternatives.append(alt)
        
        # Combine and return top alternatives
        all_alternatives = base_alternatives + precedent_alternatives
        return all_alternatives[:3]  # Return top 3
    
    def _generate_fallback_recommendation(self, clause_type: str, severity: str) -> RecommendationResult:
        """Generate fallback recommendation when no precedents are found"""
        
        return RecommendationResult(
            priority=severity,
            success_likelihood=0.5,
            difficulty="medium",
            strategy=f"Challenge the {clause_type.replace('_', ' ')} clause as potentially unfair.",
            alternatives=[
                "Seek legal counsel for specific guidance",
                "Negotiate more balanced contract terms",
                "Consider the enforceability in your jurisdiction"
            ],
            legal_precedents=[],
            confidence_score=0.0
        )

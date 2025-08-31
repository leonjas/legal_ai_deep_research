"""
Legal Recommendation Pipeline
High-level pipeline for generating legal recommendations from unfair clause detection results
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import time

from app.models.recommendation_model import RecommendationModel, RecommendationResult
from app.models.unfair_pipeline import UnfairDetectionResult

# Try to import settings
try:
    from app.utils.settings import DEFAULT_RECOMMENDATION_MODEL, MIN_CONFIDENCE_FOR_RECOMMENDATION
except ImportError:
    DEFAULT_RECOMMENDATION_MODEL = "all-MiniLM-L6-v2"
    MIN_CONFIDENCE_FOR_RECOMMENDATION = 0.6

logger = logging.getLogger(__name__)

class RecommendationPipelineResult:
    """Container for recommendation pipeline results."""
    
    def __init__(self, file_path: str, recommendations: List[Dict], 
                 model_info: Dict, processing_time: float = 0.0):
        self.file_path = file_path
        self.recommendations = recommendations
        self.model_info = model_info
        self.processing_time = processing_time
        
    def get_summary(self) -> Dict:
        """Get a summary of the recommendations."""
        high_priority = sum(1 for rec in self.recommendations if rec.get("priority") == "high")
        avg_success = sum(rec.get("success_likelihood", 0) for rec in self.recommendations) / max(len(self.recommendations), 1)
        
        return {
            "file_name": Path(self.file_path).name,
            "total_recommendations": len(self.recommendations),
            "high_priority_count": high_priority,
            "average_success_likelihood": avg_success,
            "model_used": self.model_info.get("model_name", "unknown"),
            "processing_time": self.processing_time
        }
    
    def get_high_priority_recommendations(self) -> List[Dict]:
        """Get only high priority recommendations."""
        return [rec for rec in self.recommendations if rec.get("priority") == "high"]
    
    def get_recommendations_by_success_rate(self, min_success_rate: float = 0.7) -> List[Dict]:
        """Get recommendations with success rate above threshold."""
        return [rec for rec in self.recommendations 
                if rec.get("success_likelihood", 0) >= min_success_rate]

class RecommendationPipeline:
    """
    High-level pipeline for generating legal recommendations from contract analysis
    """
    
    def __init__(self, model_name: str = DEFAULT_RECOMMENDATION_MODEL):
        self.model_name = model_name
        self.recommendation_model = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the recommendation pipeline"""
        try:
            self.recommendation_model = RecommendationModel(self.model_name)
            self.logger.info("Recommendation pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize recommendation pipeline: {e}")
            raise
    
    def process_file(self, file_path: str, 
                    unfair_detection_result: UnfairDetectionResult) -> RecommendationPipelineResult:
        """
        Process a file and generate recommendations for detected unfair clauses
        
        Args:
            file_path: Path to the contract file
            unfair_detection_result: Results from unfair clause detection
            
        Returns:
            RecommendationPipelineResult with generated recommendations
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating recommendations for {Path(file_path).name}")
            
            recommendations = []
            
            # Generate recommendations for each unfair clause
            for i, clause_data in enumerate(unfair_detection_result.unfair_clauses):
                try:
                    # Extract clause information
                    clause_text = clause_data.get("text", "")
                    clause_type = clause_data.get("category", "unknown")
                    confidence = clause_data.get("confidence", 0.0)
                    
                    # Skip low confidence clauses
                    if confidence < MIN_CONFIDENCE_FOR_RECOMMENDATION:
                        continue
                    
                    # Determine severity
                    severity = self._determine_severity(clause_type, confidence)
                    
                    # Generate recommendation
                    recommendation_result = self.recommendation_model.generate_recommendation(
                        clause_text=clause_text,
                        clause_type=clause_type,
                        severity=severity
                    )
                    
                    # Format recommendation for output
                    formatted_recommendation = self._format_recommendation(
                        clause_index=i + 1,
                        clause_data=clause_data,
                        recommendation_result=recommendation_result
                    )
                    
                    recommendations.append(formatted_recommendation)
                    
                except Exception as e:
                    self.logger.error(f"Error generating recommendation for clause {i+1}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            model_info = {
                "model_name": self.model_name,
                "total_clauses_processed": len(unfair_detection_result.unfair_clauses),
                "recommendations_generated": len(recommendations)
            }
            
            self.logger.info(f"Generated {len(recommendations)} recommendations in {processing_time:.2f}s")
            
            return RecommendationPipelineResult(
                file_path=file_path,
                recommendations=recommendations,
                model_info=model_info,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def process_contract_text(self, contract_text: str, 
                            unfair_clauses: List[Dict]) -> List[Dict]:
        """
        Process contract text directly and generate recommendations
        
        Args:
            contract_text: Full contract text
            unfair_clauses: List of detected unfair clauses
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = []
            
            for i, clause in enumerate(unfair_clauses):
                try:
                    # Extract clause information
                    clause_text = clause.text if hasattr(clause, 'text') else str(clause)
                    clause_type = clause.clause_type if hasattr(clause, 'clause_type') else "unknown"
                    confidence = clause.confidence if hasattr(clause, 'confidence') else 0.7
                    severity = clause.severity if hasattr(clause, 'severity') else "medium"
                    
                    # Generate recommendation
                    recommendation_result = self.recommendation_model.generate_recommendation(
                        clause_text=clause_text,
                        clause_type=clause_type,
                        severity=severity
                    )
                    
                    # Format for output
                    formatted_rec = {
                        "clause_index": i + 1,
                        "clause_type": clause_type,
                        "clause_text": clause_text[:200] + "..." if len(clause_text) > 200 else clause_text,
                        "priority": recommendation_result.priority,
                        "success_likelihood": recommendation_result.success_likelihood,
                        "difficulty": recommendation_result.difficulty,
                        "strategy": recommendation_result.strategy,
                        "alternatives": recommendation_result.alternatives,
                        "legal_precedents": recommendation_result.legal_precedents,
                        "confidence_score": recommendation_result.confidence_score
                    }
                    
                    recommendations.append(formatted_rec)
                    
                except Exception as e:
                    self.logger.error(f"Error processing clause {i+1}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error processing contract text: {e}")
            return []
    
    def _determine_severity(self, clause_type: str, confidence: float) -> str:
        """Determine severity based on clause type and confidence"""
        
        high_risk_types = [
            "broad_indemnification", 
            "liability_limitation", 
            "unilateral_termination"
        ]
        
        medium_risk_types = [
            "automatic_renewal",
            "mandatory_arbitration", 
            "unilateral_modification"
        ]
        
        if clause_type in high_risk_types or confidence > 0.8:
            return "high"
        elif clause_type in medium_risk_types or confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _format_recommendation(self, clause_index: int, clause_data: Dict, 
                             recommendation_result: RecommendationResult) -> Dict:
        """Format recommendation result for output"""
        
        return {
            "clause_index": clause_index,
            "clause_type": clause_data.get("category", "unknown"),
            "clause_text": clause_data.get("text", "")[:200] + "..." if len(clause_data.get("text", "")) > 200 else clause_data.get("text", ""),
            "confidence": clause_data.get("confidence", 0.0),
            "priority": recommendation_result.priority,
            "success_likelihood": recommendation_result.success_likelihood,
            "difficulty": recommendation_result.difficulty,
            "strategy": recommendation_result.strategy,
            "alternatives": recommendation_result.alternatives,
            "legal_precedents": recommendation_result.legal_precedents,
            "recommendation_confidence": recommendation_result.confidence_score,
            "model_info": {
                "model_used": self.model_name,
                "precedents_found": len(recommendation_result.legal_precedents)
            }
        }
    
    def generate_bulk_recommendations(self, unfair_results: List[UnfairDetectionResult]) -> List[RecommendationPipelineResult]:
        """
        Generate recommendations for multiple files
        
        Args:
            unfair_results: List of unfair detection results
            
        Returns:
            List of recommendation pipeline results
        """
        results = []
        
        for unfair_result in unfair_results:
            try:
                recommendation_result = self.process_file(
                    file_path=unfair_result.file_path,
                    unfair_detection_result=unfair_result
                )
                results.append(recommendation_result)
                
            except Exception as e:
                self.logger.error(f"Error processing {unfair_result.file_path}: {e}")
                continue
        
        return results
    
    def get_pipeline_statistics(self, results: List[RecommendationPipelineResult]) -> Dict[str, Any]:
        """Get statistics across multiple recommendation results"""
        
        if not results:
            return {}
        
        total_recommendations = sum(len(r.recommendations) for r in results)
        total_high_priority = sum(len(r.get_high_priority_recommendations()) for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        
        success_rates = []
        for result in results:
            for rec in result.recommendations:
                success_rates.append(rec.get("success_likelihood", 0))
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        return {
            "files_processed": len(results),
            "total_recommendations": total_recommendations,
            "high_priority_recommendations": total_high_priority,
            "average_success_likelihood": avg_success_rate,
            "total_processing_time": total_processing_time,
            "recommendations_per_file": total_recommendations / len(results),
            "high_priority_percentage": (total_high_priority / max(total_recommendations, 1)) * 100
        }

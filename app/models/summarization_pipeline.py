"""
Contract Summarization Pipeline
High-level pipeline for generating contract summaries from contract text
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import time

from app.models.contract_summarizer import ContractSummarizer, ContractSummary, EnhancedContractSummary

logger = logging.getLogger(__name__)

class SummarizationPipelineResult:
    """Container for summarization pipeline results."""
    
    def __init__(self, file_path: str, basic_summary: ContractSummary, 
                 enhanced_summary: Optional[EnhancedContractSummary] = None,
                 processing_time: float = 0.0):
        self.file_path = file_path
        self.basic_summary = basic_summary
        self.enhanced_summary = enhanced_summary
        self.processing_time = processing_time
        
    def get_summary_statistics(self) -> Dict:
        """Get statistics about the summarization results."""
        stats = {
            "file_name": Path(self.file_path).name,
            "word_count": self.basic_summary.word_count,
            "estimated_read_time": self.basic_summary.estimated_read_time,
            "contract_type": self.basic_summary.contract_type,
            "parties_count": len(self.basic_summary.parties_involved),
            "key_terms_count": len(self.basic_summary.key_terms),
            "processing_time": self.processing_time
        }
        
        if self.enhanced_summary:
            stats.update({
                "risk_level": self.enhanced_summary.risk_prioritization.get("overall_risk_level", "Unknown"),
                "actionable_insights_count": len(self.enhanced_summary.actionable_insights),
                "compliance_concerns_count": len(self.enhanced_summary.compliance_concerns)
            })
        
        return stats

class SummarizationPipeline:
    """
    High-level pipeline for generating contract summaries
    """
    
    def __init__(self):
        self.summarizer = ContractSummarizer()
        self.logger = logging.getLogger(__name__)
        
    def process_file(self, file_path: str, contract_text: str, 
                    unfair_clauses: Optional[List[Any]] = None, 
                    risk_score: Optional[float] = None,
                    include_enhanced: bool = True) -> SummarizationPipelineResult:
        """
        Process a contract file and generate summaries
        
        Args:
            file_path: Path to the contract file
            contract_text: Full contract text
            unfair_clauses: Optional list of unfair clauses for enhanced summary
            risk_score: Optional risk score for enhanced summary
            include_enhanced: Whether to generate enhanced summary
            
        Returns:
            SummarizationPipelineResult with generated summaries
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting summarization for {Path(file_path).name}")
            
            # Generate basic summary
            basic_summary = self.summarizer.summarize_contract(contract_text)
            
            # Generate enhanced summary if requested and data available
            enhanced_summary = None
            if include_enhanced and unfair_clauses is not None and risk_score is not None:
                enhanced_summary = self.summarizer.generate_enhanced_contract_summary(
                    contract_text=contract_text,
                    unfair_clauses=unfair_clauses,
                    risk_score=risk_score
                )
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Summarization completed in {processing_time:.2f}s")
            
            return SummarizationPipelineResult(
                file_path=file_path,
                basic_summary=basic_summary,
                enhanced_summary=enhanced_summary,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def process_multiple_files(self, files_data: List[Dict[str, Any]]) -> List[SummarizationPipelineResult]:
        """
        Process multiple contract files
        
        Args:
            files_data: List of dictionaries with file information:
                       [{"file_path": str, "contract_text": str, "unfair_clauses": List, "risk_score": float}]
            
        Returns:
            List of SummarizationPipelineResult objects
        """
        results = []
        
        for file_data in files_data:
            try:
                result = self.process_file(
                    file_path=file_data["file_path"],
                    contract_text=file_data["contract_text"],
                    unfair_clauses=file_data.get("unfair_clauses"),
                    risk_score=file_data.get("risk_score"),
                    include_enhanced=file_data.get("include_enhanced", True)
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_data['file_path']}: {e}")
                continue
        
        return results
    
    def generate_basic_summary_only(self, contract_text: str) -> ContractSummary:
        """
        Generate only basic summary for quick analysis
        
        Args:
            contract_text: Full contract text
            
        Returns:
            ContractSummary object
        """
        return self.summarizer.summarize_contract(contract_text)
    
    def generate_enhanced_summary_only(self, contract_text: str, unfair_clauses: List[Any], 
                                     risk_score: float) -> EnhancedContractSummary:
        """
        Generate only enhanced summary for detailed analysis
        
        Args:
            contract_text: Full contract text
            unfair_clauses: List of unfair clauses
            risk_score: Overall risk score
            
        Returns:
            EnhancedContractSummary object
        """
        return self.summarizer.generate_enhanced_contract_summary(
            contract_text=contract_text,
            unfair_clauses=unfair_clauses,
            risk_score=risk_score
        )
    
    def get_pipeline_statistics(self, results: List[SummarizationPipelineResult]) -> Dict[str, Any]:
        """Get statistics across multiple summarization results"""
        
        if not results:
            return {}
        
        total_words = sum(r.basic_summary.word_count for r in results)
        total_read_time = sum(r.basic_summary.estimated_read_time for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        
        # Contract types distribution
        contract_types = {}
        for result in results:
            contract_type = result.basic_summary.contract_type
            contract_types[contract_type] = contract_types.get(contract_type, 0) + 1
        
        # Risk levels distribution (if enhanced summaries available)
        risk_levels = {}
        enhanced_count = 0
        for result in results:
            if result.enhanced_summary:
                enhanced_count += 1
                risk_level = result.enhanced_summary.risk_prioritization.get("overall_risk_level", "Unknown")
                risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        return {
            "files_processed": len(results),
            "total_word_count": total_words,
            "average_word_count": total_words / len(results),
            "total_estimated_read_time": total_read_time,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(results),
            "contract_types_distribution": contract_types,
            "enhanced_summaries_generated": enhanced_count,
            "risk_levels_distribution": risk_levels
        }
    
    def export_summaries_to_dict(self, results: List[SummarizationPipelineResult]) -> List[Dict[str, Any]]:
        """Export summarization results to dictionary format for JSON/API output"""
        
        exported_results = []
        
        for result in results:
            basic_dict = {
                "file_path": result.file_path,
                "basic_summary": {
                    "executive_summary": result.basic_summary.executive_summary,
                    "contract_type": result.basic_summary.contract_type,
                    "parties_involved": result.basic_summary.parties_involved,
                    "key_terms": result.basic_summary.key_terms,
                    "financial_terms": result.basic_summary.financial_terms,
                    "termination_conditions": result.basic_summary.termination_conditions,
                    "governing_law": result.basic_summary.governing_law,
                    "word_count": result.basic_summary.word_count,
                    "estimated_read_time": result.basic_summary.estimated_read_time
                },
                "processing_time": result.processing_time
            }
            
            # Add enhanced summary if available
            if result.enhanced_summary:
                basic_dict["enhanced_summary"] = {
                    "executive_summary": result.enhanced_summary.executive_summary,
                    "risk_prioritization": result.enhanced_summary.risk_prioritization,
                    "actionable_insights": result.enhanced_summary.actionable_insights,
                    "negotiation_strategy": result.enhanced_summary.negotiation_strategy,
                    "compliance_concerns": result.enhanced_summary.compliance_concerns,
                    "sections_analysis": result.enhanced_summary.sections_analysis,
                    "clause_relationships": result.enhanced_summary.clause_relationships
                }
            
            exported_results.append(basic_dict)
        
        return exported_results

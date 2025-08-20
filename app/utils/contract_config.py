"""
Configuration settings for contract analysis system
"""

import os
from typing import Dict, Any

class ContractAnalysisConfig:
    """Configuration class for contract analysis settings"""
    
    # Model configurations
    DEFAULT_LEGAL_MODEL = "nlpaueb/legal-bert-base-uncased"
    FALLBACK_MODEL = "bert-base-uncased"
    SENTENCE_MODEL = "all-MiniLM-L6-v2"
    
    # Analysis thresholds
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.1
    
    # Risk scoring weights
    SEVERITY_WEIGHTS = {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3
    }
    
    # File processing limits
    MAX_FILE_SIZE_MB = 50
    MAX_TEXT_LENGTH = 1000000  # 1M characters
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt']
    
    # Model download settings
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "contract_analysis")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "models": {
                "legal_model": cls.DEFAULT_LEGAL_MODEL,
                "fallback_model": cls.FALLBACK_MODEL,
                "sentence_model": cls.SENTENCE_MODEL
            },
            "thresholds": {
                "default_confidence": cls.DEFAULT_CONFIDENCE_THRESHOLD,
                "high_confidence": cls.HIGH_CONFIDENCE_THRESHOLD,
                "low_confidence": cls.LOW_CONFIDENCE_THRESHOLD
            },
            "weights": cls.SEVERITY_WEIGHTS,
            "limits": {
                "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
                "max_text_length": cls.MAX_TEXT_LENGTH
            },
            "supported_extensions": cls.SUPPORTED_EXTENSIONS,
            "cache_dir": cls.CACHE_DIR
        }
    
    @classmethod
    def setup_cache_directory(cls):
        """Ensure cache directory exists"""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        return cls.CACHE_DIR

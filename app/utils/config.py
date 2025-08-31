"""
Configuration settings for the Legal AI Deep Research application
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "Legal AI Deep Research"
APP_VERSION = "1.0.0"

# Paths
APP_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = APP_ROOT.parent
TEMP_DIR = PROJECT_ROOT / "temp"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Model settings (these can be overridden by environment variables)
DEFAULT_UNFAIR_MODEL = "marmolpen3/lexglue-unfair-tos"
ALTERNATIVE_UNFAIR_MODELS = [
    "CodeHima/TOSRobertaV2",
    "nlpaueb/legal-bert-base-uncased",
    "bert-base-uncased"
]

# Analysis settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
MIN_CLAUSE_WORDS = 4
MIN_CONTRACT_LENGTH = 100

# Supported file types
SUPPORTED_FILE_TYPES = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword',
    'txt': 'text/plain'
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# UI Settings
STREAMLIT_CONFIG = {
    'page_title': 'Contract Analysis - Unfair Clause Detection',
    'page_icon': '📄',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Feature flags
ENABLE_ML_DETECTION = True
ENABLE_PATTERN_DETECTION = True
ENABLE_RISK_ASSESSMENT = True
ENABLE_RECOMMENDATIONS = True

# Contract analysis thresholds
RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Legal clause categories and their severity levels
CLAUSE_SEVERITY = {
    'limitation_of_liability': 'high',
    'unilateral_termination': 'high',
    'unilateral_change': 'medium',
    'content_removal': 'medium',
    'contract_by_using': 'low',
    'choice_of_law': 'low',
    'jurisdiction': 'medium',
    'arbitration': 'medium',
    'automatic_renewal': 'medium',
    'broad_indemnification': 'high',
    'unreasonable_penalties': 'high'
}

# Export settings
EXPORT_FORMATS = ['json', 'csv', 'txt', 'pdf']

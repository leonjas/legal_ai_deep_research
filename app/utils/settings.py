import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # read .env if present
except ImportError:
    # dotenv not available, continue without it
    pass

# Unfair clause detection settings
UNFAIR_MODEL = "marmolpen3/lexglue-unfair-tos"  # Category-level model
UNFAIR_FALLBACK_MODEL = "CodeHima/TOSRobertaV2"  # Binary fair/unfair
UNFAIR_MIN_CONF = 0.65

# UNFAIR-ToS category labels (8 categories)
# Note: Order will be calibrated empirically using seed examples
UNFAIR_LABELS = [
    "limitation_of_liability",
    "unilateral_termination", 
    "unilateral_change",
    "content_removal",
    "contract_by_using",
    "choice_of_law",
    "jurisdiction",
    "arbitration"
]

# Seed examples for label calibration
UNFAIR_SEEDS = {
    "limitation_of_liability": "To the extent permitted by law, we are not liable for any direct or indirect damages, even if advised of the possibility.",
    "unilateral_termination": "We may terminate your account at any time without prior notice or reason.",
    "unilateral_change": "We may change these terms at any time by posting an updated version without your consent.",
    "content_removal": "We may remove or block any user content at our sole discretion and without notice.",
    "contract_by_using": "By merely using the service, you agree to these terms without any further action.",
    "choice_of_law": "These terms are governed by the laws of the State of California.",
    "jurisdiction": "Any disputes will be subject to the exclusive jurisdiction of the courts of New York.",
    "arbitration": "Any dispute shall be resolved by binding arbitration and not in a court of law."
}

# Alternative models that can be used
ALTERNATIVE_UNFAIR_MODELS = [
    "CodeHima/TOSRobertaV2",
    "nlpaueb/legal-bert-base-uncased",
    "bert-base-uncased"
]

# Contract analysis settings
MAX_CLAUSE_LENGTH = int(os.getenv("MAX_CLAUSE_LENGTH", "512"))
MIN_CLAUSE_WORDS = int(os.getenv("MIN_CLAUSE_WORDS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

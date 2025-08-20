from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Union, Optional
import logging
from functools import lru_cache

# Try to import settings, fallback to defaults if not available
try:
    from app.utils.settings import UNFAIR_MODEL, ALTERNATIVE_UNFAIR_MODELS, UNFAIR_LABELS, UNFAIR_SEEDS
except ImportError:
    UNFAIR_MODEL = "marmolpen3/lexglue-unfair-tos"
    ALTERNATIVE_UNFAIR_MODELS = ["CodeHima/TOSRobertaV2", "nlpaueb/legal-bert-base-uncased"]
    UNFAIR_LABELS = ["limitation_of_liability", "unilateral_termination", "unilateral_change", 
                     "content_removal", "contract_by_using", "choice_of_law", "jurisdiction", "arbitration"]
    UNFAIR_SEEDS = {
        "limitation_of_liability": "To the extent permitted by law, we are not liable for any damages.",
        "unilateral_termination": "We may terminate your account at any time without notice.",
        "unilateral_change": "We may change these terms at any time without your consent.",
        "content_removal": "We may remove any content at our sole discretion.",
        "contract_by_using": "By using the service, you agree to these terms.",
        "choice_of_law": "These terms are governed by the laws of California.",
        "jurisdiction": "Any disputes will be subject to the courts of New York.",
        "arbitration": "Any dispute shall be resolved by binding arbitration."
    }

logger = logging.getLogger(__name__)

@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_name: str):
    """
    Cached model loading to avoid reloading on every instantiation.
    Uses LRU cache to keep up to 2 models in memory.
    """
    import time
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_time = time.time() - t
    logger.info(f"Tokenizer loaded in {tokenizer_time:.2f}s")
    
    # Load model
    t = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Use Apple Silicon if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model_time = time.time() - t
    logger.info(f"Model loaded in {model_time:.2f}s on device: {device}")
    
    return tokenizer, model, device

class UnfairClauseModel:
    """
    Wrapper for pre-fine-tuned UNFAIR-ToS models from Hugging Face.
    Handles batch prediction with proper device management.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the unfair clause detection model.
        
        Args:
            model_name: HuggingFace model name, defaults to UNFAIR_MODEL setting
            device: Device to run model on, will be set by cached loader
        """
        self.model_name = model_name or UNFAIR_MODEL
        # Device will be set by _load_model_and_tokenizer
        self.device = None
        
        logger.info(f"Initializing UnfairClauseModel with: {self.model_name}")
        
        # Load model with fallback options and caching
        self._load_model()
        
    def _load_model(self):
        """Load the model with fallback options and caching."""
        models_to_try = [self.model_name] + ALTERNATIVE_UNFAIR_MODELS
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                
                # Use cached loading function
                self.tokenizer, self.model, self.device = _load_model_and_tokenizer(model_name)
                
                # Get label mapping and calibrate if needed
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    self.id2label = self.model.config.id2label
                    
                    # Check if we have generic LABEL_X labels that need calibration
                    if all(str(v).startswith("LABEL_") for v in self.id2label.values()):
                        logger.info("Detected generic labels, calibrating with seed examples...")
                        self._calibrate_labels()
                    
                else:
                    # Default binary classification labels
                    self.id2label = {0: "fair", 1: "unfair"}
                    logger.warning("No id2label found, using default binary labels")
                
                self.model_name = model_name  # Update to the successfully loaded model
                logger.info(f"Successfully loaded model: {model_name}")
                logger.info(f"Label mapping: {self.id2label}")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                continue
        
        raise Exception("Failed to load any unfair clause detection model")
    
    def _calibrate_labels(self):
        """
        Calibrate label mapping using seed examples to determine correct order.
        This empirically maps model indices to meaningful category names.
        """
        try:
            logger.info("Starting label calibration with seed examples...")
            
            # Get predictions for each seed example
            idx_by_label = {}
            
            with torch.inference_mode():
                for category_name, seed_text in UNFAIR_SEEDS.items():
                    # Tokenize the seed example
                    inputs = self.tokenizer(
                        [seed_text], 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True,
                        max_length=512
                    ).to(self.device)
                    
                    # Get model prediction
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0]
                    predicted_idx = int(torch.argmax(logits).item())
                    
                    idx_by_label[category_name] = predicted_idx
                    logger.debug(f"Seed '{category_name}' -> model index {predicted_idx}")
            
            # Create ordered label list based on model predictions
            num_labels = len(self.id2label)
            ordered_labels = [None] * num_labels
            
            # Map predicted indices to category names
            for category_name, idx in idx_by_label.items():
                if idx < num_labels and ordered_labels[idx] is None:
                    ordered_labels[idx] = category_name
                else:
                    logger.warning(f"Index conflict or out of range for {category_name} -> {idx}")
            
            # Fill any gaps with remaining labels
            remaining_labels = [label for label in UNFAIR_LABELS if label not in ordered_labels]
            for i in range(num_labels):
                if ordered_labels[i] is None and remaining_labels:
                    ordered_labels[i] = remaining_labels.pop(0)
                elif ordered_labels[i] is None:
                    ordered_labels[i] = f"unknown_{i}"
            
            # Update model configuration
            self.model.config.id2label = {i: label for i, label in enumerate(ordered_labels)}
            self.model.config.label2id = {label: i for i, label in enumerate(ordered_labels)}
            self.id2label = self.model.config.id2label
            
            logger.info("Label calibration completed successfully")
            logger.info(f"Calibrated mapping: {self.id2label}")
            
        except Exception as e:
            logger.error(f"Label calibration failed: {e}")
            # Fallback to tentative mapping
            tentative_mapping = {i: label for i, label in enumerate(UNFAIR_LABELS[:len(self.id2label)])}
            self.model.config.id2label = tentative_mapping
            self.model.config.label2id = {label: i for i, label in tentative_mapping.items()}
            self.id2label = tentative_mapping
            logger.warning(f"Using tentative mapping: {self.id2label}")
    
    @torch.inference_mode()
    def predict(self, clauses: Union[str, List[str]], batch_size: int = 16) -> List[Dict]:
        """
        Predict unfair clauses from input text(s).
        
        Args:
            clauses: Single clause string or list of clause strings
            batch_size: Batch size for processing multiple clauses
            
        Returns:
            List of prediction dictionaries with clause, label, confidence, and probabilities
        """
        # Handle single string input
        if isinstance(clauses, str):
            clauses = [clauses]
        
        if not clauses:
            return []
        
        logger.info(f"Predicting unfairness for {len(clauses)} clauses")
        
        results = []
        
        # Process in batches
        for i in range(0, len(clauses), batch_size):
            batch = clauses[i:i + batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
        
        logger.info(f"Completed predictions for {len(results)} clauses")
        return results
    
    def _predict_batch(self, batch: List[str]) -> List[Dict]:
        """Process a single batch of clauses."""
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            
            # Process results
            batch_results = []
            for clause, probs in zip(batch, probabilities):
                # Get predicted class
                predicted_class_idx = max(range(len(probs)), key=lambda k: probs[k])
                predicted_label = self.id2label[predicted_class_idx]
                confidence = probs[predicted_class_idx]
                
                batch_results.append({
                    "clause": clause,
                    "label": predicted_label,
                    "confidence": float(confidence),
                    "probabilities": {self.id2label[i]: float(prob) for i, prob in enumerate(probs)},
                    "model_used": self.model_name
                })
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            # Return default results for failed batch
            return [{
                "clause": clause,
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {},
                "error": str(e),
                "model_used": self.model_name
            } for clause in batch]
    
    def is_unfair(self, prediction: Dict, threshold: float = 0.6) -> bool:
        """
        Determine if a prediction indicates an unfair clause.
        
        Args:
            prediction: Prediction dictionary from predict()
            threshold: Confidence threshold for unfair classification
            
        Returns:
            True if clause is predicted to be unfair above threshold
        """
        label = prediction.get("label", "").lower()
        confidence = prediction.get("confidence", 0.0)
        
        # Check if confidence meets threshold first
        if confidence < threshold:
            return False
        
        # Check various unfair label patterns
        fair_labels = {"fair", "label_0", "0"}
        
        # If it's explicitly a fair label, return False
        if label in fair_labels:
            return False
        
        # Check for explicit unfair patterns
        unfair_patterns = [
            "unfair", "potentially_unfair", "clearly_unfair", 
            "limitation_of_liability", "unilateral_termination", "unilateral_change",
            "content_removal", "contract_by_using", "choice_of_law", 
            "jurisdiction", "arbitration"
        ]
        
        return any(pattern in label for pattern in unfair_patterns)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "labels": self.id2label,
            "num_classes": len(self.id2label)
        }

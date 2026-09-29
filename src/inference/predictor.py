"""
Inference engine and topic prediction for news headlines and article snippets.
Supports single-item predictions, batch inference, probability breakdowns, and Hugging Face pipeline integration.
"""

from typing import List, Dict, Any, Union, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from utils.custom_exception import InferenceException
from utils.helpers import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


class TopicPredictor:
    """
    Inference predictor for news topic classification.
    """

    def __init__(
        self,
        model_or_path: Union[str, Any],
        tokenizer_or_path: Optional[Union[str, Any]] = None,
        label_names: Optional[List[str]] = None,
        device: Optional[Any] = None,
        max_length: int = 128
    ):
        self.device = device or get_device()
        self.max_length = max_length

        # Load Model
        if isinstance(model_or_path, str):
            logger.info(f"Loading model checkpoint from {model_or_path}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_or_path)
        else:
            self.model = model_or_path

        self.model.to(self.device)
        self.model.eval()

        # Load Tokenizer
        if tokenizer_or_path is None and isinstance(model_or_path, str):
            tokenizer_or_path = model_or_path

        if isinstance(tokenizer_or_path, str):
            logger.info(f"Loading tokenizer from {tokenizer_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            self.tokenizer = tokenizer_or_path

        # Determine label mappings
        if label_names:
            self.label_names = label_names
        elif hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.label_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        else:
            self.label_names = ["World", "Sports", "Business", "Sci/Tech"]

        logger.info(f"TopicPredictor initialized on {self.device}. Labels: {self.label_names}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify a single news headline.

        Args:
            text: Input news text.

        Returns:
            Dictionary containing predicted label, confidence, and all class probabilities.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = float(probs[pred_idx].item())

            return {
                "text": text,
                "predicted_label": self.label_names[pred_idx],
                "predicted_id": pred_idx,
                "confidence": round(confidence, 4),
                "probabilities": {
                    name: round(float(probs[i].item()), 4)
                    for i, name in enumerate(self.label_names)
                },
            }
        except Exception as e:
            raise InferenceException(f"Failed to run inference on text: {e}")

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Classify a list of headlines in batches.

        Args:
            texts: List of text strings.
            batch_size: Micro-batch size for evaluation.

        Returns:
            List of prediction dictionaries.
        """
        try:
            results = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_probs = torch.softmax(outputs.logits, dim=-1)
                    batch_preds = torch.argmax(batch_probs, dim=-1).cpu().numpy()

                for j, text in enumerate(chunk):
                    pred_idx = int(batch_preds[j])
                    probs = batch_probs[j]
                    results.append({
                        "text": text,
                        "predicted_label": self.label_names[pred_idx],
                        "predicted_id": pred_idx,
                        "confidence": round(float(probs[pred_idx].item()), 4),
                        "probabilities": {
                            name: round(float(probs[c].item()), 4)
                            for c, name in enumerate(self.label_names)
                        },
                    })
            return results
        except Exception as e:
            raise InferenceException(f"Failed to run batch inference: {e}")

    def get_pipeline(self) -> Any:
        """Construct a standard Hugging Face classification pipeline."""
        return pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
        )

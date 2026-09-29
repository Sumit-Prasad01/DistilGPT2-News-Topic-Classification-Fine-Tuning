"""
Metrics computation module for DistilGPT2 News Topic Classification.
Supports fast single-pass C++ computation (csrc/fast_metrics.cpp) with an optimized Scikit-Learn fallback.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src.cpp_extension import get_fast_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4,
    use_cpp: bool = True
) -> Dict[str, float]:
    """
    Compute classification metrics: Accuracy, Weighted F1, Macro F1, Precision, and Recall.

    Args:
        predictions: 1D array of predicted class indices.
        labels: 1D array of reference class indices.
        num_classes: Total number of classes.
        use_cpp: Whether to attempt C++ acceleration.

    Returns:
        Dictionary of computed scalar metrics.
    """
    fast_metrics_fn = get_fast_metrics() if use_cpp else None

    # C++ Fast Path
    if fast_metrics_fn is not None:
        try:
            preds_tensor = torch.as_tensor(predictions, dtype=torch.int64)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            metrics = fast_metrics_fn(preds_tensor, labels_tensor, num_classes)
            # Add weighted precision/recall via sklearn for completeness
            metrics["precision_weighted"] = float(
                precision_score(labels, predictions, average="weighted", zero_division=0)
            )
            metrics["recall_weighted"] = float(
                recall_score(labels, predictions, average="weighted", zero_division=0)
            )
            return metrics
        except Exception as e:
            logger.warning(f"C++ fast_compute_metrics failed ({e}); falling back to scikit-learn.")

    # Scikit-Learn Fallback Path
    acc = float(accuracy_score(labels, predictions))
    f1_weighted = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    f1_macro = float(f1_score(labels, predictions, average="macro", zero_division=0))
    prec_weighted = float(precision_score(labels, predictions, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(labels, predictions, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
    }


def get_trainer_compute_metrics(num_classes: int = 4, use_cpp: bool = True):
    """
    Generate compute_metrics function matching Hugging Face Trainer API:
        compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]
    """
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return compute_classification_metrics(predictions, labels, num_classes=num_classes, use_cpp=use_cpp)

    return compute_metrics

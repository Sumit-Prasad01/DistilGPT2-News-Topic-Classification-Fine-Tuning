"""
Evaluation, metrics computation, and diagnostic visualization package.
"""

from src.evaluation.metrics import (
    compute_classification_metrics,
    get_trainer_compute_metrics,
)
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_curves,
    extract_top_misclassifications,
)
from src.evaluation.evaluator import ModelEvaluator

__all__ = [
    "compute_classification_metrics",
    "get_trainer_compute_metrics",
    "plot_confusion_matrix",
    "plot_per_class_accuracy",
    "plot_training_curves",
    "extract_top_misclassifications",
    "ModelEvaluator",
]

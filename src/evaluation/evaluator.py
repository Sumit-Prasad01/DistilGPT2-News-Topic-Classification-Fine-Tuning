"""
Comprehensive model evaluator for DistilGPT2 News Topic Classification.
Orchestrates test split evaluation, per-class classification reports, diagnostic plots, and error reports.
"""

import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    extract_top_misclassifications,
)
from utils.custom_exception import EvaluationException
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates fine-tuned classification models on test splits, generating full metrics and artifacts.
    """

    def __init__(self, label_names: List[str], output_dir: str = "outputs"):
        self.label_names = label_names
        self.num_classes = len(label_names)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        raw_texts: Optional[List[str]] = None,
        use_cpp: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate predicted class indices against reference labels.

        Args:
            predictions: 1D array of predicted class indices.
            labels: 1D array of true class indices.
            probabilities: Softmax probabilities of shape (num_samples, num_classes).
            raw_texts: Optional list of raw text headlines for error extraction.
            use_cpp: Whether to use C++ acceleration for metrics.

        Returns:
            Dictionary containing metrics, classification report, and artifact file paths.
        """
        try:
            logger.info("Computing evaluation metrics and classification report...")

            # 1. Scalar summary metrics
            scalar_metrics = compute_classification_metrics(
                predictions=predictions,
                labels=labels,
                num_classes=self.num_classes,
                use_cpp=use_cpp,
            )

            # 2. Detailed classification report per class
            report_dict = classification_report(
                labels,
                predictions,
                target_names=self.label_names,
                output_dict=True,
                zero_division=0,
            )
            report_text = classification_report(
                labels,
                predictions,
                target_names=self.label_names,
                zero_division=0,
            )
            logger.info(f"\nClassification Report:\n{'='*55}\n{report_text}\n{'='*55}")

            # 3. Confusion Matrix
            cm = confusion_matrix(labels, predictions)

            # 4. Per-Class Accuracy
            true_counts = cm.sum(axis=1)
            per_class_accuracies = np.where(true_counts == 0, 0.0, cm.diagonal() / true_counts).tolist()

            for name, acc in zip(self.label_names, per_class_accuracies):
                scalar_metrics[f"acc_{name.replace('/', '_')}"] = float(acc)

            # 5. Generate and save diagnostic plots
            cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plot_confusion_matrix(cm, self.label_names, output_path=cm_path)

            acc_path = os.path.join(self.output_dir, "per_class_accuracy.png")
            plot_per_class_accuracy(per_class_accuracies, self.label_names, output_path=acc_path)

            artifacts = [cm_path, acc_path]

            # 6. Extract top confident misclassifications if texts & probabilities are supplied
            if raw_texts is not None and probabilities is not None:
                err_csv = os.path.join(self.output_dir, "top_misclassifications.csv")
                extract_top_misclassifications(
                    texts=raw_texts,
                    labels=labels,
                    predictions=predictions,
                    probabilities=probabilities,
                    label_names=self.label_names,
                    top_k=15,
                    output_csv=err_csv,
                )
                artifacts.append(err_csv)

            # 7. Export JSON summary
            summary = {
                "metrics": scalar_metrics,
                "per_class_accuracy": {name: acc for name, acc in zip(self.label_names, per_class_accuracies)},
                "classification_report": report_dict,
                "artifacts": artifacts,
            }

            summary_json_path = os.path.join(self.output_dir, "evaluation_summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            artifacts.append(summary_json_path)

            return summary

        except Exception as e:
            raise EvaluationException(f"Failed to evaluate model predictions: {e}")

    def evaluate_with_trainer(
        self,
        trainer: Any,
        test_dataset: Any,
        raw_texts: Optional[List[str]] = None,
        use_cpp: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate directly via a trained Hugging Face Trainer on a test dataset.
        """
        logger.info(f"Running batch predictions on test dataset ({len(test_dataset)} samples)...")
        pred_output = trainer.predict(test_dataset)

        logits = pred_output.predictions
        # Softmax probabilities
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        predictions = np.argmax(logits, axis=-1)
        labels = pred_output.label_ids

        return self.evaluate_predictions(
            predictions=predictions,
            labels=labels,
            probabilities=probs,
            raw_texts=raw_texts,
            use_cpp=use_cpp,
        )

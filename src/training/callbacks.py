"""
Hugging Face Trainer callbacks for DistilGPT2 News Topic Classification.
Logs training losses, validation metrics, and learning rates to MLflow at step and epoch resolution.
"""

from typing import Any, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from src.tracking.mlflow_manager import MLflowManager
from utils.logger import get_logger

logger = get_logger(__name__)


class MLflowLoggingCallback(TrainerCallback):
    """
    Custom callback to stream Hugging Face Trainer metrics into MLflow.
    """

    def __init__(self, mlflow_manager: MLflowManager):
        self.mlflow = mlflow_manager

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Called whenever Trainer logs intermediate training loss or learning rate."""
        if logs and self.mlflow:
            step = state.global_step
            # Extract only numeric scalar metrics
            scalar_metrics = {
                k: float(v) for k, v in logs.items() if isinstance(v, (int, float))
            }
            if scalar_metrics:
                self.mlflow.log_metrics(scalar_metrics, step=step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs
    ):
        """Called after an evaluation loop finishes on validation split."""
        if metrics and self.mlflow:
            step = state.global_step
            eval_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            if eval_metrics:
                self.mlflow.log_metrics(eval_metrics, step=step)
                logger.info(
                    f"Epoch {state.epoch:.1f} Evaluation: "
                    f"Loss={eval_metrics.get('eval_loss', 'N/A')} | "
                    f"Weighted F1={eval_metrics.get('eval_f1_weighted', 'N/A')} | "
                    f"Accuracy={eval_metrics.get('eval_accuracy', 'N/A')}"
                )

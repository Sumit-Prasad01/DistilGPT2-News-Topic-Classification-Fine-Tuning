"""
MLflow experiment tracking manager for DistilGPT2 News Topic Classification.
Handles logging of hyperparameters, training curves, validation metrics, evaluation artifacts, and model weights.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from typing import Dict, Any, Optional
from contextlib import contextmanager
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
    import mlflow.transformers
    import mlflow.pytorch
except ImportError:
    mlflow = None


class MLflowManager:
    """
    Encapsulates MLflow experiment tracking workflows.
    Provides safe fallbacks if MLflow is not configured or disabled.
    """

    def __init__(self, mlflow_config: Any):
        self.config = mlflow_config
        self.is_enabled = mlflow is not None
        self.active_run = None

        if not self.is_enabled:
            logger.warning("MLflow library is not installed. Experiment tracking is disabled.")
            return

        tracking_uri = getattr(self.config, "tracking_uri", "sqlite:///mlflow.db")
        experiment_name = getattr(self.config, "experiment_name", "DistilGPT2-News-Topic-Classification")

        try:
            if not tracking_uri.startswith(("sqlite:", "http://", "https://")):
                os.makedirs(tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow initialized: Experiment '{experiment_name}' at '{tracking_uri}'")
        except Exception as e:
            logger.error(f"Failed to set MLflow tracking URI or experiment: {e}")
            self.is_enabled = False

    def start_run(self, run_name: Optional[str] = None) -> Optional[Any]:
        """Start a new MLflow tracking run."""
        if not self.is_enabled:
            return None

        name = run_name or getattr(self.config, "run_name", None)
        try:
            self.active_run = mlflow.start_run(run_name=name)
            logger.info(
                f"Started MLflow run: ID='{self.active_run.info.run_id}', Name='{self.active_run.info.run_name}'"
            )
            return self.active_run
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        if not self.is_enabled or not self.active_run:
            return
        try:
            # Flatten or stringify nested structures
            flat_params = {}
            for k, v in params.items():
                if isinstance(v, (list, tuple, dict)):
                    flat_params[k] = str(v)
                elif v is not None:
                    flat_params[k] = v
            mlflow.log_params(flat_params)
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log scalar metrics at an optional global step."""
        if not self.is_enabled or not self.active_run:
            return
        try:
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_metrics[k] = float(v)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a file or directory as an artifact to the current run."""
        if not self.is_enabled or not self.active_run:
            return
        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
                logger.info(f"Artifact logged to MLflow: {local_path}")
            else:
                logger.warning(f"Artifact file not found: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact '{local_path}' to MLflow: {e}")

    def log_model(self, model: Any, tokenizer: Any, artifact_path: str = "model") -> None:
        """Log trained model and tokenizer to MLflow model registry."""
        if not self.is_enabled or not self.active_run:
            return
        if not getattr(self.config, "log_models", True):
            return

        try:
            logger.info("Exporting model & tokenizer artifacts to MLflow registry...")
            components = {"model": model, "tokenizer": tokenizer}
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path=artifact_path,
            )
            logger.info("Successfully registered model to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to register model to MLflow: {e}")

    def end_run(self) -> None:
        """End the current MLflow tracking run."""
        if self.is_enabled and self.active_run:
            try:
                mlflow.end_run()
                logger.info("Ended active MLflow run.")
            except Exception as e:
                logger.warning(f"Error closing MLflow run: {e}")
            finally:
                self.active_run = None

    @contextmanager
    def run_context(self, run_name: Optional[str] = None):
        """Context manager for an MLflow run."""
        self.start_run(run_name=run_name)
        try:
            yield self
        finally:
            self.end_run()

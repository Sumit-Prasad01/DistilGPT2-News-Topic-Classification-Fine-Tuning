"""
Modular training pipeline orchestrator for DistilGPT2 News Topic Classification.
Builds TrainingArguments optimized for NVIDIA RTX 3050 (4GB VRAM) and coordinates fine-tuning.
"""

import os
from typing import Any, Optional, Dict
import torch
from transformers import Trainer, TrainingArguments
from src.data.collator import GPTDataCollator
from src.evaluation.metrics import get_trainer_compute_metrics
from src.evaluation.visualizer import plot_training_curves
from src.training.callbacks import MLflowLoggingCallback
from utils.custom_exception import TrainingException
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsClassifierTrainer:
    """
    Encapsulates fine-tuning orchestration using Hugging Face Trainer with MLflow tracking and C++ acceleration.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Any,
        train_dataset: Any,
        eval_dataset: Any,
        mlflow_manager: Optional[Any] = None,
        device: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.mlflow_manager = mlflow_manager
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainer = self._build_trainer()

    def _build_training_args(self) -> TrainingArguments:
        """Construct Hugging Face TrainingArguments from configuration."""
        tc = self.config.training
        output_dir = getattr(tc, "output_dir", "./gpt-news-model")

        # Determine precision flags based on device capabilities
        is_cuda = self.device.type == "cuda"
        fp16_enabled = getattr(tc, "fp16", False) and is_cuda
        bf16_enabled = getattr(tc, "bf16", False) and is_cuda and torch.cuda.is_bf16_supported()

        # Fused AdamW optimizer is CUDA-only; fall back gracefully on CPU/MPS
        optimizer_choice = getattr(tc, "optim", "adamw_torch")
        if optimizer_choice == "adamw_torch_fused" and not is_cuda:
            optimizer_choice = "adamw_torch"

        batch_size = getattr(tc, "batch_size", 8)
        grad_accum = getattr(tc, "gradient_accumulation_steps", 2)
        eval_batch_size = getattr(tc, "eval_batch_size", 16)
        num_epochs = getattr(tc, "num_epochs", 3)
        learning_rate = float(getattr(tc, "learning_rate", 2e-5))
        weight_decay = float(getattr(tc, "weight_decay", 0.01))
        seed = int(getattr(tc, "seed", 42))

        logger.info(
            f"Configuring Training: Batch={batch_size}x{grad_accum} (Effective={batch_size*grad_accum}), "
            f"Epochs={num_epochs}, LR={learning_rate}, FP16={fp16_enabled}, Optim={optimizer_choice}"
        )

        # 2026 Hugging Face standard arguments (eval_strategy replaces evaluation_strategy)
        args_kwargs = {
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "per_device_eval_batch_size": eval_batch_size,
            "num_train_epochs": num_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": getattr(tc, "warmup_ratio", 0.05),
            "eval_strategy": getattr(tc, "eval_strategy", "epoch"),
            "save_strategy": getattr(tc, "save_strategy", "epoch"),
            "save_total_limit": getattr(tc, "save_total_limit", 1),
            "load_best_model_at_end": getattr(tc, "load_best_model_at_end", True),
            "metric_for_best_model": getattr(tc, "metric_for_best_model", "f1_weighted"),
            "fp16": fp16_enabled,
            "bf16": bf16_enabled,
            "optim": optimizer_choice,
            "logging_steps": getattr(tc, "logging_steps", 25),
            "report_to": ["none"],  # We route logging directly via our custom MLflowLoggingCallback
            "seed": seed,
            "dataloader_num_workers": getattr(tc, "dataloader_num_workers", 0),
            "dataloader_pin_memory": getattr(tc, "dataloader_pin_memory", True) and is_cuda,
        }

        return TrainingArguments(**args_kwargs)

    def _build_trainer(self) -> Trainer:
        """Instantiate Hugging Face Trainer with callbacks and custom collation."""
        training_args = self._build_training_args()

        # Dynamic Left-Padding Collator (accelerated via C++ when available)
        use_cpp_collator = getattr(self.config.training, "use_cpp_collator", True)
        data_collator = GPTDataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=getattr(self.config.data, "max_length", 128),
            use_cpp=use_cpp_collator,
        )

        # Metrics computation function
        num_classes = getattr(self.config.model, "num_labels", 4)
        use_cpp_metrics = getattr(self.config.training, "use_cpp_metrics", True)
        compute_metrics = get_trainer_compute_metrics(
            num_classes=num_classes,
            use_cpp=use_cpp_metrics
        )

        callbacks = []
        if self.mlflow_manager:
            callbacks.append(MLflowLoggingCallback(self.mlflow_manager))

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def train(self) -> Dict[str, Any]:
        """
        Execute training, plot training curves, and save final checkpoint.

        Returns:
            Dictionary containing training stats and output paths.
        """
        try:
            logger.info("Starting DistilGPT2 fine-tuning loop...")
            train_result = self.trainer.train()

            output_dir = self.trainer.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            # Save best model weights and tokenizer
            logger.info(f"Saving fine-tuned model and tokenizer to {output_dir}")
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Generate and log training loss & F1 curves
            curves_path = os.path.join(output_dir, "training_curves.png")
            plot_training_curves(self.trainer.state.log_history, output_path=curves_path)

            if self.mlflow_manager and os.path.exists(curves_path):
                self.mlflow_manager.log_artifact(curves_path)

            logger.info("Fine-tuning completed successfully!")
            return {
                "train_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "metrics": train_result.metrics,
                "output_dir": output_dir,
                "curves_path": curves_path,
            }
        except Exception as e:
            raise TrainingException(f"Error during model training: {e}")

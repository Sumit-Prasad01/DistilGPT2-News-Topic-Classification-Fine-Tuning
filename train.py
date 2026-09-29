"""
Main CLI training entry point for DistilGPT2 News Topic Classification.
Executes end-to-end fine-tuning with C++ acceleration, MLflow tracking, and comprehensive diagnostics.
"""

import argparse
import os
import sys

# Crucial Windows low-VRAM guard: Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from src.config import load_config
from src.data.data_loader import load_ag_news_dataset
from src.data.tokenizer import get_gpt_tokenizer, tokenize_dataset
from src.evaluation.evaluator import ModelEvaluator
from src.models.model_factory import build_distilgpt2_classifier
from src.tracking.mlflow_manager import MLflowManager
from src.training.trainer import NewsClassifierTrainer
from utils.helpers import get_device, set_seed, Timer, get_system_info
from utils.logger import get_logger

logger = get_logger("TrainPipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT2 on AG News topic classification.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--no-cpp", action="store_true", help="Disable C++ ops acceleration.")
    parser.add_argument("--cpu", action="store_true", help="Force execution on CPU.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing DistilGPT2 Training Pipeline...")

    # 1. Load YAML configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.no_cpp:
        config.training.use_cpp_collator = False
        config.training.use_cpp_metrics = False

    # 2. Hardware and Seed Setup
    device = torch.device("cpu") if args.cpu else get_device()
    seed = int(getattr(config.training, "seed", 42))
    set_seed(seed)

    sys_info = get_system_info()
    logger.info(f"System Info: {sys_info}")
    logger.info(f"Target compute device: {device}")

    # 3. Load & Split Dataset
    with Timer("Dataset Loading & Balanced Sampling", logger=logger):
        dataset_dict = load_ag_news_dataset(config.data, config.model)

    # 4. Tokenizer Configuration
    with Timer("Tokenizer Initialization", logger=logger):
        tokenizer = get_gpt_tokenizer(
            model_name=config.model.model_name,
            max_length=config.data.max_length,
        )

    # 5. Tokenize Dataset
    with Timer("Dataset Tokenization", logger=logger):
        # We perform truncation during map, dynamic left-padding in collator
        tokenized_datasets = tokenize_dataset(
            dataset_dict=dataset_dict,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            dynamic_padding=True,
        )

    # 6. Initialize DistilGPT2 Classifier Model
    with Timer("Model Initialization", logger=logger):
        model = build_distilgpt2_classifier(
            model_config=config.model,
            tokenizer=tokenizer,
            device=device,
        )

    # 7. MLflow Experiment Tracking Setup
    mlflow_mgr = MLflowManager(config.mlflow)
    mlflow_mgr.start_run()

    # Flatten config parameters for MLflow logging
    all_params = {
        "model_name": config.model.model_name,
        "dataset_name": config.data.dataset_name,
        "train_size": len(dataset_dict["train"]),
        "val_size": len(dataset_dict["validation"]),
        "test_size": len(dataset_dict["test"]),
        "max_length": config.data.max_length,
        "num_epochs": config.training.num_epochs,
        "batch_size": config.training.batch_size,
        "grad_accum": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "fp16": config.training.fp16,
        "optim": config.training.optim,
        "use_cpp_collator": config.training.use_cpp_collator,
        "device": str(device),
    }
    mlflow_mgr.log_params(all_params)

    # 8. Train Model
    trainer_wrapper = NewsClassifierTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        mlflow_manager=mlflow_mgr,
        device=device,
    )

    with Timer("Model Fine-Tuning Loop", logger=logger):
        train_stats = trainer_wrapper.train()

    # 9. Test Split Evaluation & Diagnostics
    logger.info("Starting comprehensive test set evaluation...")
    evaluator = ModelEvaluator(
        label_names=config.model.label_names,
        output_dir=os.path.join(config.training.output_dir, "evaluation"),
    )

    raw_test_texts = [sample["text"] for sample in dataset_dict["test"]]
    eval_summary = evaluator.evaluate_with_trainer(
        trainer=trainer_wrapper.trainer,
        test_dataset=tokenized_datasets["test"],
        raw_texts=raw_test_texts,
        use_cpp=config.training.use_cpp_metrics,
    )

    # Log test metrics and artifacts to MLflow
    mlflow_test_metrics = {f"test_{k}": v for k, v in eval_summary["metrics"].items()}
    mlflow_mgr.log_metrics(mlflow_test_metrics)

    for artifact_path in eval_summary["artifacts"]:
        mlflow_mgr.log_artifact(artifact_path)

    # 10. Register Model Checkpoint into MLflow
    mlflow_mgr.log_model(trainer_wrapper.trainer.model, tokenizer, artifact_path="model")
    mlflow_mgr.end_run()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE EXECUTION COMPLETED")
    logger.info(f"Test Accuracy:    {eval_summary['metrics']['accuracy']:.2%}")
    logger.info(f"Test Weighted F1: {eval_summary['metrics']['f1_weighted']:.4f}")
    logger.info(f"Model saved to:   {config.training.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

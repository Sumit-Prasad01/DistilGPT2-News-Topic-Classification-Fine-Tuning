"""
Standalone CLI evaluation entry point for DistilGPT2 News Topic Classification.
Evaluates a fine-tuned model checkpoint on the test set or custom dataset.
"""

import argparse
import os
import torch
from src.config import load_config
from src.data.data_loader import load_ag_news_dataset
from src.data.tokenizer import get_gpt_tokenizer, tokenize_dataset
from src.evaluation.evaluator import ModelEvaluator
from src.inference.predictor import TopicPredictor
from utils.helpers import get_device, set_seed
from utils.logger import get_logger

logger = get_logger("EvaluatePipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned DistilGPT2 news classifier.")
    parser.add_argument("--model-dir", type=str, default="./gpt-news-model", help="Path to saved model checkpoint.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--output-dir", type=str, default="./outputs/evaluation", help="Directory for plots and reports.")
    parser.add_argument("--no-cpp", action="store_true", help="Disable C++ metrics computation.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    device = get_device()
    set_seed(config.training.seed)

    logger.info(f"Loading model checkpoint from {args.model_dir}...")
    predictor = TopicPredictor(
        model_or_path=args.model_dir,
        tokenizer_or_path=args.model_dir,
        label_names=config.model.label_names,
        device=device,
        max_length=config.data.max_length,
    )

    logger.info("Loading test dataset split...")
    dataset_dict = load_ag_news_dataset(config.data, config.model)
    test_data = dataset_dict["test"]

    texts = [sample["text"] for sample in test_data]
    true_labels = [sample["label"] for sample in test_data]

    logger.info(f"Running inference on {len(texts)} test samples...")
    batch_results = predictor.predict_batch(texts, batch_size=config.training.eval_batch_size)

    predictions = [res["predicted_id"] for res in batch_results]
    probs = [list(res["probabilities"].values()) for res in batch_results]

    import numpy as np
    preds_arr = np.array(predictions)
    labels_arr = np.array(true_labels)
    probs_arr = np.array(probs)

    evaluator = ModelEvaluator(label_names=config.model.label_names, output_dir=args.output_dir)
    summary = evaluator.evaluate_predictions(
        predictions=preds_arr,
        labels=labels_arr,
        probabilities=probs_arr,
        raw_texts=texts,
        use_cpp=not args.no_cpp,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    for k, v in summary["metrics"].items():
        if isinstance(v, float):
            print(f"{k:<25}: {v:.4f}")
    print("=" * 60)
    print(f"Artifacts saved in: {args.output_dir}\n")


if __name__ == "__main__":
    main()

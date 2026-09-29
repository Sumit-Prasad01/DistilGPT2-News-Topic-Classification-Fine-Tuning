"""
CLI prediction tool for DistilGPT2 News Topic Classification.
Supports single headline prediction and interactive shell mode.
"""

import argparse
import json
import sys
from src.config import load_config
from src.inference.predictor import TopicPredictor
from utils.helpers import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict news topic using fine-tuned DistilGPT2.")
    parser.add_argument("--model-dir", type=str, default="./gpt-news-model", help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--text", type=str, default=None, help="News headline or article snippet to classify.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prediction shell.")
    return parser.parse_args()


def display_prediction(result: dict):
    print("\n" + "=" * 50)
    print(f"Text:             {result['text']}")
    print(f"Predicted Topic:  \033[1m\033[32m{result['predicted_label']}\033[0m")
    print(f"Confidence:       {result['confidence']:.2%}")
    print("-" * 50)
    print("Class Probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {label:<10}: {prob:>6.2%} {bar}")
    print("=" * 50 + "\n")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    predictor = TopicPredictor(
        model_or_path=args.model_dir,
        label_names=config.model.label_names,
        device=device,
        max_length=config.data.max_length,
    )

    if args.text:
        res = predictor.predict(args.text)
        display_prediction(res)
    elif args.interactive:
        print("\n=== DistilGPT2 News Classification Interactive Shell ===")
        print("Type a news headline and press Enter. (Type 'quit' or 'exit' to exit)\n")
        while True:
            try:
                line = input("Headline > ").strip()
                if not line:
                    continue
                if line.lower() in ("quit", "exit", "q"):
                    print("Exiting...")
                    break
                res = predictor.predict(line)
                display_prediction(res)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
    else:
        sample_headlines = [
            "NASA's James Webb Space Telescope discovers ancient distant galaxy",
            "Federal Reserve signals potential interest rate cuts amid economic slowdown",
            "Real Madrid secures thrilling 3-2 victory in Champions League semi-final",
            "United Nations summit addresses urgent global climate crisis negotiations",
        ]
        print("No --text provided. Running demonstration on sample headlines:\n")
        for sample in sample_headlines:
            res = predictor.predict(sample)
            display_prediction(res)


if __name__ == "__main__":
    main()

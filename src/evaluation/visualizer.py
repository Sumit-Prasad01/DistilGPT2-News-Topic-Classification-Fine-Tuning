"""
Visualization and diagnostic utilities for DistilGPT2 News Topic Classification.
Generates confusion matrix heatmaps, per-class accuracy charts, training curves, and error analysis reports.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    output_path: str = "outputs/confusion_matrix.png",
    normalize: bool = False
) -> str:
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        label_names: List of category names.
        output_path: Target PNG file path.
        normalize: Whether to normalize matrix entries as proportions.

    Returns:
        Absolute path to saved PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    display_cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    fmt = ".2%" if normalize else "d"

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        display_cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True,
    )
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix — Test Set"
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Topic", fontsize=11)
    plt.ylabel("True Topic", fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")
    return output_path


def plot_per_class_accuracy(
    accuracies: List[float],
    label_names: List[str],
    output_path: str = "outputs/per_class_accuracy.png"
) -> str:
    """
    Plot and save bar chart of accuracy per news topic.

    Args:
        accuracies: List of accuracy floats (between 0.0 and 1.0).
        label_names: List of category names.
        output_path: Target PNG file path.

    Returns:
        Absolute path to saved PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    bar_colors = colors[: len(label_names)]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(label_names, accuracies, color=bar_colors, edgecolor="black", width=0.55)
    plt.bar_label(
        bars,
        fmt="%.1f%%",
        labels=[f"{a * 100:.1f}%" for a in accuracies],
        padding=3,
        fontsize=10,
        fontweight="bold"
    )

    plt.title("Per-Class Accuracy — Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("News Category", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class accuracy chart saved to: {output_path}")
    return output_path


def plot_training_curves(
    log_history: List[Dict[str, Any]],
    output_path: str = "outputs/training_curves.png"
) -> Optional[str]:
    """
    Plot training loss and validation metrics across epochs from Hugging Face log_history.

    Args:
        log_history: Trainer state log_history list.
        output_path: Output PNG path.

    Returns:
        Path to saved PNG or None if insufficient logs.
    """
    train_losses = []
    eval_losses = []
    eval_f1s = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_losses.append((entry["epoch"], entry["loss"]))
        if "eval_loss" in entry and "epoch" in entry:
            eval_losses.append((entry["epoch"], entry["eval_loss"]))
        if "eval_f1_weighted" in entry and "epoch" in entry:
            eval_f1s.append((entry["epoch"], entry["eval_f1_weighted"]))

    if not train_losses and not eval_losses:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    if train_losses:
        tr_epochs, tr_vals = zip(*train_losses)
        axes[0].plot(tr_epochs, tr_vals, label="Train Loss", color="#4e79a7", marker="o")
    if eval_losses:
        ev_epochs, ev_vals = zip(*eval_losses)
        axes[0].plot(ev_epochs, ev_vals, label="Val Loss", color="#e15759", marker="s", linestyle="--")
    axes[0].set_title("Loss Trajectory", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Validation F1 curve
    if eval_f1s:
        f1_epochs, f1_vals = zip(*eval_f1s)
        axes[1].plot(f1_epochs, f1_vals, label="Val Weighted F1", color="#2ca02c", marker="^")
        axes[1].set_title("Validation F1 Score", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Weighted F1")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to: {output_path}")
    return output_path


def extract_top_misclassifications(
    texts: List[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: List[str],
    top_k: int = 10,
    output_csv: Optional[str] = "outputs/top_misclassifications.csv"
) -> pd.DataFrame:
    """
    Extract the top K misclassified examples where the model had highest confidence.

    Args:
        texts: Raw text inputs.
        labels: True integer labels.
        predictions: Predicted integer labels.
        probabilities: Softmax probabilities of shape (num_samples, num_classes).
        label_names: List of category names.
        top_k: Number of highest-confidence error examples to return.
        output_csv: Optional CSV path to save the error report.

    Returns:
        Pandas DataFrame of top misclassifications.
    """
    misclassified_mask = labels != predictions
    indices = np.where(misclassified_mask)[0]

    if len(indices) == 0:
        logger.info("No misclassifications found!")
        return pd.DataFrame()

    records = []
    for idx in indices:
        true_label = label_names[labels[idx]]
        pred_label = label_names[predictions[idx]]
        confidence = float(probabilities[idx][predictions[idx]])
        records.append({
            "text": texts[idx],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="confidence", ascending=False).head(top_k).reset_index(drop=True)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"Top {len(df)} misclassified examples saved to: {output_csv}")

    return df

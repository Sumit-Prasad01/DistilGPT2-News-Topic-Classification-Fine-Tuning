"""
Data loading and balanced subset extraction for AG News dataset.
Mirrors the balanced sampling methodology from the reference Colab notebook.
"""

from typing import Any
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from utils.custom_exception import DataProcessingException
from utils.logger import get_logger

logger = get_logger(__name__)


def sample_balanced(dataset_split: Dataset, n_per_class: int, num_labels: int, seed: int = 42) -> Dataset:
    """
    Sample exactly n_per_class examples from each class, returning a balanced dataset.

    Args:
        dataset_split: Hugging Face Dataset split.
        n_per_class: Number of samples per class.
        num_labels: Total number of classes.
        seed: Random seed for reproducibility.

    Returns:
        Balanced and shuffled Hugging Face Dataset.
    """
    try:
        indices = []
        labels = dataset_split["label"]

        for class_id in range(num_labels):
            class_indices = [i for i, l in enumerate(labels) if l == class_id]
            if len(class_indices) < n_per_class:
                raise ValueError(
                    f"Class {class_id} only has {len(class_indices)} examples, requested {n_per_class}."
                )

            # Reproducible class selection
            rng = np.random.default_rng(seed + class_id)
            chosen = rng.choice(class_indices, size=n_per_class, replace=False).tolist()
            indices.extend(chosen)

        # Interleave classes to avoid sequential clustering
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        return dataset_split.select(indices)
    except Exception as e:
        raise DataProcessingException(f"Failed to create balanced sample: {e}")


def load_ag_news_dataset(data_config: Any, model_config: Any) -> DatasetDict:
    """
    Download raw AG News dataset and create balanced train, validation, and test splits.

    Args:
        data_config: Data configuration with train_size, val_size, test_size, and seed.
        model_config: Model configuration containing num_labels.

    Returns:
        DatasetDict containing 'train', 'validation', and 'test' splits.
    """
    try:
        logger.info(f"Loading raw dataset: {data_config.dataset_name}")
        raw_dataset = load_dataset(data_config.dataset_name)

        num_labels = model_config.num_labels
        train_per_class = data_config.train_size // num_labels
        val_per_class = data_config.val_size // num_labels
        test_per_class = data_config.test_size // num_labels

        logger.info(
            f"Creating balanced splits: Train={data_config.train_size} ({train_per_class}/class), "
            f"Val={data_config.val_size} ({val_per_class}/class), "
            f"Test={data_config.test_size} ({test_per_class}/class)"
        )

        train_data = sample_balanced(
            raw_dataset["train"], train_per_class, num_labels, data_config.seed
        )
        test_full = sample_balanced(
            raw_dataset["test"], test_per_class + val_per_class, num_labels, data_config.seed
        )

        val_data = test_full.select(range(data_config.val_size))
        test_data = test_full.select(
            range(data_config.val_size, data_config.val_size + data_config.test_size)
        )

        dataset = DatasetDict({
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        })

        logger.info("Successfully created balanced DatasetDict splits.")
        return dataset
    except Exception as e:
        raise DataProcessingException(f"Error loading and preparing AG News splits: {e}")

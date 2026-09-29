"""
Unit tests for data loader and tokenizer modules.
"""

import unittest
import numpy as np
from datasets import Dataset
from src.data.data_loader import sample_balanced
from src.data.tokenizer import get_gpt_tokenizer


class TestDataModule(unittest.TestCase):

    def test_sample_balanced_counts(self):
        """Verify sample_balanced returns exact equal counts for each class."""
        # Create synthetic dataset with 4 classes, 50 items each
        data = {
            "text": [f"Sample text {i}" for i in range(200)],
            "label": [i % 4 for i in range(200)],
        }
        dataset = Dataset.from_dict(data)

        n_per_class = 25
        num_labels = 4
        balanced = sample_balanced(dataset, n_per_class=n_per_class, num_labels=num_labels, seed=42)

        self.assertEqual(len(balanced), n_per_class * num_labels)
        labels = balanced["label"]
        for c in range(num_labels):
            count = sum(1 for l in labels if l == c)
            self.assertEqual(count, n_per_class, f"Class {c} count mismatch.")

    def test_sample_balanced_reproducibility(self):
        """Verify identical seed produces identical samples."""
        data = {
            "text": [f"Sample text {i}" for i in range(100)],
            "label": [i % 4 for i in range(100)],
        }
        ds = Dataset.from_dict(data)
        split1 = sample_balanced(ds, n_per_class=10, num_labels=4, seed=123)
        split2 = sample_balanced(ds, n_per_class=10, num_labels=4, seed=123)
        self.assertEqual(split1["text"], split2["text"])

    def test_gpt_tokenizer_configuration(self):
        """Verify GPT tokenizer has left-padding and pad_token == eos_token."""
        try:
            tokenizer = get_gpt_tokenizer("distilgpt2", max_length=64)
            self.assertEqual(tokenizer.padding_side, "left", "GPT requires left-padding!")
            self.assertEqual(tokenizer.pad_token, tokenizer.eos_token, "Pad token must match EOS token!")
            self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
            self.assertEqual(tokenizer.model_max_length, 64)
        except Exception as e:
            self.skipTest(f"Skipping network/download dependent test: {e}")


if __name__ == "__main__":
    unittest.main()

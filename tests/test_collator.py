"""
Unit tests for dynamic left-padding collator.
Verifies correct padding direction, attention mask generation, and C++ vs Python output equivalence.
"""

import unittest
import torch
from src.data.collator import GPTDataCollator
from src.cpp_extension import is_cpp_available


class TestCollator(unittest.TestCase):

    def setUp(self):
        self.pad_token_id = 50256
        self.sample_batch = [
            {"input_ids": [101, 102, 103], "labels": 0},
            {"input_ids": [201, 202, 203, 204, 205], "labels": 1},
            {"input_ids": [301], "labels": 2},
        ]

    def test_python_collator_left_padding(self):
        """Verify Python fallback properly places pad tokens on the LEFT."""
        collator = GPTDataCollator(pad_token_id=self.pad_token_id, max_length=10, use_cpp=False)
        batch = collator(self.sample_batch)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Max length in batch is 5
        self.assertEqual(input_ids.shape, (3, 5))
        self.assertEqual(attention_mask.shape, (3, 5))
        self.assertEqual(labels.tolist(), [0, 1, 2])

        # Sample 0 has len 3: first 2 tokens must be PAD, remaining 3 must be [101, 102, 103]
        self.assertEqual(input_ids[0, 0].item(), self.pad_token_id)
        self.assertEqual(input_ids[0, 1].item(), self.pad_token_id)
        self.assertEqual(input_ids[0, 2:].tolist(), [101, 102, 103])

        # Mask for sample 0: [0, 0, 1, 1, 1]
        self.assertEqual(attention_mask[0].tolist(), [0, 0, 1, 1, 1])

        # Sample 2 has len 1: first 4 tokens must be PAD
        self.assertEqual(input_ids[2, :4].tolist(), [self.pad_token_id] * 4)
        self.assertEqual(input_ids[2, 4].item(), 301)
        self.assertEqual(attention_mask[2].tolist(), [0, 0, 0, 0, 1])

    def test_cpp_collator_equivalence(self):
        """If C++ extension is compiled, verify its output matches Python collator exactly."""
        if not is_cpp_available():
            self.skipTest("C++ extension not compiled; skipping equivalence test.")

        py_collator = GPTDataCollator(pad_token_id=self.pad_token_id, max_length=10, use_cpp=False)
        cpp_collator = GPTDataCollator(pad_token_id=self.pad_token_id, max_length=10, use_cpp=True)

        py_res = py_collator(self.sample_batch)
        cpp_res = cpp_collator(self.sample_batch)

        torch.testing.assert_close(py_res["input_ids"], cpp_res["input_ids"])
        torch.testing.assert_close(py_res["attention_mask"], cpp_res["attention_mask"])
        torch.testing.assert_close(py_res["labels"], cpp_res["labels"])


if __name__ == "__main__":
    unittest.main()

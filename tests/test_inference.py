"""
Unit tests for inference and prediction pipeline.
"""

import unittest
from unittest.mock import MagicMock
import torch
from src.inference.predictor import TopicPredictor


class TestInference(unittest.TestCase):

    def test_predictor_output_contract(self):
        """Verify TopicPredictor produces valid schema and probability distributions."""
        label_names = ["World", "Sports", "Business", "Sci/Tech"]

        # Create mock model returning fixed logits
        mock_model = MagicMock()
        mock_model.config.id2label = {i: name for i, name in enumerate(label_names)}
        mock_model.return_value.logits = torch.tensor([[1.0, 5.0, 0.5, 0.2]])  # Class 1 (Sports) is highest

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 64
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        predictor = TopicPredictor(
            model_or_path=mock_model,
            tokenizer_or_path=mock_tokenizer,
            label_names=label_names,
            device=torch.device("cpu"),
        )

        result = predictor.predict("Sample sports headline")

        self.assertIn("predicted_label", result)
        self.assertEqual(result["predicted_label"], "Sports")
        self.assertEqual(result["predicted_id"], 1)
        self.assertGreater(result["confidence"], 0.5)

        # Probabilities must sum to ~1.0
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()

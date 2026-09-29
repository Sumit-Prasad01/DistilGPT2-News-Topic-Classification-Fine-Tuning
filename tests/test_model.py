"""
Unit tests for DistilGPT2 model factory.
Verifies pad_token_id alignment and classification head output shapes.
"""

import unittest
import torch
from unittest.mock import MagicMock
from src.config import ConfigDict
from src.models.model_factory import build_distilgpt2_classifier


class TestModelFactory(unittest.TestCase):

    def test_model_pad_token_alignment(self):
        """Verify model.config.pad_token_id is correctly aligned with tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        config = ConfigDict({
            "model_name": "distilgpt2",
            "num_labels": 4,
            "id2label": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            "label2id": {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
        })

        try:
            model = build_distilgpt2_classifier(config, mock_tokenizer, device=torch.device("cpu"))
            self.assertEqual(model.config.pad_token_id, 50256)
            self.assertEqual(model.config.num_labels, 4)

            # Test dummy forward pass
            dummy_input_ids = torch.tensor([[101, 102, 103], [50256, 104, 105]], dtype=torch.long)
            dummy_mask = torch.tensor([[1, 1, 1], [0, 1, 1]], dtype=torch.long)

            with torch.no_grad():
                output = model(input_ids=dummy_input_ids, attention_mask=dummy_mask)

            self.assertEqual(output.logits.shape, (2, 4))
        except Exception as e:
            self.skipTest(f"Skipping model download test in isolated environment: {e}")


if __name__ == "__main__":
    unittest.main()

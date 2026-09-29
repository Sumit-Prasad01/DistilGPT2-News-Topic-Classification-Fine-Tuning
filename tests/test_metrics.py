"""
Unit tests for evaluation metrics computation.
Verifies mathematical correctness of Accuracy, Weighted F1, Macro F1, and C++ vs Python consistency.
"""

import unittest
import numpy as np
from src.evaluation.metrics import compute_classification_metrics
from src.cpp_extension import is_cpp_available


class TestMetrics(unittest.TestCase):

    def test_perfect_accuracy(self):
        """Verify perfect score when predictions match labels exactly."""
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        preds = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        metrics = compute_classification_metrics(preds, labels, num_classes=4, use_cpp=False)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1_weighted"], 1.0)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)

    def test_known_metrics_calculation(self):
        """Verify metrics on a known hand-calculated scenario."""
        # 4 samples: 2 correct, 2 wrong
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 1, 0])

        metrics = compute_classification_metrics(preds, labels, num_classes=2, use_cpp=False)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_weighted"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], 0.5)

    def test_cpp_metrics_equivalence(self):
        """If C++ is compiled, test that C++ outputs match scikit-learn within 1e-5."""
        if not is_cpp_available():
            self.skipTest("C++ extension not compiled; skipping equivalence test.")

        rng = np.random.default_rng(42)
        preds = rng.integers(0, 4, size=200)
        labels = rng.integers(0, 4, size=200)

        py_metrics = compute_classification_metrics(preds, labels, num_classes=4, use_cpp=False)
        cpp_metrics = compute_classification_metrics(preds, labels, num_classes=4, use_cpp=True)

        self.assertAlmostEqual(py_metrics["accuracy"], cpp_metrics["accuracy"], places=5)
        self.assertAlmostEqual(py_metrics["f1_weighted"], cpp_metrics["f1_weighted"], places=5)
        self.assertAlmostEqual(py_metrics["f1_macro"], cpp_metrics["f1_macro"], places=5)


if __name__ == "__main__":
    unittest.main()

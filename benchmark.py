"""
Benchmarking suite for DistilGPT2 News Topic Classification.
Measures performance speedup of C++ acceleration versus Python across:
1. Dynamic left-padding batch collation.
2. Metrics and confusion matrix computation.
3. Model inference latency percentiles (P50, P95, P99) and throughput.
"""

import argparse
import time
import numpy as np
import torch
from src.config import load_config
from src.cpp_extension import is_cpp_available, get_fast_collator, get_fast_metrics
from src.data.collator import GPTDataCollator
from src.evaluation.metrics import compute_classification_metrics
from utils.helpers import get_device, format_duration


def benchmark_collator(num_batches=200, batch_size=16, seq_len=90, pad_token_id=50256):
    print("\n" + "=" * 60)
    print("1. BENCHMARK: BATCH DYNAMIC COLLATION (C++ vs PYTHON)")
    print("=" * 60)

    # Generate synthetic variable-length batches
    batches = []
    rng = np.random.default_rng(42)
    for _ in range(num_batches):
        batch = []
        for _ in range(batch_size):
            length = int(rng.integers(low=20, high=seq_len))
            tokens = rng.integers(low=10, high=50000, size=length).tolist()
            batch.append({"input_ids": tokens, "labels": int(rng.integers(0, 4))})
        batches.append(batch)

    # Python collator
    py_collator = GPTDataCollator(pad_token_id=pad_token_id, max_length=128, use_cpp=False)
    # Warmup
    for b in batches[:10]:
        _ = py_collator(b)

    t0 = time.perf_counter()
    for b in batches:
        _ = py_collator(b)
    py_time = time.perf_counter() - t0

    print(f"Python Collator: Total Time = {py_time*1000:.2f} ms | "
          f"Throughput = {len(batches)*batch_size / py_time:.1f} samples/sec")

    # C++ collator
    cpp_collator = GPTDataCollator(pad_token_id=pad_token_id, max_length=128, use_cpp=True)
    if cpp_collator.fast_collator_fn is not None:
        for b in batches[:10]:
            _ = cpp_collator(b)
        t0 = time.perf_counter()
        for b in batches:
            _ = cpp_collator(b)
        cpp_time = time.perf_counter() - t0

        speedup = py_time / cpp_time if cpp_time > 0 else 1.0
        print(f"C++ Collator:    Total Time = {cpp_time*1000:.2f} ms | "
              f"Throughput = {len(batches)*batch_size / cpp_time:.1f} samples/sec")
        print(f"\033[1m\033[32m>>> C++ Acceleration Speedup: {speedup:.2f}x faster\033[0m")
    else:
        print("C++ Collator not compiled; run 'python setup.py build_ext --inplace' to benchmark C++ ops.")


def benchmark_metrics(num_trials=500, sample_size=1000, num_classes=4):
    print("\n" + "=" * 60)
    print("2. BENCHMARK: METRICS & CONFUSION MATRIX COMPUTATION")
    print("=" * 60)

    rng = np.random.default_rng(42)
    preds = rng.integers(0, num_classes, size=(num_trials, sample_size))
    labels = rng.integers(0, num_classes, size=(num_trials, sample_size))

    # Python Scikit-Learn
    t0 = time.perf_counter()
    for i in range(num_trials):
        _ = compute_classification_metrics(preds[i], labels[i], num_classes=num_classes, use_cpp=False)
    py_time = time.perf_counter() - t0
    print(f"Scikit-Learn:  Total Time = {py_time*1000:.2f} ms | Avg per eval = {py_time/num_trials*1000:.3f} ms")

    # C++ Metrics
    if is_cpp_available():
        t0 = time.perf_counter()
        for i in range(num_trials):
            _ = compute_classification_metrics(preds[i], labels[i], num_classes=num_classes, use_cpp=True)
        cpp_time = time.perf_counter() - t0
        speedup = py_time / cpp_time if cpp_time > 0 else 1.0
        print(f"C++ Metrics:   Total Time = {cpp_time*1000:.2f} ms | Avg per eval = {cpp_time/num_trials*1000:.3f} ms")
        print(f"\033[1m\033[32m>>> C++ Metrics Speedup: {speedup:.2f}x faster\033[0m")
    else:
        print("C++ fast_metrics not compiled; using Scikit-Learn.")


def benchmark_inference(model_path="./gpt-news-model", num_samples=100):
    print("\n" + "=" * 60)
    print("3. BENCHMARK: INFERENCE LATENCY & THROUGHPUT")
    print("=" * 60)

    try:
        from src.inference.predictor import TopicPredictor
        device = get_device()
        predictor = TopicPredictor(model_or_path=model_path, device=device)

        sample_text = "Global tech leaders gather to discuss next-generation artificial intelligence regulation."

        # Warmup
        for _ in range(5):
            _ = predictor.predict(sample_text)

        latencies = []
        for _ in range(num_samples):
            t0 = time.perf_counter()
            _ = predictor.predict(sample_text)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        print(f"Device: {device}")
        print(f"P50 Latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"Single-item Throughput: {1000.0 / np.mean(latencies):.1f} queries/sec")
    except Exception as e:
        print(f"Inference benchmark skipped (Model not yet trained at {model_path}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks.")
    parser.add_argument("--num-batches", type=int, default=150, help="Number of batches to collate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for collation.")
    parser.add_argument("--model-path", type=str, default="./gpt-news-model", help="Path to trained model.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("STARTING DISTILGPT2 PERFORMANCE BENCHMARK")
    print(f"C++ Extension Status: {'AVAILABLE' if is_cpp_available() else 'NOT COMPILED (Using Python)'}")
    print("=" * 60)

    benchmark_collator(num_batches=args.num_batches, batch_size=args.batch_size)
    benchmark_metrics()
    benchmark_inference(model_path=args.model_path)
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

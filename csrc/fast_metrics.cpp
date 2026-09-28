/**
 * Fast Single-Pass Metrics and Confusion Matrix Computation in C++.
 */

#include <torch/extension.h>
#include <vector>
#include <map>
#include <string>

std::map<std::string, double> fast_compute_metrics(
    const torch::Tensor& predictions,
    const torch::Tensor& references,
    int64_t num_classes)
{
    TORCH_CHECK(predictions.dim() == 1, "Predictions must be 1D tensor");
    TORCH_CHECK(references.dim() == 1, "References must be 1D tensor");
    TORCH_CHECK(predictions.sizes() == references.sizes(), "Predictions and references must have the same size");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    int64_t n = predictions.numel();
    if (n == 0) {
        return {
            {"accuracy", 0.0},
            {"f1_weighted", 0.0},
            {"f1_macro", 0.0}
        };
    }

    auto pred_cpu = predictions.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto ref_cpu = references.to(torch::kCPU).to(torch::kInt64).contiguous();

    auto pred_acc = pred_cpu.accessor<int64_t, 1>();
    auto ref_acc = ref_cpu.accessor<int64_t, 1>();

    // Confusion Matrix: cm[actual][predicted]
    std::vector<std::vector<int64_t>> cm(num_classes, std::vector<int64_t>(num_classes, 0));
    int64_t total_correct = 0;

    for (int64_t i = 0; i < n; ++i) {
        int64_t p = pred_acc[i];
        int64_t r = ref_acc[i];
        if (p >= 0 && p < num_classes && r >= 0 && r < num_classes) {
            cm[r][p]++;
            if (p == r) {
                total_correct++;
            }
        }
    }

    double accuracy = static_cast<double>(total_correct) / static_cast<double>(n);

    // Per-class metrics
    double macro_f1 = 0.0;
    double weighted_f1 = 0.0;

    for (int64_t c = 0; c < num_classes; ++c) {
        int64_t tp = cm[c][c];
        int64_t actual_count = 0;
        int64_t pred_count = 0;

        for (int64_t j = 0; j < num_classes; ++j) {
            actual_count += cm[c][j];
            pred_count += cm[j][c];
        }

        double precision = (pred_count > 0) ? static_cast<double>(tp) / static_cast<double>(pred_count) : 0.0;
        double recall = (actual_count > 0) ? static_cast<double>(tp) / static_cast<double>(actual_count) : 0.0;
        double f1 = 0.0;
        if (precision + recall > 1e-9) {
            f1 = (2.0 * precision * recall) / (precision + recall);
        }

        macro_f1 += f1;
        weighted_f1 += f1 * (static_cast<double>(actual_count) / static_cast<double>(n));
    }

    macro_f1 /= static_cast<double>(num_classes);

    return {
        {"accuracy", accuracy},
        {"f1_weighted", weighted_f1},
        {"f1_macro", macro_f1}
    };
}

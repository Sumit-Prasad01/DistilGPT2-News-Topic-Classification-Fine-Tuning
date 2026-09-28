/**
 * PyBind11 Python bindings for C++ ops.
 */

#include <torch/extension.h>
#include <vector>
#include <map>
#include <string>

// Declarations of functions implemented in fast_collator.cpp and fast_metrics.cpp
std::pair<torch::Tensor, torch::Tensor> fast_left_pad_collate(
    const std::vector<std::vector<int64_t>>& token_batch,
    int64_t pad_token_id,
    int64_t max_length);

std::map<std::string, double> fast_compute_metrics(
    const torch::Tensor& predictions,
    const torch::Tensor& references,
    int64_t num_classes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "High-performance C++ extensions for DistilGPT2 sequence classification";
    m.def(
        "fast_left_pad_collate",
        &fast_left_pad_collate,
        "Multithreaded dynamic left-padding collator for GPT models",
        py::arg("token_batch"),
        py::arg("pad_token_id"),
        py::arg("max_length")
    );
    m.def(
        "fast_compute_metrics",
        &fast_compute_metrics,
        "Fast single-pass accuracy and weighted/macro F1 computation",
        py::arg("predictions"),
        py::arg("references"),
        py::arg("num_classes")
    );
}

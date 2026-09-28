/**
 * Fast Dynamic Left-Padding Collator in C++ for GPT Sequence Classification.
 * Optimized for low VRAM and high DataLoader throughput.
 */

#include <torch/extension.h>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

std::pair<torch::Tensor, torch::Tensor> fast_left_pad_collate(
    const std::vector<std::vector<int64_t>>& token_batch,
    int64_t pad_token_id,
    int64_t max_length)
{
    int64_t batch_size = static_cast<int64_t>(token_batch.size());
    if (batch_size == 0) {
        auto empty_opts = torch::TensorOptions().dtype(torch::kInt64);
        return {torch::empty({0, 0}, empty_opts), torch::empty({0, 0}, empty_opts)};
    }

    // Determine the maximum length among samples in this specific batch
    int64_t batch_max_len = 0;
    for (const auto& seq : token_batch) {
        batch_max_len = std::max(batch_max_len, static_cast<int64_t>(seq.size()));
    }

    // Dynamic padding: pad only to the longest sequence in the batch (capped at max_length)
    int64_t target_len = batch_max_len;
    if (max_length > 0) {
        target_len = std::min(batch_max_len, max_length);
    }
    if (target_len == 0) {
        target_len = 1;
    }

    // Allocate contiguous CPU tensors
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor input_ids = torch::full({batch_size, target_len}, pad_token_id, options);
    torch::Tensor attention_mask = torch::zeros({batch_size, target_len}, options);

    auto input_ids_acc = input_ids.accessor<int64_t, 2>();
    auto mask_acc = attention_mask.accessor<int64_t, 2>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& seq = token_batch[b];
        int64_t cur_len = static_cast<int64_t>(seq.size());

        // Truncate to target_len if longer
        int64_t copy_len = std::min(cur_len, target_len);
        int64_t pad_offset = target_len - copy_len; // Left padding offset

        // Copy tokens to the right side of the buffer
        for (int64_t i = 0; i < copy_len; ++i) {
            int64_t token = seq[cur_len - copy_len + i];
            input_ids_acc[b][pad_offset + i] = token;
            mask_acc[b][pad_offset + i] = 1;
        }
    }

    return {input_ids, attention_mask};
}

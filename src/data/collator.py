"""
Custom dynamic left-padding collator for DistilGPT2 fine-tuning.
Provides seamless acceleration via C++ (csrc/fast_collator.cpp) with an optimized PyTorch fallback.
"""

from typing import List, Dict, Any
import torch
from src.cpp_extension import get_fast_collator
from utils.logger import get_logger

logger = get_logger(__name__)


class GPTDataCollator:
    """
    Data collator that dynamically left-pads batches of tokenized sequences.
    Ensures that the last token in every sample is aligned with the actual sequence end.
    """

    def __init__(self, pad_token_id: int, max_length: int = 128, use_cpp: bool = True):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.use_cpp = use_cpp
        self.fast_collator_fn = get_fast_collator() if use_cpp else None

        if self.fast_collator_fn is not None:
            logger.info("GPTDataCollator: Using C++ fast_left_pad_collate acceleration.")
        else:
            logger.info("GPTDataCollator: Using vectorized PyTorch left-padding fallback.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of sample dicts into a padded batch dictionary.

        Args:
            batch: List of dictionaries with 'input_ids', optionally 'attention_mask' and 'labels'.

        Returns:
            Dictionary containing 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        # Extract labels if present
        labels = None
        if "labels" in batch[0]:
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        elif "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        # C++ Fast Path
        if self.fast_collator_fn is not None:
            token_batch = [
                item["input_ids"].tolist() if isinstance(item["input_ids"], torch.Tensor)
                else item["input_ids"]
                for item in batch
            ]
            input_ids, attention_mask = self.fast_collator_fn(
                token_batch, self.pad_token_id, self.max_length
            )
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if labels is not None:
                result["labels"] = labels
            return result

        # PyTorch Vectorized Fallback Path
        input_ids_list = [item["input_ids"] for item in batch]
        batch_size = len(input_ids_list)

        # Determine dynamic length: longest sequence in this batch, capped at max_length
        actual_max = max(len(seq) for seq in input_ids_list)
        seq_len = min(actual_max, self.max_length) if self.max_length > 0 else actual_max
        seq_len = max(seq_len, 1)

        padded_input_ids = torch.full((batch_size, seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i, seq in enumerate(input_ids_list):
            if isinstance(seq, torch.Tensor):
                cur_len = seq.size(0)
                copy_len = min(cur_len, seq_len)
                pad_offset = seq_len - copy_len
                # Left-padding: copy to the tail end of the row
                padded_input_ids[i, pad_offset:] = seq[cur_len - copy_len:]
                attention_mask[i, pad_offset:] = 1
            else:
                cur_len = len(seq)
                copy_len = min(cur_len, seq_len)
                pad_offset = seq_len - copy_len
                padded_input_ids[i, pad_offset:] = torch.tensor(
                    seq[cur_len - copy_len:], dtype=torch.long
                )
                attention_mask[i, pad_offset:] = 1

        result = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            result["labels"] = labels

        return result

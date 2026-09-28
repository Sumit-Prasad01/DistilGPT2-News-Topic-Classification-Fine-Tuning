"""
GPT Tokenizer configuration and dataset tokenization utilities.
Enforces the 2 critical GPT fine-tuning fixes: left-padding and EOS pad token assignment.
"""

from typing import Any
from transformers import AutoTokenizer
from datasets import DatasetDict
from utils.custom_exception import TokenizerException
from utils.logger import get_logger

logger = get_logger(__name__)


def get_gpt_tokenizer(model_name: str = "distilgpt2", max_length: int = 128) -> Any:
    """
    Load and configure the GPT tokenizer with necessary causal model adjustments.

    Args:
        model_name: Hugging Face model identifier (e.g. 'distilgpt2' or 'gpt2').
        max_length: Maximum sequence length.

    Returns:
        Configured AutoTokenizer instance.
    """
    try:
        logger.info(f"Loading and configuring tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT FIX 1: Reuse EOS token as PAD token (GPT has no pad token by default)
        tokenizer.pad_token = tokenizer.eos_token

        # GPT FIX 2: Set padding side to 'left' so the final token is always the real text token
        tokenizer.padding_side = "left"

        tokenizer.model_max_length = max_length

        logger.info(
            f"Tokenizer configured: pad_token='{tokenizer.pad_token}' (id={tokenizer.pad_token_id}), "
            f"padding_side='{tokenizer.padding_side}', max_length={tokenizer.model_max_length}"
        )
        return tokenizer
    except Exception as e:
        raise TokenizerException(f"Failed to configure tokenizer for {model_name}: {e}")


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    max_length: int = 128,
    dynamic_padding: bool = False
) -> DatasetDict:
    """
    Tokenize all splits of the dataset.

    Args:
        dataset_dict: DatasetDict containing raw text and labels.
        tokenizer: Configured GPT tokenizer.
        max_length: Maximum token length.
        dynamic_padding: If True, skips static max_length padding (deferred to data collator).

    Returns:
        Tokenized DatasetDict with 'input_ids', 'attention_mask', and 'labels'.
    """
    try:
        def tokenize_batch(examples):
            # When dynamic padding is used, we only truncate to max_length and let collator pad
            pad_mode = False if dynamic_padding else "max_length"
            return tokenizer(
                examples["text"],
                padding=pad_mode,
                truncation=True,
                max_length=max_length,
            )

        logger.info(f"Tokenizing dataset (dynamic_padding={dynamic_padding})...")
        tokenized_dict = dataset_dict.map(
            tokenize_batch,
            batched=True,
            desc="Tokenizing splits",
        )

        # Drop original text column to save memory
        if "text" in tokenized_dict["train"].column_names:
            tokenized_dict = tokenized_dict.remove_columns(["text"])

        # Rename label -> labels for Hugging Face Trainer compatibility
        if "label" in tokenized_dict["train"].column_names:
            tokenized_dict = tokenized_dict.rename_column("label", "labels")

        tokenized_dict.set_format("torch")
        logger.info(f"Tokenization complete. Available features: {list(tokenized_dict['train'].features.keys())}")
        return tokenized_dict
    except Exception as e:
        raise TokenizerException(f"Failed to tokenize dataset: {e}")

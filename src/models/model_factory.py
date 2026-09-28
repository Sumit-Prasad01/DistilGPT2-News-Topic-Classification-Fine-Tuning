"""
Model factory for DistilGPT2 Sequence Classification.
Configures classification head, maps labels, and aligns pad token IDs.
"""

from typing import Any, Optional
import torch
from transformers import AutoModelForSequenceClassification
from utils.custom_exception import ModelConfigurationException
from utils.logger import get_logger

logger = get_logger(__name__)


def build_distilgpt2_classifier(
    model_config: Any,
    tokenizer: Any,
    device: Optional[Any] = None
) -> Any:
    """
    Instantiate and configure DistilGPT2 for sequence classification.

    Args:
        model_config: Model configuration containing model_name, num_labels, id2label, label2id.
        tokenizer: Configured GPT tokenizer with pad_token_id.
        device: Target compute device (CUDA / MPS / CPU).

    Returns:
        Configured AutoModelForSequenceClassification model instance.
    """
    try:
        model_name = getattr(model_config, "model_name", "distilgpt2")
        num_labels = getattr(model_config, "num_labels", 4)
        id2label = getattr(model_config, "id2label", None)
        label2id = getattr(model_config, "label2id", None)

        logger.info(f"Loading sequence classification model: {model_name} (num_labels={num_labels})")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # GPT FIX 3: Tell the model which token ID is the pad token.
        # GPT has no built-in pad_token_id, so we set it here to match the tokenizer.
        # Without this, GPT cannot correctly mask out padding during the forward pass.
        model.config.pad_token_id = tokenizer.pad_token_id

        if device is not None:
            model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            f"Model successfully loaded on {device or 'default device'}. "
            f"Total params: {total_params:,} | Trainable params: {trainable_params:,}"
        )
        return model
    except Exception as e:
        raise ModelConfigurationException(f"Failed to build sequence classification model: {e}")

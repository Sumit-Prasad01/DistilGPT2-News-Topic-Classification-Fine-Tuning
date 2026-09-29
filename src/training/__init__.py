"""
Training orchestration and callback integration package.
"""

from src.training.callbacks import MLflowLoggingCallback
from src.training.trainer import NewsClassifierTrainer

__all__ = ["MLflowLoggingCallback", "NewsClassifierTrainer"]

"""
Custom exception hierarchy for DistilGPT2 News Topic Classification.
Provides structured, descriptive errors across data, modeling, C++ extensions, and training.
"""

class NewsClassifierException(Exception):
    """Base exception for all news topic classification errors."""
    def __init__(self, message: str = "", details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataProcessingException(NewsClassifierException):
    """Raised when data loading, balanced sampling, or dataset splitting fails."""
    pass


class TokenizerException(NewsClassifierException):
    """Raised when tokenization, pad token alignment, or truncation fails."""
    pass


class ModelConfigurationException(NewsClassifierException):
    """Raised when model instantiation, weight loading, or pad_token_id configuration fails."""
    pass


class CppExtensionException(NewsClassifierException):
    """Raised when C++ extensions fail to build, load, or execute."""
    pass


class TrainingException(NewsClassifierException):
    """Raised when model training, optimization, or checkpointing encounters an error."""
    pass


class EvaluationException(NewsClassifierException):
    """Raised when computing evaluation metrics, confusion matrix, or reports fails."""
    pass


class InferenceException(NewsClassifierException):
    """Raised during prediction, batch inference, or postprocessing."""
    pass

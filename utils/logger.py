"""
Production logging utility for DistilGPT2 News Topic Classification.
Provides formatted console logging and optional file logging with thread-safe configuration.
"""

import logging
import os
import sys
from typing import Optional

# ANSI Color Codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds ANSI colors to console logs."""

    LEVEL_COLORS = {
        logging.DEBUG: GRAY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD + RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{record.levelname:<8}{RESET}"
        record.name = f"{CYAN}{record.name}{RESET}"
        return super().format(record)


def get_logger(name: str = "NewsClassifier", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger.

    Args:
        name: Name of the logger (typically __name__).
        log_file: Optional path to file where logs should be appended.
        level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger instance configured with console and optional file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple duplicate handlers
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = "%(asctime)s | %(levelname)s | %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Use color formatter if stdout is an interactive terminal or supports ANSI
    console_formatter = ColoredFormatter(console_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

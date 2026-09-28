"""
Configuration loader for DistilGPT2 News Topic Classification.
Loads YAML configuration files into structured, type-safe objects with property access.
"""

import os
from typing import Any, Dict, List, Optional
from utils.custom_exception import NewsClassifierException
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


class ConfigDict(dict):
    """Dictionary subclass enabling attribute-style dot access (e.g. config.model.name)."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")


def _parse_yaml_fallback(filepath: str) -> Dict[str, Any]:
    """Basic fallback parser in case PyYAML is not yet installed in the current environment."""
    result = {}
    current_section = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_section = line[:-1].strip()
                result[current_section] = {}
            elif current_section and line.startswith("  "):
                sub_line = line.strip()
                if ":" in sub_line:
                    k, v = sub_line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Basic type conversion
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    elif v.lower() in ("null", "none", "~", ""):
                        v = None
                    elif v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    elif v.startswith("'") and v.endswith("'"):
                        v = v[1:-1]
                    else:
                        try:
                            if "." in v or "e" in v.lower():
                                v = float(v)
                            else:
                                v = int(v)
                        except ValueError:
                            pass
                    result[current_section][k] = v
                elif sub_line.startswith("- "):
                    item = sub_line[2:].strip().strip("\"'")
                    if not isinstance(result[current_section], list):
                        # Convert to list if it was initialized as dict
                        result[current_section] = []
                    result[current_section].append(item)
    return result


def load_config(config_path: str = "config/config.yaml") -> ConfigDict:
    """
    Load a YAML configuration file.

    Args:
        config_path: Relative or absolute path to the YAML file.

    Returns:
        ConfigDict object supporting both dot and bracket access.
    """
    if not os.path.exists(config_path):
        raise NewsClassifierException(f"Configuration file not found at: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    if yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
    else:
        logger.warning("PyYAML not found in environment; using built-in YAML parser fallback.")
        raw_cfg = _parse_yaml_fallback(config_path)

    # Add helper properties for model labels
    if "model" in raw_cfg:
        labels = raw_cfg["model"].get("label_names", ["World", "Sports", "Business", "Sci/Tech"])
        raw_cfg["model"]["num_labels"] = len(labels)
        raw_cfg["model"]["id2label"] = {i: name for i, name in enumerate(labels)}
        raw_cfg["model"]["label2id"] = {name: i for i, name in enumerate(labels)}

    return ConfigDict(raw_cfg)

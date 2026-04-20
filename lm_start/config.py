"""Configuration management for lm-start."""

import yaml
from pathlib import Path

from lm_start.constants import LM_START_DIR, LM_START_CONFIG_DIR


def ensure_config_dir():
    """Create ~/.lm-start/ and config/ subdirectory if they don't exist."""
    LM_START_DIR.mkdir(parents=True, exist_ok=True)
    LM_START_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    """Load YAML file, return empty dict if file doesn't exist."""
    if not Path(path).exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data):
    """Save dict to YAML file."""
    ensure_config_dir()
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

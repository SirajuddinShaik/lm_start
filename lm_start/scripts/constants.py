"""Configuration constants for lm-start."""

from pathlib import Path

# Base directory for all lm-start configuration
LM_START_DIR = Path.home() / ".lm-start"

# Subdirectory for config files
LM_START_CONFIG_DIR = LM_START_DIR / "config"

# Config file paths
CREDENTIALS_FILE = LM_START_CONFIG_DIR / "credentials.yaml"
SYSTEM_FILE = LM_START_CONFIG_DIR / "system.yaml"

# -*- coding: utf-8 -*-
"""
Project configuration for local execution.
Provides path resolution relative to project root.
"""

from pathlib import Path

# Project root directory (where this file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directory (created on first run)
DATA_DIR = PROJECT_ROOT / "data"

# Mask directories
ORIGINAL_MASKS_DIR = PROJECT_ROOT / "Original_Masks"
CUSTOM_MASKS_DIR = PROJECT_ROOT / "Custom_Masks"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


def ensure_dirs():
    """Create required directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)

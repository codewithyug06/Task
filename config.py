"""
Global Configuration & Dynamic Path Resolution
Automatically resolves repository paths across local machines, Google Colab, Kaggle, and Linux/macOS.
"""

import os
from pathlib import Path

# Repo Root: directory containing this config.py
REPO_ROOT = Path(__file__).resolve().parent

# Data Directories
DATA_FIXED_DIR = REPO_ROOT / "data_fixed"
DATA_DIR = DATA_FIXED_DIR if DATA_FIXED_DIR.exists() else (REPO_ROOT / "data")

# Module and Output Directories
TASK1_DIR = REPO_ROOT / "task1_finetune"
TASK2_DIR = REPO_ROOT / "task2_rag"
TASK3_DIR = REPO_ROOT / "task3_optimization"
LORA_ADAPTER_DIR = TASK1_DIR / "lora_adapter"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Default Model Hub Identifiers
DEFAULT_LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

def get_model_path(env_var="MODEL_PATH", default=DEFAULT_LLM_ID):
    """
    Returns the model path or Hugging Face Hub ID.
    Priority:
    1. Explicit environment variable (e.g. MODEL_PATH)
    2. Default Hugging Face Hub repository identifier
    """
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]
    return default

def get_embedding_model_path(env_var="EMBEDDING_MODEL_PATH", default=DEFAULT_EMBED_ID):
    """
    Returns the embedding model path or Hugging Face Hub ID.
    Priority:
    1. Explicit environment variable (e.g. EMBEDDING_MODEL_PATH)
    2. Default Hugging Face Hub repository identifier
    """
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]
    return default

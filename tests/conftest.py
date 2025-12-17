"""
Pytest configuration and fixtures.
This file is automatically loaded by pytest before running tests.
"""

import os
import sys

# Force PyTorch to use CPU only (disable CUDA)
# This must be set BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add src to path if needed
import pathlib

project_root = pathlib.Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

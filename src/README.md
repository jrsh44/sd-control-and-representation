# Source Code 

This directory contains the core library code for the project.

## Structure

### `data/`
Data handling and caching utilities:

- **`cache.py`** - Fast numpy memmap-based cache for layer representations
- **`dataset.py`** - PyTorch Dataset implementations for cached representations
- **`prompts.py`** - Prompt loading utilities shared across models

### `models/`
Model implementations and configurations:

- **`config.py`** - Model registry and configuration definitions

#### `models/sae/`
Sparse Autoencoder implementation:

- **`feature_selection.py`** - Feature selection and concept filtering functions
- **`sd_integration.py`** - Integration utilities for SAE with Stable Diffusion

##### `models/sae/training/`
SAE training infrastructure:

- **`config.py`** - Training configuration data classes
- **`losses.py`** - Loss functions for SAE training
- **`metrics.py`** - Metrics computation (sparsity, reconstruction, etc.)
- **`trainer.py`** - Main SAE trainer class with callbacks and logging
- **`utils.py`** - Training helper functions

#### `models/sd_v1_5/`
Stable Diffusion v1.5 specific implementations:

- **`hooks.py`** - Hook registration and activation capture during generation
- **`layers.py`** - Layer path definitions for all SD v1.5 modules

### `utils/`
General utility functions:

- **`fid.py`** - Fréchet Inception Distance calculation utilities
- **`model_loader.py`** - Model downloading and loading from HuggingFace/Google Drive
- **`NudeNet_detector.py`** - NudeNet detection wrapper
- **`nudenet.py`** - NudeNet wrapper for image scoring
- **`RepresentationModifier.py`** - SAE-based representation modification for concept intervention
- **`script_functions.py`** - Helper functions for scripts
- **`visualization.py`** - Image display and visualization utilities
- **`wandb.py`** - Weights & Biases logging utilities

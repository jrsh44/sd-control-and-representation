# Notebooks

This directory contains Jupyter notebooks for demonstrations, analysis, and experiments.

## Files

### `analysis.ipynb`
Comprehensive analysis of SAE evaluation metrics. Loads and visualizes results from CLIP, FID, LPIPS, and NudeNet scores across different configurations (k16, k32, k64). Includes detailed plots and comparisons of model performance.

### `feature_selection_scores_analysis.ipynb`
Analysis of feature selection scores for SAE training. Visualizes feature importance, distribution patterns, and interactive plots to understand which features contribute most to concept representation.

### `sd_1_5_usage_example.ipynb`
Basic usage example of Stable Diffusion 1.5 model. Demonstrates how to load the model, generate images from text prompts, and perform basic inference operations.

### `sd_1_5_caching_reprezentation_example.ipynb`
Example of capturing and caching layer representations during Stable Diffusion generation. Shows how to hook into model layers and extract intermediate activations for analysis.
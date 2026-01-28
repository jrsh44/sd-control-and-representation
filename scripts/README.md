# Scripts

This directory contains scripts for training, evaluation, and data processing.

Each script includes detailed usage examples and documentation in its header. Open any `.py` file to see specific command-line arguments and usage patterns.

## Structure

### Root Files

- **`download_sae_models.py`** - Download SAE models from HuggingFace repository to local directory

### `data/`
Data preparation and export utilities:

- **`download_wandb_images.py`** - Download images from W&B runs for analysis
- **`export_wandb_runs.py`** - Export W&B run data to CSV format
- **`extract_cc3m_prompts.py`** - Extract text prompts from CC3M dataset
- **`merge_feature_sums.py`** - Merge SAE feature sum files from multiple runs

### `image_evaluation/`
Scripts for calculating image quality and similarity metrics:

- **`calculate_clip_image_image_scores.py`** - Compute CLIP similarity between image pairs (original vs intervention)
- **`calculate_clip_prompt_image_scores.py`** - Compute CLIP similarity between prompts and generated images
- **`calculate_fid.py`** - Calculate Fréchet Inception Distance scores
- **`calculate_lpips_scores.py`** - Calculate LPIPS perceptual similarity scores
- **`calculate_nudnet_scores.py`** - Run NudeNet detection on generated images
- **`calculate_means_cov.py`** - Compute mean and covariance statistics for FID calculation
- **`calculate_means_cov_coco.py`** - Compute FID statistics for COCO-2017 reference dataset
- **`nudity_unlearning_evaluation.py`** - Comprehensive evaluation pipeline for nudity concept unlearning
- **`*.sh`** - Shell scripts for batch evaluation jobs

### `sae/`
Sparse Autoencoder training and analysis:

- **`train.py`** - Train SAE on cached SD layer representations
- **`feature_selection.py`** - Select and analyze important SAE features for specific concepts
- **`generate_feature_heatmaps.py`** - Generate activation heatmaps for SAE features
- **`feature_selection_cc3m-wds.sh`** - Feature selection on CC3M dataset
- **`feature_selection_nudity.sh`** - Feature selection on nudity dataset
- **`train.sh`** - Training job script
- **`generate_feature_heatmaps.sh`** - Batch heatmap generation

### `sd_v1_5/`
Stable Diffusion layer representation caching and intervention:

- **`cache_rep_from_file.py`** - Cache layer representations from prompt file (supports SLURM array jobs)
- **`cache_rep_from_file_and_class.py`** - Cache representations with class labels
- **`cache_rep_from_objects_dir_and_style_.py`** - Cache representations for object-style combinations
- **`generate_unlearned_cache_from_file.py`** - Generate images using SAE concept intervention
- **`generate_unlearned_cache_batch.py`** - Batch generation with intervention
- **`generate_unlearned_image.py`** - Generate single images with concept unlearning
- **`merge_cache_metadata.py`** - Merge metadata from multiple cache runs
- **`*.sh`** - Shell scripts for different configurations (s16/s32/s64 = SAE sparsity levels, pt = per-timestep mode)

### `tests/`
Testing and validation scripts:

- **`image_generation.py`** - Test SD image generation on cluster
- **`generate_unlearned_image_grid.py`** - Generate grid of images with different intervention parameters
- **`is_cuda_available.py`** - Check CUDA availability and GPU info
- **`nudenet_test.py`** - Test NudeNet detection functionality
- **`nudity_detection.py`** - Nudity detection utilities
- **`test_nudenet_wrapper.py`** - Test custom NudeNet wrapper

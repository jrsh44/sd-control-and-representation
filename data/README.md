# Data Directory

This directory contains datasets and analysis results for the project.

## Structure

### `analysis/`
Evaluation metrics for different SAE configurations (k16, k32, k64):
- `clip_image_scores.csv` - CLIP image-to-image similarity scores
- `clip_prompt_scores.csv` - CLIP prompt-to-image similarity scores  
- `fid_scores.csv` - Fréchet Inception Distance scores
- `lpips_scores.csv` - Learned Perceptual Image Patch Similarity scores
- `nudenet_scores.csv` - NudeNet detection scores
- `per_timestep/` - Same metrics computed per diffusion timestep

### `cc3m-wds/`
Prompts extracted from CC3M (Conceptual Captions 3M) dataset:
- `prompts_fs.txt` - Prompts for feature selection
- `prompts_sae.txt` - Prompts for SAE training

### `nudity/`
Dataset for nudity concept unlearning evaluation:
- `classes.txt` - Object classes (20 classes)
- `prompts_*.txt` - Text prompts for different evaluation tasks

### `unlearn_canvas/`
UnlearnCanvas dataset for concept removal evaluation:
- `class.txt` - 20 object classes (e.g., "Fishes", "Flame", "Statues")
- `styles.txt` - 61 artistic styles (e.g., "Rust", "Seed_Images")
- `prompts/` - 80 prompts per class for image generation (e.g., "A butterfly emerging from a jeweled cocoon")

Used for feature selection in SAE and evaluating concept unlearning effectiveness.

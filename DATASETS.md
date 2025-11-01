# Datasets used in project

## 1. UnlearnCanvas

### Dataset includes:
- 20 classes (objects) (e.g. "Fishes", "Flame", "Statues", "Trees" )
- For each object: 80 prompts. (e.g. "A butterfly emerging from a jeweled cocoon", "Horses pulling an ancient war chariot." )
- 61 styles (e.g., "Rust", "Seed_Images")

### Format of files
All files are in txt format. Each line in the file is a single observation.

### Brief description of the content/task
A collection of text prompts for image generation in Stable Diffusion.

### Task
Evaluation of concept removal (concept unlearning).

### Used for
1. Feature selection in SAE (difference between "true" and "false" activations)
2. Evaluation of unlearning effectiveness (whether the "cartoon" concept disappears, but the "photo" concept remains)

## 📁 Dataset Dir Structure

```
data/
├── UnlearnCanvas_resources/
│   ├── anchor_prompts/
│   │   ├── finetune_prompts/    
│   │   |  ├── sd_prompt_Architectures.txt     # file contains 80 prompts for a specific class
|   |   |  └── ...
│   ├── class.txt # file contains 20 classes
│   └── styles.txt # file contains 61 styles
```

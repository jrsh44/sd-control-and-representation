# SAE Concept Intervention Dashboard

An interactive web application for visualizing and testing concept unlearning in Stable Diffusion v1.5 using Sparse Autoencoders.

## 🎬 Demo

[![Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1RSldFm64GFiDlkjNmmz-jksBUxa--YIV/view?usp=sharing)


## 📋 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | - | NVIDIA GPU (8GB+ VRAM) |
| CUDA | - | 11.8+ |
| RAM | 12GB | 16GB |
| Disk Space | 20GB SSD | 50GB SSD |
| CPU | 4 cores | 8 cores |



## 🚀 Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Download SAE models

The dashboard requires pre-trained SAE models. Download them from HuggingFace:

```bash
# Download all available models
uv run scripts/download_sae_models.py --all

# Or list available models first
uv run scripts/download_sae_models.py --list
```

Models will be downloaded to the `models/` directory.

### 3. Run the dashboard

```bash
uv run python dashboard/app.py
```

### 4. Open in browser

Navigate to `http://127.0.0.1:7860`


## 📖 Workflow

The dashboard is divided into three main panels:

### Panel 1: Base Model

1. Select device: GPU or CPU
2. Click **Load Model** to initialize base model - Stable Diffusion v1.5
3. Wait for the panel border to turn green

| Border Color | Status |
|--------------|--------|
| 🟡 Yellow | Not loaded |
| 🔵 Blue | Loading... |
| 🟢 Green | Ready |

### Panel 2: Concept Intervention

1. Select an SAE model from the dropdown (e.g., *Nudity SAE (Top-K 32)*)
2. Click **Load SAE** and wait for green status
3. Configure concepts to intervene on:
   - **Checkbox** — Enable/disable intervention for concept
   - **Strength** — Intensity of concept suppression (0–32)
   - **Neurons** — Number of SAE neurons to modify (1–32)

### Panel 3: Image Generation

1. Enter a **prompt** describing the image
2. Configure generation settings:
   - **Guidance Scale** (1–20)
   - **Seed**
   - **Neuron Selection Mode**: *Per-Timestep* or *Global*
3. Click **Generate Image** to create both original and intervention images

#### Evaluation Metrics

After generation, the dashboard displays:

| Metric | Description |
|--------|-------------|
| **NudeNet Detection** | Per-class confidence scores  |
| **CLIP Score** | Semantic similarity between: Original↔Intervention, Prompt→Original, Prompt→Intervention |

#### Heatmaps

Visualize SAE feature activations:
1. Enter **timesteps** to analyze (e.g., `10, 25, 40`)
2. Select a **concept** from the active interventions
3. Click **Generate Heatmaps**

---

## ⚙️ Configuration

SAE models and concepts are defined in `dashboard/config/sae_config.yaml`.

### Available SAE Models

| Model | Top-K |
|-------|-------|
| Nudity SAE (Top-K 16) | 16 | 
| Nudity SAE (Top-K 32) | 32 | 
| Nudity SAE (Top-K 64) | 64 | 

### Adding a New SAE Model

```yaml
sae_models:
  - id: "my_new_model"
    name: "My SAE Model"
    
    hyperparameters:
      topk: 32              
      nb_concepts: 46080    
      learning_rate: 1.0e-3
      warmup_steps: 100000
      auxiliary_loss: 0.0625
      epochs: 2
      batch_size: 4096
    
    files:
      model_dir: "path/to/model/dir"
    
    layer:
      id: "UNET_UP_1_ATT_1"
    
    concepts:
      - id: "concept_key"        # Must match key in feature_sums file
        name: "Display Name"
        description: "Description shown in UI"
```

---
## 🗂️ Project Structure

```
dashboard/
├── app.py                    # Main entry point
├── style.css                 # Custom styling
├── config/
│   ├── sae_config.yaml       # SAE models & concepts configuration
│   └── sae_config_loader.py  # YAML parser
├── core/
│   ├── model_loader.py       # SD, SAE, NudeNet loading
│   └── state.py              # Application state management
└── utils/
    ├── clip_score.py         # CLIP similarity metrics
    ├── detection.py          # NudeNet detection formatting
    ├── heatmap.py            # SAE activation visualization
    └── cuda.py               # GPU detection & configuration
```
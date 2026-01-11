# SD Control - Dashboard

Interactive dashboard for concept unlearning in Stable Diffusion v1.5 using Sparse Autoencoders.**


## ✨ Key Features

- **Side-by-Side Comparison** — Generate original and unlearned images with the same seed for direct comparison
- **Multi-Concept Intervention** — Select and suppress multiple concepts (e.g., nudity classes) simultaneously
- **NudeNet Detection** — Automatic content detection with visual metrics (enabled by default)
- **Configuration-Driven** — YAML-based SAE model and layer configuration

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web UI | Gradio 4.x |
| ML Framework | PyTorch 2.x |
| Diffusion | Diffusers (SD v1.5) |
| SAE | overcomplete (TopKSAE) |
| Detection | NudeNet |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **CUDA GPU** with 6GB+ VRAM (recommended) or CPU fallback
- **uv** package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/sd-control-and-representation.git
cd sd-control-and-representation

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

The dashboard uses a YAML configuration file for SAE models:

```
dashboard/config/sae_config.yaml
```

Default settings work out-of-the-box. For custom SAE models, edit the config file to point to your model paths.

### Run

```bash
# Activate environment (if using venv)
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Launch the dashboard
python dashboard/app.py
```

Dashboard opens at: **http://127.0.0.1:7860**

---

## 📁 Project Structure

```
dashboard/
├── app.py                 # Main application entry point
├── style.css              # Neural Lab Terminal theme
├── config/
│   └── sae_config.yaml    # SAE model configurations
├── core/
│   ├── model_loader.py    # Model loading utilities
│   └── state.py           # State management
└── utils/
    ├── cuda.py            # GPU compatibility
    └── detection.py       # NudeNet integration

```

---

## 🎯 Basic Usage

1. **Load Models** — Click "🔄 LOAD MODELS" to initialize SD v1.5 and SAE
2. **Select Concepts** — Choose concepts to unlearn (e.g., `FEMALE_BREAST_EXPOSED`)
3. **Enter Prompt** — Type your generation prompt
4. **Generate** — Click "⚡ INITIATE GENERATION" to create comparison images
5. **Review** — View original vs. unlearned results side-by-side

---

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| Steps | 50 | Diffusion steps |
| Guidance | 7.5 | Classifier-free guidance scale |
| Seed | -1 | Random seed (-1 = random) |
| Layer | UNET_UP_1_ATT_1 | Target U-Net layer for intervention |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Restart dashboard, reduce steps |
| `Port 7860 in use` | Change port in `app.py` or kill existing process |
| `ModuleNotFoundError` | Run `uv sync` or `pip install -e .` |
| Models won't load | Check network connection for HuggingFace downloads |

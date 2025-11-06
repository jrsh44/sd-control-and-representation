# SD Control and Representation

Representation caching and analysis for Stable Diffusion models.

## Project Structure

```
sd-control-and-representation/
├── src/                          # Source code
│   ├── models/                   # Model-specific implementations
│   │   ├── __init__.py
│   │   ├── sd_v1_5/             # Stable Diffusion v1.5
│   │   │   ├── __init__.py
│   │   │   ├── layers.py        # Layer definitions using enum
│   │   │   └── hooks.py         # Representation capture logic
│   │   │
│   │   └── sae/                 # Sparse Autoencoder
│   │       ├── training.py      # SAE training utilities
│   │       ├── feature_selection.py  # Feature selection methods
│   │       └── sd_integration.py     # SD model integration
│   │
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   ├── cache.py             # Saving cached Reprezentation
│   │   ├── dataset.py           # Datasets loading
│   │   └── prompts.py           # Prompt loading utilities
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── wandb.py             # WandB logging & system metrics
│   │   └── visualization.py     # Visualization helpers
│   │
│   └── tests/                    # Test files
│
├── scripts/                      # Executable scripts
│   ├── sd_v1_5/                 # SD 1.5 specific scripts
│   │   ├── generate_cache.py   # Cache generation script
│   │   └── generate_cache.sh   # SLURM batch script
│   │
│   └── sae/                      # SAE training scripts (future)
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── sd_1_5_caching_reprezentation_example.ipynb
│   ├── sd_1_5_sae_usage_examples.ipynb
│   └── sd_1_5_usage_example.ipynb
│
├── data/                         # Data files
│   └── unlearn_canvas/
│       ├── class.txt
│       ├── styles.txt
│       └── prompts/
│           ├── *.txt            # Prompt files by object
│           └── test/            # Test prompts
│
├── .env                          # Environment variables
├── pyproject.toml               # Python project configuration
└── README.md
```

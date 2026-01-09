import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env")

from src.models.sd_v1_5.layers import LayerPath  # noqa: E402


def parse_layer_names(layer_names: List[str]) -> List[LayerPath]:
    layers = []
    for name in layer_names:
        try:
            layer = LayerPath[name.upper()]
            layers.append(layer)
        except KeyError:
            print(f"Warning: Unknown layer name '{name}', skipping...")
    return layers

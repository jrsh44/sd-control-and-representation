"""
SAE Configuration Loader

Loads SAE model configurations from YAML files.

Structure:
    sae_models (list) → layers (list) → concepts (list)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConceptConfig:
    """Configuration for a single concept."""

    id: str
    name: str  # Must match key in merged_feature_sums.pt

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptConfig":
        return cls(id=data["id"], name=data["name"])


@dataclass
class LayerConfig:
    """Configuration for a layer that an SAE model was trained on."""

    id: str  # e.g., "UNET_UP_1_ATT_1"
    path: str  # e.g., "unet.up_blocks.1.attentions.1.transformer_blocks.0"
    concepts: list[ConceptConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "LayerConfig":
        concepts = [ConceptConfig.from_dict(c) for c in data.get("concepts", [])]
        return cls(id=data["id"], path=data["path"], concepts=concepts)

    def get_concept(self, concept_id: str) -> ConceptConfig | None:
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        return None


@dataclass
class SAEFilesConfig:
    """File paths for an SAE model."""

    model_dir: str
    model_file: str = "model.pt"
    feature_sums_file: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "SAEFilesConfig":
        return cls(
            model_dir=data.get("model_dir", ""),
            model_file=data.get("model_file", "model.pt"),
            feature_sums_file=data.get("feature_sums_file", ""),
        )


@dataclass
class SAEModelConfig:
    """Configuration for an SAE model."""

    id: str
    name: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    files: SAEFilesConfig | None = None
    layers: list[LayerConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SAEModelConfig":
        files_data = data.get("files", {})
        files = SAEFilesConfig.from_dict(files_data) if files_data else None
        layers = [LayerConfig.from_dict(layer) for layer in data.get("layers", [])]

        return cls(
            id=data["id"],
            name=data["name"],
            hyperparameters=data.get("hyperparameters", {}),
            files=files,
            layers=layers,
        )

    def get_layer(self, layer_id: str) -> LayerConfig | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def get_layer_ids(self) -> list[str]:
        return [layer.id for layer in self.layers]


@dataclass
class SAEConfig:
    """Root configuration."""

    version: str
    base_model_id: str
    paths: dict[str, dict[str, str]] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    sae_models: list[SAEModelConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SAEConfig":
        config = data.get("config", {})
        base_model = config.get("base_model", {})

        sae_models = [SAEModelConfig.from_dict(sae_data) for sae_data in data.get("sae_models", [])]

        return cls(
            version=config.get("version", "1.0.0"),
            base_model_id=base_model.get("model_id", ""),
            paths=config.get("paths", {}),
            defaults=config.get("defaults", {}),
            sae_models=sae_models,
        )

    @classmethod
    def load(cls, config_path: str | Path) -> "SAEConfig":
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def get_sae_model(self, model_id: str) -> SAEModelConfig | None:
        for sae in self.sae_models:
            if sae.id == model_id:
                return sae
        return None

    def get_layer(self, sae_model_id: str, layer_id: str) -> LayerConfig | None:
        sae = self.get_sae_model(sae_model_id)
        if sae:
            return sae.get_layer(layer_id)
        return None

    def get_concept(
        self, sae_model_id: str, layer_id: str, concept_id: str
    ) -> ConceptConfig | None:
        layer = self.get_layer(sae_model_id, layer_id)
        if layer:
            return layer.get_concept(concept_id)
        return None

    def get_default_influence_factor(self) -> float:
        return self.defaults.get("influence_factor", 100.0)

    def get_default_features_number(self) -> int:
        return self.defaults.get("features_number", 25)


# =============================================================================
# Public API
# =============================================================================


def load_sae_config(config_path: str | Path | None = None) -> SAEConfig:
    """Load SAE configuration from the default or specified path."""
    if config_path is None:
        config_path = Path(__file__).parent / "sae_config.yaml"
    return SAEConfig.load(config_path)


def get_sae_model_choices(config: SAEConfig) -> list[tuple[str, str]]:
    """Get SAE model choices for Gradio dropdown. Returns (name, id) tuples."""
    return [(sae.name, sae.id) for sae in config.sae_models]


def get_layer_choices(config: SAEConfig, sae_model_id: str) -> list[tuple[str, str]]:
    """Get layer choices for Gradio dropdown. Returns (id, id) tuples."""
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model:
        return []
    return [(layer.id, layer.id) for layer in sae_model.layers]


def get_concept_choices(
    config: SAEConfig, sae_model_id: str, layer_id: str
) -> list[tuple[str, str]]:
    """Get concept choices for Gradio checkboxes. Returns (name, id) tuples."""
    layer = config.get_layer(sae_model_id, layer_id)
    if not layer:
        return []
    return [(concept.name, concept.id) for concept in layer.concepts]


def get_feature_sums_path(config: SAEConfig, sae_model_id: str) -> Path | None:
    """
    Get the full path to the feature sums file for an SAE model.

    Args:
        config: SAE configuration
        sae_model_id: ID of the SAE model

    Returns:
        Full Path to the feature sums file, or None if not configured
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model or not sae_model.files:
        return None

    feature_sums_file = sae_model.files.feature_sums_file
    if not feature_sums_file:
        return None

    # Get base path from config
    local_paths = config.paths.get("local", {})
    feature_sums_dir = local_paths.get("feature_sums_dir", "data/feature_sums")

    # Build full path relative to project root
    project_root = Path(__file__).parent.parent.parent  # dashboard/config -> project root
    full_path = project_root / feature_sums_dir / feature_sums_file

    return full_path


def get_sae_hyperparameters(config: SAEConfig, sae_model_id: str) -> dict[str, Any] | None:
    """
    Get hyperparameters for an SAE model.

    Args:
        config: SAE configuration
        sae_model_id: ID of the SAE model

    Returns:
        Dict with topk, nb_concepts, etc. or None if not found
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model:
        return None
    return sae_model.hyperparameters


def get_model_path(config: SAEConfig, sae_model_id: str) -> Path | None:
    """
    Get the full path to the SAE model weights file.

    Args:
        config: SAE configuration
        sae_model_id: ID of the SAE model

    Returns:
        Full Path to the model weights file, or None if not configured
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model or not sae_model.files:
        return None

    model_dir = sae_model.files.model_dir
    model_file = sae_model.files.model_file
    if not model_dir or not model_file:
        return None

    # Get base path from config
    local_paths = config.paths.get("local", {})
    models_dir = local_paths.get("models_dir", "./models/sae")

    # Build full path relative to project root
    project_root = Path(__file__).parent.parent.parent  # dashboard/config -> project root
    full_path = project_root / models_dir / model_dir / model_file

    return full_path


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    config = load_sae_config()
    print(f"Config version: {config.version}")
    print(f"SAE models: {[s.id for s in config.sae_models]}")
    print(
        f"Defaults: influence={config.get_default_influence_factor()}, features={config.get_default_features_number()}"
    )

    for sae in config.sae_models:
        print(f"\n{sae.name}:")
        print(f"  Layers: {sae.get_layer_ids()}")
        for layer in sae.layers:
            print(f"  {layer.id}: {len(layer.concepts)} concepts")

        # Test feature sums path
        feature_path = get_feature_sums_path(config, sae.id)
        if feature_path:
            exists = "✓ EXISTS" if feature_path.exists() else "✗ NOT FOUND"
            print(f"  Feature sums: {feature_path} [{exists}]")

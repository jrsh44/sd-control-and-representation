"""SAE model configuration loader from YAML files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ConceptConfig:
    """Configuration for a single concept."""

    id: str
    name: str  # Must match key in merged_feature_sums.pt
    description: str = ""  # Short description for UI tooltip

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptConfig":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
        )


@dataclass
class SAEModelConfig:
    """Configuration for an SAE model."""

    id: str
    name: str
    layer_id: str = ""
    model_dir: str = ""
    model_file: str = "model.pt"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    concepts: list[ConceptConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SAEModelConfig":
        concepts = [ConceptConfig.from_dict(c) for c in data.get("concepts", [])]

        return cls(
            id=data["id"],
            name=data["name"],
            layer_id=data.get("layer_id", ""),
            model_dir=data.get("model_dir", ""),
            model_file=data.get("model_file", "model.pt"),
            hyperparameters=data.get("hyperparameters", {}),
            concepts=concepts,
        )

    def get_concept(self, concept_id: str) -> ConceptConfig | None:
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        return None


@dataclass
class SAEConfig:
    """Root configuration."""

    base_model_id: str
    sae_models: list[SAEModelConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SAEConfig":
        config = data.get("config", {})
        base_model = config.get("base_model", {})

        sae_models = [SAEModelConfig.from_dict(sae_data) for sae_data in data.get("sae_models", [])]

        return cls(
            base_model_id=base_model.get("model_id", ""),
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

    def get_concept(self, sae_model_id: str, concept_id: str) -> ConceptConfig | None:
        sae = self.get_sae_model(sae_model_id)
        if sae:
            return sae.get_concept(concept_id)
        return None


def load_sae_config(config_path: str | Path | None = None) -> SAEConfig:
    """Load SAE configuration from the default or specified path.

    Args:
        config_path: Optional path to config file. Uses default if None.

    Returns:
        Loaded SAEConfig object.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "sae_config.yaml"
    return SAEConfig.load(config_path)


def get_sae_model_choices(config: SAEConfig) -> list[tuple[str, str]]:
    """Get SAE model choices for Gradio dropdown.

    Args:
        config: SAE configuration object.

    Returns:
        List of (name, id) tuples.
    """
    return [(sae.name, sae.id) for sae in config.sae_models]


def get_concept_choices(config: SAEConfig, sae_model_id: str) -> list[tuple[str, str, str]]:
    """Get concept choices for Gradio checkboxes.

    Args:
        config: SAE configuration object.
        sae_model_id: ID of the SAE model.

    Returns:
        List of (name, id, description) tuples.
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model:
        return []
    return [(concept.name, concept.id, concept.description) for concept in sae_model.concepts]


def get_layer_id(config: SAEConfig, sae_model_id: str) -> str:
    """Get the layer ID for an SAE model.

    Args:
        config: SAE configuration object.
        sae_model_id: ID of the SAE model.

    Returns:
        Layer ID string or empty string if not found.
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model:
        return ""
    return sae_model.layer_id


def get_feature_sums_path(config: SAEConfig, sae_model_id: str) -> Path | None:
    """Get the full path to the feature sums file for an SAE model.

    Args:
        config: SAE configuration.
        sae_model_id: ID of the SAE model.

    Returns:
        Full Path to the feature sums file, or None if not configured.
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model or not sae_model.model_dir:
        return None

    project_root = Path(__file__).parent.parent.parent  # dashboard/config -> project root
    full_path = project_root / sae_model.model_dir / "merged_feature_sums.pt"

    return full_path


def get_sae_hyperparameters(config: SAEConfig, sae_model_id: str) -> dict[str, Any] | None:
    """Get hyperparameters for an SAE model.

    Args:
        config: SAE configuration.
        sae_model_id: ID of the SAE model.

    Returns:
        Dict with topk, nb_concepts, etc. or None if not found.
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model:
        return None
    return sae_model.hyperparameters


def get_model_path(config: SAEConfig, sae_model_id: str) -> Path | None:
    """Get the full path to the SAE model weights file.

    Args:
        config: SAE configuration.
        sae_model_id: ID of the SAE model.

    Returns:
        Full Path to the model weights file, or None if not configured.
    """
    sae_model = config.get_sae_model(sae_model_id)
    if not sae_model or not sae_model.model_dir:
        return None

    project_root = Path(__file__).parent.parent.parent  # dashboard/config -> project root
    full_path = project_root / sae_model.model_dir / sae_model.model_file

    return full_path


if __name__ == "__main__":
    config = load_sae_config()
    print(f"SAE models: {[s.id for s in config.sae_models]}")

    for sae in config.sae_models:
        print(f"\n{sae.name}:")
        print(f"  Layer: {sae.layer_id}")
        print(f"  Concepts: {len(sae.concepts)}")

        feature_path = get_feature_sums_path(config, sae.id)
        if feature_path:
            exists = "✓ EXISTS" if feature_path.exists() else "✗ NOT FOUND"
            print(f"  Feature sums: {feature_path} [{exists}]")

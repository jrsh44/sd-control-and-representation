#!/usr/bin/env python3
"""
Cache management for representation storage.
"""

from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import torch


class RepresentationCache:
    def __init__(self, cache_dir: Path, use_fp16: bool = True):
        """
        Args:
            cache_dir: Base directory for cache
            use_fp16: Store as float16 (50% size reduction)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_fp16 = use_fp16

        # Accumulate metadata for all saved representations
        self._metadata_buffer: Dict[str, List[Dict]] = {}

    def get_dataset_path(self, layer_name: str) -> Path:
        """
        Get storage path for a layer's dataset.

        Args:
            layer_name: Name of the layer

        Returns:
            Path: Directory path for the layer
        """
        return self.cache_dir / f"{layer_name.lower()}"

    def check_exists(self, layer_name: str, object_name: str, style: str, prompt_nr: int) -> bool:
        """
        Check if a specific representation exists by reading metadata file.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number

        Returns:
            bool: True if representation exists
        """
        dataset_path = self.get_dataset_path(layer_name)
        metadata_path = dataset_path / "metadata.parquet"

        if not metadata_path.exists():
            return False

        try:
            metadata_table = pq.read_table(str(metadata_path))
            objects = metadata_table["object"].to_pylist()
            styles = metadata_table["style"].to_pylist()
            prompt_nrs = metadata_table["prompt_nr"].to_pylist()

            for obj, sty, pnr in zip(objects, styles, prompt_nrs, strict=True):
                if obj == object_name and sty == style and pnr == prompt_nr:
                    return True

            return False

        except Exception:
            return False

    def save_representation(
        self,
        layer_name: str,
        object_name: str,
        style: str,
        prompt_nr: int,
        prompt_text: str,
        representation: torch.Tensor,
        num_steps: int,
        guidance_scale: float,
    ):
        """
        Save a single representation to disk and accumulate metadata.

        Args:
            layer_name: Name of the layer
            object_name: Object name
            style: Style name
            prompt_nr: Prompt number
            prompt_text: Full prompt text
            representation: Tensor representation
            num_steps: Number of inference steps used
            guidance_scale: Guidance scale used
        """
        dataset_path = self.get_dataset_path(layer_name)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Convert tensor
        tensor = representation.cpu().half() if self.use_fp16 else representation.cpu()
        original_shape = list(tensor.shape)
        flat_representation = tensor.flatten().numpy()

        # Find next file number
        existing_files = sorted(dataset_path.glob("data-*.parquet"))
        file_id = len(existing_files)

        # Save tensor data
        tensor_schema = pa.schema(
            [
                ("representation", pa.large_list(pa.float16() if self.use_fp16 else pa.float32())),
                ("original_shape", pa.large_list(pa.int64())),
            ]
        )

        tensor_table = pa.table(
            {
                "representation": [flat_representation],
                "original_shape": [original_shape],
            },
            schema=tensor_schema,
        )

        file_path = dataset_path / f"data-{file_id:05d}.parquet"
        pq.write_table(tensor_table, file_path, compression="zstd", compression_level=3)

        # Accumulate metadata for later saving
        if layer_name not in self._metadata_buffer:
            self._metadata_buffer[layer_name] = []

        self._metadata_buffer[layer_name].append(
            {
                "object": object_name,
                "style": style,
                "prompt_nr": prompt_nr,
                "prompt_text": prompt_text,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
            }
        )

    def save_metadata(self):
        """
        Save all accumulated metadata to metadata.parquet files.
        Call this at the end of generation to persist metadata.
        """
        for layer_name, metadata_records in self._metadata_buffer.items():
            if not metadata_records:
                continue

            dataset_path = self.get_dataset_path(layer_name)
            metadata_path = dataset_path / "metadata.parquet"

            # Build metadata columns
            metadata_objects = [r["object"] for r in metadata_records]
            metadata_styles = [r["style"] for r in metadata_records]
            metadata_prompt_nrs = [r["prompt_nr"] for r in metadata_records]
            metadata_prompt_texts = [r["prompt_text"] for r in metadata_records]
            metadata_num_steps = [r["num_steps"] for r in metadata_records]
            metadata_guidance_scales = [r["guidance_scale"] for r in metadata_records]

            metadata_schema = pa.schema(
                [
                    ("object", pa.string()),
                    ("style", pa.string()),
                    ("prompt_nr", pa.int64()),
                    ("prompt_text", pa.string()),
                    ("num_steps", pa.int64()),
                    ("guidance_scale", pa.float32()),
                ]
            )

            metadata_table = pa.table(
                {
                    "object": metadata_objects,
                    "style": metadata_styles,
                    "prompt_nr": metadata_prompt_nrs,
                    "prompt_text": metadata_prompt_texts,
                    "num_steps": metadata_num_steps,
                    "guidance_scale": metadata_guidance_scales,
                },
                schema=metadata_schema,
            )

            # Append to existing metadata or create new
            if metadata_path.exists():
                existing_metadata = pq.read_table(str(metadata_path))
                combined_metadata = pa.concat_tables([existing_metadata, metadata_table])
                pq.write_table(combined_metadata, metadata_path, compression="zstd")
            else:
                pq.write_table(metadata_table, metadata_path, compression="zstd")

        # Clear buffer after saving
        self._metadata_buffer.clear()

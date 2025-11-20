"""
Fast memmap-based dataset for representations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class RepresentationDataset(Dataset):
    """
    Fast memmap-based dataset with filtering support.
    """

    def __init__(
        self,
        cache_dir: Path,
        layer_name: str,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        filter_fn: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
        use_local_copy: bool = True,
    ):
        """
        Args:
            cache_dir: Base cache directory
            layer_name: Layer name (e.g., 'unet_up_1_att_1')
            transform: Optional transform function
            return_metadata: If True, return (data, metadata) tuple
            filter_fn: Optional filter function: (index_entry) -> bool
                       Works on lightweight index entries (no prompt_text)
            indices: Pre-computed indices to use (overrides filter_fn)
            use_local_copy: If True and data is on network FS, copy to /tmp for fast access
        """
        self.return_metadata = return_metadata
        self.transform = transform

        layer_dir = cache_dir / layer_name

        if not layer_dir.exists():
            raise ValueError(f"Layer not found: {layer_dir}")

        data_path = layer_dir / "data.npy"

        # Check if data is on slow network filesystem
        # Can be disabled via environment variable: NPY_DISABLE_LOCAL_COPY=1
        self._local_copy_path = None
        disable_copy = os.environ.get("NPY_DISABLE_LOCAL_COPY", "0") == "1"

        if use_local_copy and not disable_copy and self._is_network_fs(data_path):
            print(
                "  ⚡ Detected network filesystem - copying to local /tmp for faster access..."
            )
            self._local_copy_path = self._copy_to_local_tmp(data_path, layer_name)
            if self._local_copy_path != data_path:  # Copy succeeded
                data_path = self._local_copy_path
                print(f"  ✓ Using local copy: {data_path}")
        elif disable_copy:
            print("  ℹ️  Local copy disabled via NPY_DISABLE_LOCAL_COPY environment variable")

        # Load info to get shape and dtype
        info_path = layer_dir / "info.json"
        if not info_path.exists():
            raise ValueError(f"Missing info.json in {layer_dir}")

        with open(info_path, "r") as f:
            info = json.load(f)

        total_samples = info["total_samples"]
        feature_dim = info["feature_dim"]
        dtype_str = info["dtype"]

        # Parse dtype string (e.g., "<class 'numpy.float16'>" -> np.float16)
        if "float16" in dtype_str:
            dtype = np.float16
        elif "float32" in dtype_str:
            dtype = np.float32
        else:
            dtype = np.float32  # fallback

        # Load as pure memmap
        self._full_data = np.memmap(
            str(data_path),
            dtype=dtype,
            mode="r",
            shape=(total_samples, feature_dim),
        )
        print(f"  ✓ Loaded memmap: {total_samples} x {feature_dim}, dtype={dtype}")

        # Load metadata from info.json if needed (for filtering or metadata)
        self._metadata = None
        if filter_fn is not None or return_metadata or indices is not None:
            # Load metadata from info.json
            self._metadata = info.get("metadata", [])
            if not self._metadata:
                raise ValueError(f"No metadata found in {info_path}")
        else:
            print("  Skipping metadata load (no filtering/metadata needed) - faster startup")


        # Apply filtering
        if indices is not None:
            # Use pre-computed indices
            self.indices = indices
            self.use_direct_indexing = False
            print(f"  Using {len(indices)} pre-filtered indices")
        elif filter_fn is not None:
            # Apply filter on metadata
            if self._metadata is None:
                raise ValueError("Metadata not loaded but filter_fn provided")
            print("  Applying filter function on metadata...")
            self.indices = [i for i, entry in enumerate(self._metadata) if filter_fn(entry)]
            self.use_direct_indexing = False
            print(f"  Filtered: {len(self.indices)}/{len(self._metadata)} samples")
        else:
            # No filtering - use direct indexing
            self.indices = None
            self.use_direct_indexing = True
            self.total_samples = len(self._full_data)
            print("  Using direct indexing (no filter)")

        print(f"Loaded NPY dataset from {data_path}")
        print(f"  Total samples: {len(self._full_data)}")
        print(f"  Active samples: {len(self)} (after filtering)")
        print(f"  Data shape: {self._full_data.shape}")
        print(f"  Dtype: {self._full_data.dtype}")
        print(f"  Size on disk: {data_path.stat().st_size / 1e9:.2f} GB")

    def __len__(self):
        if self.use_direct_indexing:
            return self.total_samples
        return len(self.indices)

    def __getitem__(self, idx):
        # Map filtered index to original index
        if self.use_direct_indexing:
            real_idx = idx
        else:
            real_idx = self.indices[idx]

        rep_fp16 = self._full_data[real_idx].copy()
        rep = torch.from_numpy(rep_fp16).float()

        if self.transform is not None:
            rep = self.transform(rep)

        if not self.return_metadata:
            return rep

        # Return with metadata (if loaded)
        if self._metadata is not None:
            return rep, self._metadata[real_idx]
        else:
            # No metadata available
            return rep, {}

    def _is_network_fs(self, path: Path) -> bool:
        """Check if path is on network filesystem (Lustre, NFS, etc.)"""
        try:
            import subprocess

            result = subprocess.run(["df", "-T", str(path)], capture_output=True, text=True)  # noqa: S603, S607
            output = result.stdout
            # Check for common network FS types
            return any(fs in output for fs in ["lustre", "nfs", "cifs", "smb", "@o2ib"])
        except Exception:
            return False

    def _copy_to_local_tmp(self, source_path: Path, layer_name: str) -> Path:
        """Copy data file to local /tmp for fast access with safety checks"""
        import time

        # Create unique tmp directory for this job
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        tmp_dir = Path(f"/tmp/npy_cache_{job_id}_{layer_name}")  # noqa: S108
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = tmp_dir / "data.npy"

        if tmp_path.exists():
            print(f"  ✓ Local copy already exists: {tmp_path}")
            return tmp_path

        # Verify /tmp has enough space
        file_size_bytes = source_path.stat().st_size
        file_size_gb = file_size_bytes / 1e9

        stat = os.statvfs("/tmp")  # noqa: S108
        available_bytes = stat.f_bavail * stat.f_frsize
        available_gb = available_bytes / 1e9

        # 20% margin for safety
        required_gb = file_size_gb * 1.2

        if available_gb < required_gb:
            print("  ⚠️  WARNING: Insufficient /tmp space!")
            print(f"     Need: {required_gb:.1f}GB, Available: {available_gb:.1f}GB")
            print("     Skipping local copy - will use network storage (slower)")
            print("     Request node with more /tmp space or use smaller dataset")
            return source_path  # Fall back to network storage

        print(
            f"  ✓ /tmp space check: {available_gb:.1f}GB available, {required_gb:.1f}GB needed"
        )

        # Copy file with progress
        print(f"  📦 Copying {file_size_gb:.2f}GB to local /tmp...")
        start = time.time()

        try:
            shutil.copy2(source_path, tmp_path)
        except OSError as e:
            print(f"  ✗ Copy failed: {e}")
            print("     Falling back to network storage")
            # Clean up partial copy
            if tmp_path.exists():
                tmp_path.unlink()
            if tmp_dir.exists() and not list(tmp_dir.iterdir()):
                tmp_dir.rmdir()
            return source_path  # Fall back to network storage

        elapsed = time.time() - start
        speed_gbps = file_size_gb / elapsed if elapsed > 0 else 0
        print(f"  ✓ Copy complete in {elapsed:.1f}s ({speed_gbps:.2f} GB/s)")
        print(f"  ✓ Using local copy: {tmp_path}")

        return tmp_path

    def __del__(self):
        """Cleanup local copy on dataset destruction"""
        if hasattr(self, "_local_copy_path") and self._local_copy_path:
            try:
                # Only clean up it's job's tmp file
                if "/tmp/npy_cache_" in str(self._local_copy_path):  # noqa: S108
                    tmp_dir = self._local_copy_path.parent
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:  # noqa: S110
                pass  # Ignore cleanup errors

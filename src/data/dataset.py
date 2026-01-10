"""
Fast memmap-based dataset for representations.
"""

import bisect
import functools
import json
import os
import random
import shutil
import signal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
        return_timestep: bool = False,  # NEW
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
            print("  ⚡ Detected network filesystem - copying to local /tmp for faster access...")
            self._local_copy_path = self._copy_to_local_tmp(data_path, layer_name)
            if self._local_copy_path != data_path:  # Copy succeeded
                data_path = self._local_copy_path
                print(f"  ✓ Using local copy: {data_path}")
                # Register cleanup to ensure it runs even on keyboard interrupt
                self._register_cleanup()
        elif disable_copy:
            print("  ℹ️  Local copy disabled via NPY_DISABLE_LOCAL_COPY environment variable")

        # Load info to get shape and dtype
        metadata_path = layer_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Missing metadata.json in {layer_dir}")

        with open(metadata_path, "r") as f:
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

        # Load metadata from metadata.json if needed (for filtering or metadata)
        self._metadata = None
        self.return_timestep = return_timestep  # NEW
        if filter_fn is not None or return_metadata or return_timestep or indices is not None:
            # Load entries from metadata.json (can be "entries" or "metadata" key)
            self._metadata = info.get("entries", info.get("metadata", []))
            if not self._metadata:
                raise ValueError(f"No entries/metadata found in {metadata_path}")
        else:
            print("  Skipping metadata load (no filtering/metadata needed) - faster startup")

        # Apply filtering
        if indices is not None:
            # Use pre-computed indices
            self.indices = indices
            self.use_direct_indexing = False
            print(f"  Using {len(indices)} pre-filtered indices")
        elif filter_fn is not None:
            # Apply filter on metadata (each entry is a prompt with multiple samples)
            if self._metadata is None:
                raise ValueError("Metadata not loaded but filter_fn provided")
            print("  Applying filter function on metadata...")
            self.indices = []
            filtered_prompts = 0
            for entry in self._metadata:
                if filter_fn(entry):
                    # Add ALL sample indices for this prompt
                    # Note: If filtering by index range and entries overlap the boundary,
                    # some indices may be outside the desired range. The caller should
                    # handle this by using an appropriate filter_fn that checks both
                    # start_idx and end_idx, or by post-filtering the indices.
                    self.indices.extend(range(entry["start_idx"], entry["end_idx"]))
                    filtered_prompts += 1
            self.use_direct_indexing = False
            print(
                f"  Filtered: {len(self.indices)} samples from {filtered_prompts}/{len(self._metadata)} prompts"
            )
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

        # Store feature_dim as instance variable for external access
        self._feature_dim = feature_dim

        # Build sorted start_idx list for binary search (if metadata loaded)
        self._entry_start_indices = None
        if self._metadata:
            self._entry_start_indices = [entry["start_idx"] for entry in self._metadata]
            print(
                f"  Built binary search index for {len(self._entry_start_indices)} metadata entries"
            )

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension of the representations."""
        return self._feature_dim

    @functools.lru_cache(maxsize=8192)  # Cache lookups for consecutive indices  # noqa: B019
    def _find_entry_for_index(self, real_idx: int) -> Optional[Dict]:
        """Find metadata entry that contains the given sample index using binary search."""
        if self._metadata is None or self._entry_start_indices is None:
            return None

        # Binary search O(log n) to find the entry containing real_idx
        # Find rightmost entry whose start_idx <= real_idx
        pos = bisect.bisect_right(self._entry_start_indices, real_idx) - 1

        if pos < 0 or pos >= len(self._metadata):
            return None

        entry = self._metadata[pos]

        # Verify the index is actually within this entry's range
        if entry["start_idx"] <= real_idx < entry["end_idx"]:
            # Convert to hashable dict for caching
            return entry

        return None

    def _get_metadata_for_index(self, real_idx: int) -> Dict:
        """Get metadata for a given sample index."""
        entry = self._find_entry_for_index(real_idx)
        if entry is None:
            return {}
        return entry

    def _calculate_timestep(self, real_idx: int) -> int:
        """
        Calculate timestep for a given sample index.

        Each prompt's data is stored as [n_timesteps, n_spatial, features]
        flattened to (n_timesteps * n_spatial, features).

        So for a sample at position i within the prompt's range:
        timestep = (i - start_idx) // n_spatial
        """
        entry = self._find_entry_for_index(real_idx)
        if entry is None:
            # This can happen if:
            # 1. Metadata is incomplete
            # 2. Index filtering created indices outside entry ranges
            # 3. Binary search failed (shouldn't happen if metadata is correct)
            # import warnings
            # warnings.warn(
            #     f"Could not find metadata entry for index {real_idx}. "
            #     f"Returning timestep 0 as fallback. This may indicate a filtering issue.",
            #     RuntimeWarning,
            #     stacklevel=2
            # )
            return 0  # Fallback

        position_in_entry = real_idx - entry["start_idx"]
        n_spatial = entry["n_spatial"]
        timestep = position_in_entry // n_spatial

        return timestep

    def __len__(self):
        if self.use_direct_indexing:
            return self.total_samples
        return len(self.indices)

    def __getitem__(self, idx):
        if self.use_direct_indexing:
            real_idx = idx
        else:
            real_idx = self.indices[idx]

        # Read from memmap and convert to torch (no intermediate copy needed)
        # torch.from_numpy() creates a view without copying, then .float() converts dtype
        rep_fp16 = self._full_data[real_idx]
        rep = torch.from_numpy(rep_fp16.copy()).float()  # Copy only once during torch conversion

        if self.transform is not None:
            rep = self.transform(rep)

        # NEW: Return timestep if requested
        if self.return_timestep:
            # Find which entry this sample belongs to and calculate timestep
            # Each entry has samples from [start_idx, end_idx)
            # Structure: [n_timesteps, n_spatial] flattened to (n_timesteps * n_spatial)
            # So: timestep = (position_within_entry) // n_spatial
            timestep = self._calculate_timestep(real_idx)
            if self.return_metadata:
                return (
                    rep,
                    timestep,
                    self._get_metadata_for_index(real_idx),
                )
            return rep, timestep

        if not self.return_metadata:
            return rep

        return rep, self._get_metadata_for_index(real_idx)

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
        """
        Copy data file to local /tmp for fast access with safety checks.

        Uses file locking to ensure only one process copies the file when
        multiple SLURM array jobs run on the same node.

        Uses reference counting to track how many processes are using the
        shared copy, so cleanup only happens when the last process exits.
        """
        import fcntl
        import hashlib
        import time

        # Create unique tmp directory for this SLURM job array (shared across tasks)
        # Use SLURM_ARRAY_JOB_ID if available (shared), otherwise SLURM_JOB_ID
        array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
        job_id = array_job_id if array_job_id else os.environ.get("SLURM_JOB_ID", "local")

        path_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:8]
        tmp_dir = Path(f"/tmp/npy_cache_{job_id}_{layer_name}_{path_hash}")  # noqa: S108
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = tmp_dir / "data.npy"
        lock_path = tmp_dir / ".copy.lock"
        done_path = tmp_dir / ".copy.done"
        refcount_path = tmp_dir / ".refcount"

        # Use file locking for all operations to ensure atomicity
        with open(lock_path, "w") as lock_file:
            try:
                # Try to acquire exclusive lock (non-blocking first)
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # Another process is copying, wait for it
                    print("  ⏳ Another process is copying data, waiting...")
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # Blocking wait

                # After acquiring lock, check if copy was completed by another process
                if done_path.exists() and tmp_path.exists():
                    # Increment refcount and return
                    count = 0
                    if refcount_path.exists():
                        try:
                            count = int(refcount_path.read_text().strip() or "0")
                        except (ValueError, OSError):
                            count = 0
                    refcount_path.write_text(str(count + 1))
                    print(f"  ✓ Local copy ready (refcount={count + 1}): {tmp_path}")
                    return tmp_path

                # We have the lock and need to do the copy
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
                    return source_path  # Fall back to network storage

                print(f"  ✓ /tmp space: {available_gb:.1f}GB available, {required_gb:.1f}GB needed")

                # Copy file with progress
                print(f"  📦 Copying {file_size_gb:.2f}GB to local /tmp...")
                start = time.time()

                try:
                    shutil.copy2(source_path, tmp_path)
                    # Mark copy as complete and set initial refcount
                    done_path.touch()
                    refcount_path.write_text("1")  # First user
                except OSError as e:
                    print(f"  ✗ Copy failed: {e}")
                    print("     Falling back to network storage")
                    # Clean up partial copy
                    if tmp_path.exists():
                        tmp_path.unlink()
                    return source_path  # Fall back to network storage

                elapsed = time.time() - start
                speed_gbps = file_size_gb / elapsed if elapsed > 0 else 0
                print(f"  ✓ Copy complete in {elapsed:.1f}s ({speed_gbps:.2f} GB/s)")
                print(f"  ✓ Using local copy (refcount=1): {tmp_path}")

                return tmp_path

            finally:
                # Release lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _register_cleanup(self):
        """Register cleanup handlers for signals"""

        # Register for common signals (SIGINT = Ctrl+C, SIGTERM = kill)
        def signal_handler(signum, frame):
            self.__del__()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def __del__(self):
        """Cleanup local copy only when last user exits (reference counting)"""
        if not hasattr(self, "_local_copy_path") or not self._local_copy_path:
            return

        if "/tmp/npy_cache_" not in str(self._local_copy_path):  # noqa: S108
            return

        try:
            import fcntl

            tmp_dir = self._local_copy_path.parent
            lock_path = tmp_dir / ".copy.lock"
            refcount_path = tmp_dir / ".refcount"

            if not tmp_dir.exists():
                return

            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    # Read and decrement refcount
                    count = 1
                    if refcount_path.exists():
                        try:
                            count = int(refcount_path.read_text().strip() or "1")
                        except (ValueError, OSError):
                            count = 1
                    count -= 1

                    if count <= 0:
                        # Last user - safe to delete
                        # Release lock before deleting (lock file is inside tmp_dir)
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        print(f"\n  🧹 Cleaned up local copy (last user): {tmp_dir}")
                    else:
                        # Other tasks still using it
                        refcount_path.write_text(str(count))
                        print(f"\n  ℹ️  Local copy still in use by {count} task(s)")
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    raise

            self._local_copy_path = None
        except Exception:  # noqa: S110
            pass  # Ignore cleanup errors

    @staticmethod
    def compute_train_val_indices(
        metadata_path: Path,
        validation_percent: float = 10.0,
        seed: int = 42,
    ) -> Tuple[List[int], List[int]]:
        """
        Compute train/val indices by splitting at the prompt level.

        This ensures all samples from a single prompt stay together
        in either train or validation set.

        Args:
            metadata_path: Path to metadata.json file
            validation_percent: Percentage for validation (0-100)
            seed: Random seed for reproducibility

        Returns:
            (train_indices, val_indices) - Lists of sample indices
        """
        with open(metadata_path, "r") as f:
            info = json.load(f)

        entries = info.get("entries", [])
        if not entries:
            raise ValueError(f"No entries found in {metadata_path}")

        # Get unique prompt keys
        prompt_to_entries: Dict[Tuple, List[int]] = {}
        for i, entry in enumerate(entries):
            key = (entry["prompt_nr"], entry.get("object", ""))
            if key not in prompt_to_entries:
                prompt_to_entries[key] = []
            prompt_to_entries[key].append(i)

        # Shuffle prompts and split
        prompt_keys = list(prompt_to_entries.keys())
        random.seed(seed)
        random.shuffle(prompt_keys)

        n_val = max(1, int(len(prompt_keys) * validation_percent / 100))
        val_prompt_keys = set(prompt_keys[:n_val])

        # Collect sample indices for each split
        train_indices = []
        val_indices = []

        for entry in entries:
            key = (entry["prompt_nr"], entry.get("object", ""))
            start_idx = entry["start_idx"]
            end_idx = entry["end_idx"]

            # Add all sample indices for this entry
            sample_indices = list(range(start_idx, end_idx))

            if key in val_prompt_keys:
                val_indices.extend(sample_indices)
            else:
                train_indices.extend(sample_indices)

        return train_indices, val_indices

    @classmethod
    def create_train_val_split(
        cls,
        cache_dir: Path,
        layer_name: str,
        validation_percent: float = 10.0,
        seed: int = 42,
        **kwargs,
    ) -> Tuple["RepresentationDataset", "RepresentationDataset"]:
        """
        Create train and validation datasets from a single data source.

        This is MUCH more efficient than copying data - it uses index-based
        splitting so both datasets share the same underlying memmap file.

        Args:
            cache_dir: Base cache directory
            layer_name: Layer name (e.g., 'unet_up_1_att_1')
            validation_percent: Percentage for validation (0-100)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to RepresentationDataset

        Returns:
            (train_dataset, val_dataset) tuple

        Example:
            train_ds, val_ds = RepresentationDataset.create_train_val_split(
                cache_dir=Path("/path/to/representations"),
                layer_name="unet_up_1_att_1",
                validation_percent=10.0,
            )
        """
        layer_dir = cache_dir / layer_name
        metadata_path = layer_dir / "metadata.json"

        print(f"Computing train/val split for {layer_name}...")
        train_indices, val_indices = cls.compute_train_val_indices(
            metadata_path=metadata_path,
            validation_percent=validation_percent,
            seed=seed,
        )

        print(f"  Train: {len(train_indices):,} samples")
        print(f"  Val: {len(val_indices):,} samples")

        # Create datasets with pre-computed indices
        # Note: use_local_copy only for train to avoid double copying
        train_dataset = cls(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=train_indices,
            use_local_copy=kwargs.pop("use_local_copy", True),
            **kwargs,
        )

        # Val dataset shares the same memmap - no local copy needed
        val_dataset = cls(
            cache_dir=cache_dir,
            layer_name=layer_name,
            indices=val_indices,
            use_local_copy=False,  # Share memmap with train
            **kwargs,
        )

        return train_dataset, val_dataset

"""
CUDA and GPU Utilities Module

Provides GPU detection, compute capability checks, and CUDA compatibility verification.
"""

import torch


def get_gpu_compute_capability() -> tuple[int, int] | None:
    """Get the GPU compute capability (SM version).

    Returns:
        Tuple of (major, minor) version or None if not available.
        e.g., (8, 0) for SM80 (Ampere), (7, 5) for SM75 (Turing)
    """
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def get_pytorch_supported_cuda_archs() -> list[int]:
    """Get list of CUDA SM architectures supported by current PyTorch build.

    Returns:
        List of supported SM versions (e.g., [37, 50, 60, 70, 75, 80, 86, 90])
    """
    try:
        if hasattr(torch.cuda, "get_arch_list"):
            arch_list = torch.cuda.get_arch_list()
            sm_versions = []
            for arch in arch_list:
                if arch.startswith("sm_"):
                    try:
                        sm_versions.append(int(arch.split("_")[1]))
                    except (ValueError, IndexError):
                        pass
            return sorted(sm_versions)
    except Exception:
        pass
    # Default known supported versions for typical PyTorch builds
    return [37, 50, 60, 61, 70, 75, 80, 86, 90]


# Module-level constants (computed once at import)
GPU_COMPUTE_CAPABILITY = get_gpu_compute_capability()
FLASH_ATTENTION_SUPPORTED = GPU_COMPUTE_CAPABILITY is not None and GPU_COMPUTE_CAPABILITY[0] >= 8
PYTORCH_SUPPORTED_ARCHS = get_pytorch_supported_cuda_archs()


def check_cuda_compatibility() -> tuple[bool, str, bool]:
    """Check if CUDA is available and compatible with GPU architecture.

    Returns:
        Tuple of (is_compatible, status_message, flash_attention_ok)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available", False

    try:
        gpu_name = torch.cuda.get_device_name(0)

        # Check compute capability
        if GPU_COMPUTE_CAPABILITY:
            major, minor = GPU_COMPUTE_CAPABILITY
            gpu_sm = major * 10 + minor
            sm_version = f"SM{major}{minor}"

            # Check if GPU architecture is supported by PyTorch
            max_supported = max(PYTORCH_SUPPORTED_ARCHS) if PYTORCH_SUPPORTED_ARCHS else 90

            if gpu_sm > max_supported:
                return (
                    False,
                    f"GPU {gpu_name} ({sm_version}) unsupported - current PyTorch build supports up to SM{max_supported}.",
                    False,
                )

        import warnings

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()

            for w in caught_warnings:
                if "not compatible with the current PyTorch installation" in str(w.message):
                    return (
                        False,
                        f"GPU {gpu_name} not compatible with PyTorch - update PyTorch",
                        False,
                    )

        if GPU_COMPUTE_CAPABILITY:
            major, minor = GPU_COMPUTE_CAPABILITY
            sm_version = f"SM{major}{minor}"
            if FLASH_ATTENTION_SUPPORTED:
                return True, f"Compatible (GPU: {gpu_name}, {sm_version})", True
            else:
                return (
                    True,
                    f"Compatible (GPU: {gpu_name}, {sm_version} - Flash Attention disabled)",
                    False,
                )

        return True, f"Compatible (GPU: {gpu_name})", False
    except Exception as e:
        error_msg = str(e)
        if "sm_" in error_msg or "not compatible" in error_msg.lower():
            return False, "GPU architecture not supported by PyTorch version", False
        return False, f"CUDA error: {error_msg[:50]}", False


CUDA_COMPATIBLE, CUDA_STATUS, CUDA_FLASH_ATTENTION_OK = check_cuda_compatibility()

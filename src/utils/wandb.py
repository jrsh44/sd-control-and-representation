from typing import Dict

import GPUtil
import psutil


def get_system_metrics(device: str) -> Dict:
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    gpu_metrics = {}
    if device == "cuda":
        try:
            gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
            gpu_metrics = {
                "gpu_memory_used": gpu.memoryUsed,
                "gpu_memory_total": gpu.memoryTotal,
                "gpu_load": gpu.load * 100,
            }
        except Exception:
            print("Warning: Could not retrieve GPU metrics.")

    return {"cpu_percent": cpu_percent, "memory_percent": memory.percent, **gpu_metrics}

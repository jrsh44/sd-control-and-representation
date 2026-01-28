"""Centralized state management for the dashboard."""

import platform
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class SystemState(Enum):
    """Main system states"""

    INITIALIZING = "initializing"
    IDLE = "idle"
    LOADING_MODEL = "loading_model"
    GENERATING = "generating"
    DETECTING = "detecting"
    ERROR = "error"


class ModelLoadState(Enum):
    """Individual model states"""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass
class GenerationProgress:
    """Track generation progress"""

    phase: str  # "original" or "unlearned"
    current_step: int
    total_steps: int
    start_time: float

    @property
    def progress(self) -> float:
        """Get progress as a float between 0 and 1"""
        return self.current_step / self.total_steps if self.total_steps > 0 else 0.0

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

    @property
    def estimated_remaining(self) -> float:
        """Estimate remaining time based on current progress"""
        if self.current_step == 0:
            return 0.0
        time_per_step = self.elapsed_time / self.current_step
        remaining_steps = self.total_steps - self.current_step
        return time_per_step * remaining_steps

    def format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


class DashboardState:
    """Centralized state management for the dashboard"""

    def __init__(self):
        self.system_state = SystemState.IDLE

        self.model_states: Dict[str, ModelLoadState] = {
            "sd_base": ModelLoadState.NOT_LOADED,
            "sae": ModelLoadState.NOT_LOADED,
            "nudenet": ModelLoadState.NOT_LOADED,
        }

        self.sd_pipe = None
        self.sae_model = None
        self.sae_stats = None
        self.nudenet_detector = None
        self.clip_model = None

        self.load_times: Dict[str, Optional[float]] = {
            "sd_base": None,
            "sae": None,
            "nudenet": None,
        }

        self.generation: Optional[GenerationProgress] = None

        self.logs = []
        self.max_logs = 50

        self.last_generation_time: Optional[float] = None

        self.temp_dir = Path(tempfile.gettempdir()) / "sd-dashboard"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "info") -> str:
        """Add a log entry and return formatted log string.

        Args:
            message: Log message text.
            level: Log level ('info', 'success', 'error', 'warning', 'loading').

        Returns:
            Formatted log string containing all recent logs.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        prefix = {
            "info": "[INFO]",
            "success": "[SUCCESS]",
            "error": "[ERROR]",
            "warning": "[WARNING]",
            "loading": "[LOADING]",
        }.get(level, "")

        log_entry = f"[{timestamp}] {prefix} {message}"
        self.logs.append(log_entry)

        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

        return "\n".join(self.logs)

    def set_model_state(
        self, model: str, model_state: ModelLoadState, load_time: Optional[float] = None
    ):
        """Update model state.

        Args:
            model: Model identifier ('sd_base', 'sae', 'nudenet').
            model_state: New state for the model.
            load_time: Optional load time in seconds.
        """
        self.model_states[model] = model_state
        if load_time is not None:
            self.load_times[model] = load_time

    def get_model_status_text(self) -> str:
        """Generate model status display.

        Returns:
            Formatted markdown text showing status of all models.
        """
        lines = ["**MODEL STATUS**", ""]

        status_icons = {
            ModelLoadState.NOT_LOADED: "○",
            ModelLoadState.LOADING: "◐",
            ModelLoadState.LOADED: "●",
            ModelLoadState.FAILED: "✕",
        }

        model_names = {
            "sd_base": "SD v1.5",
            "sae": "SAE",
            "nudenet": "NudeNet",
        }

        for model, model_state in self.model_states.items():
            icon = status_icons.get(model_state, "?")
            name = model_names.get(model, model)
            status = model_state.value.replace("_", " ").title()

            if self.load_times.get(model):
                status += f" ({self.load_times[model]:.1f}s)"

            lines.append(f"{icon} **{name}:** {status}")

        return "\n".join(lines)

    def start_generation(self, total_steps: int, phase: str):
        """Start tracking generation progress.

        Args:
            total_steps: Total number of denoising steps.
            phase: Generation phase ('original' or 'unlearned').
        """
        self.generation = GenerationProgress(
            phase=phase, current_step=0, total_steps=total_steps, start_time=time.time()
        )

    def update_generation(self, step: int):
        """Update generation progress.

        Args:
            step: Current step number.
        """
        if self.generation:
            self.generation.current_step = step

    def get_progress_text(self) -> str:
        """Get formatted progress text.

        Returns:
            Formatted markdown text showing generation progress.
        """
        if not self.generation:
            return "*Ready to generate*"

        g = self.generation
        progress_pct = int(g.progress * 100)
        progress_bar = "█" * int(g.progress * 20) + "░" * (20 - int(g.progress * 20))

        return f"""**GENERATION PROGRESS**

**Phase:** {g.phase.title()}
**Step:** {g.current_step}/{g.total_steps} `[{progress_bar}]` {progress_pct}%

**Elapsed:** {g.format_time(g.elapsed_time)} | **Remaining:** {g.format_time(g.estimated_remaining)}
"""

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dictionary containing OS, Python, PyTorch, CUDA, and GPU details.
        """
        info = {
            "OS": platform.system(),
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "Temp Directory": str(self.temp_dir),
        }

        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
            info["CUDA Version"] = torch.version.cuda
            info["GPU Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"

        return info

    def clear_logs(self):
        """Clear all logs.

        Returns:
            Formatted log string with clear confirmation message.
        """
        self.logs = []
        return self.log("System log cleared", "info")

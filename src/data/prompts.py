"""
Prompt loading utilities.
Shared across all model implementations.
"""

from pathlib import Path
from typing import Dict


def load_prompts_from_directory(prompts_dir: Path) -> Dict[str, Dict[int, str]]:
    """
    Load prompts from .txt files in the specified directory.
    Each file represents one object, and each line has format: ID; prompt
    Skips entries with empty prompts.

    Args:
        prompts_dir: Path to directory containing .txt files

    Returns:
        Dictionary mapping object names to dict of {prompt_id: prompt_text}
    """
    prompts_by_object = {}
    txt_files = sorted(prompts_dir.glob("*.txt"))

    for txt_file in txt_files:
        object_name = txt_file.stem.replace("sd_prompt_", "")
        prompts = {}
        skipped = 0

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ";" in line:
                    parts = line.split(";", 1)
                    prompt_id = int(parts[0].strip())
                    prompt_text = parts[1].strip() if len(parts) > 1 else ""

                    if not prompt_text:
                        skipped += 1
                        continue

                    prompts[prompt_id] = prompt_text

        if prompts:
            prompts_by_object[object_name] = prompts
            status = f"  Loaded {len(prompts)} prompts for object: {object_name}"
            if skipped > 0:
                status += f" (skipped {skipped} empty)"
            print(status)

    return prompts_by_object

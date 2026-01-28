"""
Prompt loading utilities.
Shared across all model implementations.
"""

from pathlib import Path
from typing import Dict, List, Tuple


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


def load_base_prompts(path: Path) -> List[Tuple[int, str]]:
    """
    Load base prompts file with {} placeholders.
    Expected format: id;prompt_with_{}

    Args:
        path: Path to base prompts file

    Returns:
        List of tuples (prompt_id, prompt_template)
    """
    prompts: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                idx_str, template = line.split(";", 1)
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    idx = len(prompts) + 1
            else:
                idx = len(prompts) + 1
                template = line
            prompts.append((idx, template.strip()))
    return prompts


def load_classes_file(path: Path) -> Dict[int, str]:
    """
    Load classes file.
    Expected format: id;label

    Args:
        path: Path to classes file

    Returns:
        Dictionary mapping class_id to class_label
    """
    classes: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                idx_str, label = line.split(";", 1)
                try:
                    idx = int(idx_str.strip())
                except ValueError:
                    continue
                classes[idx] = label.strip()
            else:
                idx = max(classes.keys(), default=0) + 1
                classes[idx] = line
    return classes


def build_prompts_by_class(
    base_prompts: List[Tuple[int, str]],
    class_map: Dict[int, str],
    selected_ids: List[int],
) -> Dict[str, Dict[int, str]]:
    """
    Build prompts for each class by filling {} placeholders.

    Args:
        base_prompts: List of (prompt_id, prompt_template) tuples
        class_map: Dictionary mapping class_id to class_label
        selected_ids: List of class IDs to process

    Returns:
        Dictionary mapping class_label -> {prompt_nr: prompt_text}
    """
    prompts_by_class: Dict[str, Dict[int, str]] = {}
    for cid in selected_ids:
        if cid not in class_map:
            print(f"Warning: class id {cid} not found in classes file, skipping")
            continue
        label = class_map[cid]
        obj_prompts: Dict[int, str] = {}
        for idx, template in base_prompts:
            if "{}" in template:
                prompt_text = template.replace("{}", label)
            else:
                prompt_text = f"{template} {label}"
            obj_prompts[idx] = prompt_text
        prompts_by_class[label] = obj_prompts
    return prompts_by_class

"""Unit tests for src/data/prompts.py"""

import tempfile
from pathlib import Path
import pytest
from src.data.prompts import (
    build_prompts_by_class,
    load_base_prompts,
    load_classes_file,
    load_prompts_from_directory,
)


def test_load_prompts_from_directory(tmp_path):
    """Test loading prompts from directory."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    test_file = prompts_dir / "sd_prompt_cat.txt"
    test_file.write_text("1; A cat\n2; Another cat\n")

    result = load_prompts_from_directory(prompts_dir)

    assert "cat" in result
    assert len(result["cat"]) == 2
    assert result["cat"][1] == "A cat"


def test_load_base_prompts(tmp_path):
    """Test loading base prompt templates."""
    prompts_file = tmp_path / "base.txt"
    prompts_file.write_text("1;A photo of {}\n2;A painting of {}\n")

    result = load_base_prompts(prompts_file)

    assert len(result) == 2
    assert result[0] == (1, "A photo of {}")


def test_load_classes_file(tmp_path):
    """Test loading classes file."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("1;cat\n2;dog\n")

    result = load_classes_file(classes_file)

    assert len(result) == 2
    assert result[1] == "cat"


def test_build_prompts_by_class():
    """Test building prompts for classes."""
    base_prompts = [(1, "A photo of {}")]
    class_map = {1: "cat"}

    result = build_prompts_by_class(base_prompts, class_map, [1])

    assert result["cat"][1] == "A photo of cat"


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

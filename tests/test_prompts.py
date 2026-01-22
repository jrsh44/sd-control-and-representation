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


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# load_prompts_from_directory Tests
# =============================================================================


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


def test_load_prompts_empty_directory():
    """Test loading prompts from empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        result = load_prompts_from_directory(prompts_dir)

        assert result == {}


def test_load_prompts_skip_empty_lines():
    """Test that empty lines are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        prompt_file = prompts_dir / "test.txt"
        prompt_file.write_text("1; prompt 1\n\n2; prompt 2\n")

        result = load_prompts_from_directory(prompts_dir)

        assert "test" in result
        assert len(result["test"]) == 2


def test_load_prompts_skip_empty_prompts():
    """Test that entries with empty prompts are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        prompt_file = prompts_dir / "test.txt"
        prompt_file.write_text("1; prompt 1\n2; \n3; prompt 3\n")

        result = load_prompts_from_directory(prompts_dir)

        assert "test" in result
        assert len(result["test"]) == 2
        assert 1 in result["test"]
        assert 3 in result["test"]
        assert 2 not in result["test"]


def test_load_prompts_multiple_files():
    """Test loading prompts from multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        (prompts_dir / "cats.txt").write_text("1; a cat\n2; two cats\n")
        (prompts_dir / "dogs.txt").write_text("1; a dog\n")

        result = load_prompts_from_directory(prompts_dir)

        assert "cats" in result
        assert "dogs" in result
        assert len(result["cats"]) == 2
        assert len(result["dogs"]) == 1


def test_load_prompts_sd_prompt_prefix():
    """Test that sd_prompt_ prefix is removed from object names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        prompt_file = prompts_dir / "sd_prompt_test_obj.txt"
        prompt_file.write_text("1; test prompt\n")

        result = load_prompts_from_directory(prompts_dir)

        assert "test_obj" in result
        assert "sd_prompt_test_obj" not in result


def test_load_prompts_unicode():
    """Test loading prompts with Unicode characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_dir = Path(tmpdir)

        prompt_file = prompts_dir / "test.txt"
        prompt_file.write_text("1; 猫の写真\n2; café scene\n", encoding="utf-8")

        result = load_prompts_from_directory(prompts_dir)

        assert "test" in result
        assert result["test"][1] == "猫の写真"
        assert result["test"][2] == "café scene"


# =============================================================================
# load_base_prompts Tests
# =============================================================================


def test_load_base_prompts(tmp_path):
    """Test loading base prompt templates."""
    prompts_file = tmp_path / "base.txt"
    prompts_file.write_text("1;A photo of {}\n2;A painting of {}\n")

    result = load_base_prompts(prompts_file)

    assert len(result) == 2
    assert result[0] == (1, "A photo of {}")


def test_load_base_prompts_simple():
    """Test loading base prompts with simple format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_file = Path(tmpdir) / "base.txt"
        prompt_file.write_text("1; a photo of {}\n2; {} in nature\n")

        result = load_base_prompts(prompt_file)

        assert len(result) == 2
        assert result[0] == (1, "a photo of {}")
        assert result[1] == (2, "{} in nature")


def test_load_base_prompts_no_id():
    """Test loading base prompts without IDs (sequential numbering)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_file = Path(tmpdir) / "base.txt"
        prompt_file.write_text("a photo of {}\n{} in nature\n")

        result = load_base_prompts(prompt_file)

        assert len(result) == 2
        assert result[0][0] == 1  # Auto-assigned ID
        assert result[1][0] == 2


def test_load_base_prompts_mixed_format():
    """Test loading base prompts with mixed ID/no-ID format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_file = Path(tmpdir) / "base.txt"
        prompt_file.write_text("10; a photo of {}\n{} in nature\n20; {} at sunset\n")

        result = load_base_prompts(prompt_file)

        assert len(result) == 3
        assert result[0] == (10, "a photo of {}")
        assert result[1][0] == 2
        assert result[2] == (20, "{} at sunset")


def test_load_base_prompts_invalid_id():
    """Test loading base prompts with invalid ID falls back to sequential."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_file = Path(tmpdir) / "base.txt"
        prompt_file.write_text("invalid_id; a photo of {}\n")

        result = load_base_prompts(prompt_file)

        assert len(result) == 1
        assert result[0][0] == 1  # Falls back to sequential


def test_load_base_prompts_empty_lines():
    """Test loading base prompts skips empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_file = Path(tmpdir) / "base.txt"
        prompt_file.write_text("1; a photo of {}\n\n2; {} in nature\n\n")

        result = load_base_prompts(prompt_file)

        assert len(result) == 2


# =============================================================================
# load_classes_file Tests
# =============================================================================


def test_load_classes_file(tmp_path):
    """Test loading classes file."""
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("1;cat\n2;dog\n")

    result = load_classes_file(classes_file)

    assert len(result) == 2
    assert result[1] == "cat"


def test_load_classes_file_simple():
    """Test loading classes file with IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classes_file = Path(tmpdir) / "classes.txt"
        classes_file.write_text("1; cat\n2; dog\n3; bird\n")

        result = load_classes_file(classes_file)

        assert len(result) == 3
        assert result[1] == "cat"
        assert result[2] == "dog"
        assert result[3] == "bird"


def test_load_classes_file_no_id():
    """Test loading classes file without IDs (auto-increment)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classes_file = Path(tmpdir) / "classes.txt"
        classes_file.write_text("cat\ndog\nbird\n")

        result = load_classes_file(classes_file)

        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result


def test_load_classes_file_mixed_format():
    """Test loading classes file with mixed ID/no-ID format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classes_file = Path(tmpdir) / "classes.txt"
        classes_file.write_text("10; cat\ndog\n20; bird\n")

        result = load_classes_file(classes_file)

        assert len(result) == 3
        assert result[10] == "cat"
        assert 11 in result  # Auto-increment from max (10)
        assert result[20] == "bird"


def test_load_classes_file_invalid_id():
    """Test loading classes file with invalid ID is skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classes_file = Path(tmpdir) / "classes.txt"
        classes_file.write_text("invalid; cat\n1; dog\n")

        result = load_classes_file(classes_file)

        # Invalid ID is skipped, only valid entries
        assert len(result) == 1
        assert result[1] == "dog"


def test_load_classes_file_empty_lines():
    """Test loading classes file skips empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        classes_file = Path(tmpdir) / "classes.txt"
        classes_file.write_text("1; cat\n\n2; dog\n\n")

        result = load_classes_file(classes_file)

        assert len(result) == 2


# =============================================================================
# build_prompts_by_class Tests
# =============================================================================


def test_build_prompts_by_class():
    """Test building prompts for classes."""
    base_prompts = [(1, "A photo of {}")]
    class_map = {1: "cat"}

    result = build_prompts_by_class(base_prompts, class_map, [1])

    assert result["cat"][1] == "A photo of cat"


def test_build_prompts_by_class_simple():
    """Test building prompts by class with placeholder replacement."""
    base_prompts = [(1, "a photo of {}"), (2, "{} in nature")]
    class_map = {1: "cat", 2: "dog"}
    selected_ids = [1, 2]

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    assert "cat" in result
    assert "dog" in result
    assert result["cat"][1] == "a photo of cat"
    assert result["cat"][2] == "cat in nature"
    assert result["dog"][1] == "a photo of dog"


def test_build_prompts_by_class_no_placeholder():
    """Test building prompts when template has no placeholder."""
    base_prompts = [(1, "beautiful scenery")]
    class_map = {1: "cat"}
    selected_ids = [1]

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    # When no {}, should append class name
    assert result["cat"][1] == "beautiful scenery cat"


def test_build_prompts_by_class_missing_class():
    """Test building prompts with missing class ID."""
    base_prompts = [(1, "a photo of {}")]
    class_map = {1: "cat"}
    selected_ids = [1, 999]  # 999 doesn't exist

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    # Should only have cat, not 999
    assert len(result) == 1
    assert "cat" in result


def test_build_prompts_by_class_empty_selected():
    """Test building prompts with empty selected IDs."""
    base_prompts = [(1, "a photo of {}")]
    class_map = {1: "cat", 2: "dog"}
    selected_ids = []

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    assert result == {}


def test_build_prompts_by_class_multiple_placeholders():
    """Test building prompts with multiple placeholders."""
    base_prompts = [(1, "a {} sitting with another {}")]
    class_map = {1: "cat"}
    selected_ids = [1]

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    # Should replace all {} occurrences
    assert "cat" in result["cat"][1]


def test_build_prompts_by_class_special_characters():
    """Test building prompts with special characters in class names."""
    base_prompts = [(1, "a photo of {}")]
    class_map = {1: "cat & dog"}
    selected_ids = [1]

    result = build_prompts_by_class(base_prompts, class_map, selected_ids)

    assert "cat & dog" in result
    assert result["cat & dog"][1] == "a photo of cat & dog"

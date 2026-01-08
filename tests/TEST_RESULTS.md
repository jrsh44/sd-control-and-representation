# Unit Test Results

This document describes all unit tests and their execution results.

## Test Execution

### How to Run Tests

On SLURM cluster:
```bash
# Get compute node
srun --account=mi2lab --partition=short --time=0-01:10:00 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --mem=40G --pty /bin/bash

# Navigate to project and activate environment
cd /mnt/evafs/groups/mi2lab/bjezierski/sd-control-and-representation
source .venv/bin/activate

# Run tests
tests/test.sh -v
```

Alternatively, use pytest directly:
```bash
pytest tests/ -v
```

---

## Test Results

**Latest Execution:**
- **Date:** December 17, 2025
- **Total Tests:** 32
- **Passed:** 32
- **Failed:** 0
- **Execution Time:** 409.15 seconds (6:49)

```
================================================= test session starts ==================================================
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
collected 32 items

tests/test_cache.py::test_cache_initialization PASSED                                                [  3%]
tests/test_cache.py::test_cache_fp32_vs_fp16 PASSED                                                  [  6%]
tests/test_cache.py::test_save_metadata PASSED                                                       [  9%]
tests/test_cache.py::test_initialize_layer PASSED                                                    [ 12%]
tests/test_cache.py::test_atomic_counter PASSED                                                      [ 15%]
tests/test_cache.py::test_save_representation PASSED                                                 [ 18%]
tests/test_dataset.py::test_dataset_initialization PASSED                                            [ 21%]
tests/test_dataset.py::test_dataset_getitem PASSED                                                   [ 25%]
tests/test_dataset.py::test_dataset_filtering PASSED                                                 [ 28%]
tests/test_dataset.py::test_dataset_with_metadata PASSED                                             [ 31%]
tests/test_dataset.py::test_dataset_with_timestep PASSED                                             [ 34%]
tests/test_dataset.py::test_dataset_with_indices PASSED                                              [ 37%]
tests/test_dataset.py::test_dataset_missing_layer PASSED                                             [ 40%]
tests/test_prompts.py::test_load_prompts_from_directory PASSED                                       [ 43%]
tests/test_prompts.py::test_load_base_prompts PASSED                                                 [ 46%]
tests/test_prompts.py::test_load_classes_file PASSED                                                 [ 50%]
tests/test_prompts.py::test_build_prompts_by_class PASSED                                            [ 53%]
tests/test_sae_training.py::test_compute_avg_max_cosine_similarity PASSED                            [ 56%]
tests/test_sae_training.py::test_criterion_laux PASSED                                               [ 59%]
tests/test_sae_training.py::test_extract_input PASSED                                                [ 62%]
tests/test_sae_training.py::test_compute_reconstruction_error PASSED                                 [ 65%]
tests/test_sae_training.py::test_log_metrics PASSED                                                  [ 68%]
tests/test_sae_training.py::test_extract_input_error PASSED                                          [ 71%]
tests/test_sd_v1_5.py::test_layer_path_enum PASSED                                                   [ 75%]
tests/test_sd_v1_5.py::test_get_nested_module_basic PASSED                                           [ 78%]
tests/test_sd_v1_5.py::test_get_nested_module_deep PASSED                                            [ 81%]
tests/test_visualization.py::test_display_image PASSED                                               [ 84%]
tests/test_visualization.py::test_display_sequence PASSED                                            [ 87%]
tests/test_visualization.py::test_display_gif PASSED                                                 [ 90%]
tests/test_visualization.py::test_display_image_invalid_input PASSED                                 [ 93%]
tests/test_visualization.py::test_display_sequence_empty_list PASSED                                 [ 96%]
tests/test_visualization.py::test_display_sequence_sampling PASSED                                   [100%]

============================================ 32 passed in 409.15s (0:06:49) ============================================
```

---

## Test Details

### Cache Tests (test_cache.py) - 6 tests

**test_cache_initialization**
- **Purpose:** Verifies that RepresentationCache can be initialized correctly
- **What it tests:** Creates a cache instance and checks that the cache directory exists and dtype is set to fp16
- **Result:** ✅ PASSED

**test_cache_fp32_vs_fp16**
- **Purpose:** Tests that cache can be created with different precision levels
- **What it tests:** Creates two caches (fp32 and fp16) and verifies their dtypes are correct
- **Result:** ✅ PASSED

**test_save_metadata**
- **Purpose:** Tests metadata persistence to disk
- **What it tests:** Creates a cache, saves metadata for a layer, and verifies the metadata file exists
- **Result:** ✅ PASSED

**test_initialize_layer**
- **Purpose:** Tests memmap layer initialization with correct shape
- **What it tests:** Initializes a layer with specific dimensions and verifies the memmap array shape, dtype, and metadata files are created
- **Result:** ✅ PASSED

**test_atomic_counter**
- **Purpose:** Tests atomic counter for multi-process safe index allocation
- **What it tests:** Increments counter twice and verifies sequential index allocation (0, then 10)
- **Result:** ✅ PASSED

**test_save_representation**
- **Purpose:** Tests saving a representation tensor to cache
- **What it tests:** Initializes layer, creates test representation, saves it, and verifies data was written to memmap
- **Result:** ✅ PASSED

---

### Dataset Tests (test_dataset.py) - 7 tests

**test_dataset_initialization**
- **Purpose:** Tests basic dataset initialization without filtering
- **What it tests:** Creates test dataset with metadata and verifies it loads with correct sample count and shape
- **Result:** ✅ PASSED

**test_dataset_getitem**
- **Purpose:** Tests retrieving individual items from dataset
- **What it tests:** Gets first item from dataset and verifies it's a PyTorch tensor with correct shape and dtype
- **Result:** ✅ PASSED

**test_dataset_filtering**
- **Purpose:** Tests dataset filtering by metadata attributes
- **What it tests:** Applies filter function to select only "cat" samples and verifies correct count (5 out of 10)
- **Result:** ✅ PASSED

**test_dataset_with_metadata**
- **Purpose:** Tests dataset configured to return metadata along with data
- **What it tests:** Creates dataset with return_metadata=True and verifies initialization
- **Result:** ✅ PASSED

**test_dataset_with_timestep**
- **Purpose:** Tests dataset configured to return timestep information
- **What it tests:** Creates dataset with return_timestep=True and verifies initialization
- **Result:** ✅ PASSED

**test_dataset_with_indices**
- **Purpose:** Tests dataset with pre-computed index list
- **What it tests:** Provides specific indices [0,1,2] and verifies dataset length matches
- **Result:** ✅ PASSED

**test_dataset_missing_layer**
- **Purpose:** Tests error handling for missing layer directory
- **What it tests:** Attempts to load nonexistent layer and verifies ValueError is raised
- **Result:** ✅ PASSED

---

### Prompt Tests (test_prompts.py) - 4 tests

**test_load_prompts_from_directory**
- **Purpose:** Tests loading prompts from a directory structure
- **What it tests:** Creates temporary directory with prompt files, loads them, and verifies content
- **Result:** ✅ PASSED

**test_load_base_prompts**
- **Purpose:** Tests loading base prompts from CSV file
- **What it tests:** Creates a CSV file with base prompts and verifies they are loaded correctly
- **Result:** ✅ PASSED

**test_load_classes_file**
- **Purpose:** Tests loading class definitions from file
- **What it tests:** Creates a classes file, loads it, and checks that classes are parsed correctly
- **Result:** ✅ PASSED

**test_build_prompts_by_class**
- **Purpose:** Tests building prompts by combining base prompts with classes
- **What it tests:** Verifies that prompts are correctly constructed from templates and class names
- **Result:** ✅ PASSED

---

### SAE Training Tests (test_sae_training.py) - 6 tests

**test_compute_avg_max_cosine_similarity**
- **Purpose:** Tests cosine similarity computation for SAE weight matrix
- **What it tests:** Computes average max cosine similarity and verifies result is in valid range [0.99, 1.0]
- **Result:** ✅ PASSED

**test_criterion_laux**
- **Purpose:** Tests auxiliary loss criterion for SAE training
- **What it tests:** Computes loss with reconstruction and auxiliary terms, verifies it's non-negative
- **Result:** ✅ PASSED

**test_extract_input**
- **Purpose:** Tests input extraction from different batch formats
- **What it tests:** Verifies that data can be extracted from tuples, dicts, and tensors correctly
- **Result:** ✅ PASSED

**test_compute_reconstruction_error**
- **Purpose:** Tests R² reconstruction error computation
- **What it tests:** Creates identical input and reconstruction, verifies R² score ≈ 1.0 for perfect reconstruction
- **Result:** ✅ PASSED

**test_log_metrics**
- **Purpose:** Tests metrics logging with different monitoring levels
- **What it tests:** Verifies logging behavior at level 0 (no logging) and level 1 (basic logging with lr and loss)
- **Result:** ✅ PASSED

**test_extract_input_error**
- **Purpose:** Tests error handling for invalid batch dictionary
- **What it tests:** Passes dict without 'data' key and verifies ValueError is raised with correct message
- **Result:** ✅ PASSED

---

### Stable Diffusion v1.5 Tests (test_sd_v1_5.py) - 3 tests

**test_layer_path_enum**
- **Purpose:** Tests that layer path enumeration includes important layers
- **What it tests:** Checks that LayerPath enum contains UNET_UP_1_ATT_1 (most important layer for this project)
- **Result:** ✅ PASSED

**test_get_nested_module_basic**
- **Purpose:** Tests retrieving nested PyTorch modules by path
- **What it tests:** Creates simple nested module structure and retrieves submodules by string path
- **Result:** ✅ PASSED

**test_get_nested_module_deep**
- **Purpose:** Tests retrieving deeply nested modules
- **What it tests:** Verifies that deeply nested module paths (e.g., "a.b.c") work correctly
- **Result:** ✅ PASSED

---

### Visualization Tests (test_visualization.py) - 6 tests

**test_display_image**
- **Purpose:** Tests single image display function with mocked matplotlib
- **What it tests:** Creates test image, calls display_image(), and verifies plt.imshow and plt.show are called correctly with the image
- **Result:** ✅ PASSED

**test_display_sequence**
- **Purpose:** Tests displaying a sequence of images with mocked matplotlib
- **What it tests:** Creates 3 images, calls display_sequence(), and verifies plt.imshow is called for each image and plt.show is called once
- **Result:** ✅ PASSED

**test_display_gif**
- **Purpose:** Tests GIF display functionality with mocked IPython display
- **What it tests:** Creates frames, calls display_gif(), and verifies IPython.display.display is called to show the GIF
- **Result:** ✅ PASSED

**test_display_image_invalid_input**
- **Purpose:** Tests error handling for invalid input to display_image
- **What it tests:** Passes a string instead of PIL Image and verifies plt.show is not called (error is handled gracefully)
- **Result:** ✅ PASSED

**test_display_sequence_empty_list**
- **Purpose:** Tests edge case handling for empty image list
- **What it tests:** Passes empty list to display_sequence() and verifies plt.show is not called (handled gracefully)
- **Result:** ✅ PASSED

**test_display_sequence_sampling**
- **Purpose:** Tests display_sequence sampling rate logic
- **What it tests:** Creates 10 images with sampling_rate=3, verifies plt.imshow is called 4 times (indices 0, 3, 6, 9 - last always included)
- **Result:** ✅ PASSED

---

## Summary by Module

| Module | Tests | Passed | Coverage |
|--------|-------|--------|----------|
| Cache | 6 | 6 | Initialization, dtypes, metadata, layer setup, atomic counter, saving |
| Dataset | 7 | 7 | Loading, filtering, indices, metadata, timestep, error handling |
| Prompts | 4 | 4 | Directory loading, CSV parsing, class files, prompt building |
| SAE Training | 6 | 6 | Similarity, loss, input extraction, reconstruction, logging, errors |
| SD v1.5 | 3 | 3 | Layer enumeration, module access, deep nesting |
| Visualization | 6 | 6 | Mocked matplotlib/IPython, error handling, edge cases, sampling logic |
| **TOTAL** | **32** | **32** | **100% pass rate** |

---

## Notes

- All tests run on **CPU only** (CUDA_VISIBLE_DEVICES="" is set in conftest.py)
- PyTorch behavior is identical on CPU vs GPU for logic testing
- Tests requiring GPU are marked with `@pytest.mark.requires_gpu` and skipped on CPU
- Total test execution time: ~409 seconds (6:49)
- **Visualization tests use mocking** to avoid headless environment issues - they verify function calls instead of actual display
- Added 16 new tests total: 3 cache, 7 dataset, 3 training, 3 visualization error/edge cases

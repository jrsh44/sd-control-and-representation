# Unit Tests Documentation

This directory contains unit tests for the `sd-control-and-representation` project.

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Test Files

### test_cache.py
Tests for `src/data/cache.py` - RepresentationCache class for storing neural network representations.

| Function | Description |
|----------|-------------|
| `test_cache_initialization` | Verifies cache creates directory and sets correct dtype. |
| `test_cache_fp32_vs_fp16` | Tests cache initialization with different precision levels. |
| `test_cache_get_layer_path` | Verifies layer path construction returns correct Path object. |
| `test_cache_pid_tracking` | Checks that cache tracks the current process ID. |
| `test_cache_array_job_primary` | Tests cache with primary SLURM array job (task 0). |
| `test_cache_array_job_secondary` | Tests cache with secondary SLURM array job (task > 0). |
| `test_cache_no_array_job` | Tests cache behavior without array job configuration. |
| `test_save_metadata` | Verifies metadata is saved to temporary JSONL file. |
| `test_initialize_layer` | Tests memmap creation with correct shape and dtype. |
| `test_initialize_layer_creates_metadata` | Verifies metadata.json contains correct layer info. |
| `test_cache_multiple_layers` | Tests cache handling multiple layers simultaneously. |
| `test_atomic_counter` | Verifies atomic counter increments correctly. |
| `test_atomic_counter_multiple_increments` | Tests sequential counter increments return correct offsets. |
| `test_save_representation` | Tests saving representation tensor to cache. |
| `test_save_representation_shape_handling` | Verifies correct handling of 4D tensor shapes. |
| `test_save_representation_metadata_content` | Checks metadata entries contain correct fields. |
| `test_cache_dtype_conversion` | Tests torch tensor to numpy dtype conversion. |

---

### test_config.py
Tests for `src/models/config.py` - Model configuration and registry.

| Function | Description |
|----------|-------------|
| `test_model_config_creation` | Tests ModelConfig named tuple creation with all fields. |
| `test_model_config_default_pipeline` | Verifies default pipeline_type is 'sd_v1_5'. |
| `test_model_registry_sd_v1_5` | Tests SD_V1_5 model configuration values. |
| `test_model_registry_finetuned_saeuron` | Tests FINETUNED_SAEURON model configuration. |
| `test_model_registry_sd_v3` | Tests SD_V3 model configuration values. |
| `test_model_registry_enum_properties` | Verifies all registry enums have required properties. |
| `test_model_registry_unique_names` | Checks all models have unique config names. |
| `test_model_registry_value_access` | Tests accessing ModelConfig through value attribute. |

---

### test_dataset.py
Tests for `src/data/dataset.py` - RepresentationDataset for loading cached representations.

| Function | Description |
|----------|-------------|
| `test_dataset_initialization` | Tests basic dataset initialization without filtering. |
| `test_dataset_getitem` | Verifies item retrieval returns correct tensor shape and dtype. |
| `test_dataset_missing_layer` | Tests error handling for non-existent layer. |
| `test_dataset_feature_dim_property` | Verifies feature_dim property returns correct value. |
| `test_dataset_filtering` | Tests dataset filtering with simple filter function. |
| `test_dataset_complex_filter` | Tests filtering with multiple conditions. |
| `test_dataset_direct_indexing` | Verifies direct indexing mode when no filter applied. |
| `test_dataset_indirect_indexing_with_filter` | Tests indirect indexing with filter function. |
| `test_dataset_with_indices` | Tests dataset with pre-computed index list. |
| `test_dataset_empty_indices` | Verifies behavior with empty indices list. |
| `test_dataset_out_of_order_indices` | Tests dataset with non-sequential indices. |
| `test_dataset_with_metadata` | Tests return_metadata option. |
| `test_dataset_with_timestep` | Tests return_timestep option. |
| `test_dataset_with_transform` | Verifies custom transform function is applied. |
| `test_dataset_metadata_key_fallback` | Tests 'metadata' key fallback when 'entries' missing. |
| `test_dataset_fp32_dtype` | Tests dataset with float32 data. |
| `test_dataset_large_feature_dim` | Tests dataset with large feature dimension (2048). |
| `test_dataset_missing_metadata_with_no_filter` | Tests loading without metadata entries. |

---

### test_hooks.py
Tests for `src/models/sd_v1_5/hooks.py` and `layers.py` - Module path navigation utilities.

| Function | Description |
|----------|-------------|
| `test_layer_path_enum` | Verifies LayerPath enum has expected values. |
| `test_get_nested_module_single_level` | Tests single-level module attribute access. |
| `test_get_nested_module_multi_level` | Tests multi-level nested module access. |
| `test_get_nested_module_deep_nesting` | Tests deeply nested module retrieval (3+ levels). |
| `test_get_nested_module_with_sequential` | Tests index-based access in Sequential containers. |
| `test_get_nested_module_with_modulelist` | Tests index-based access in ModuleList. |
| `test_get_nested_module_negative_index` | Tests negative index support for module access. |
| `test_get_nested_module_complex_path` | Tests complex paths with mixed indices and attributes. |
| `test_get_nested_module_invalid_path` | Verifies AttributeError for invalid module paths. |
| `test_get_nested_module_invalid_index` | Verifies IndexError for out-of-bounds indices. |
| `test_get_nested_module_empty_path` | Tests empty path returns model itself. |
| `test_get_nested_module_parametric` | Tests access to various module types (Conv, BN, etc.). |

---

### test_model_loader.py
Tests for `src/utils/model_loader.py` - Model downloading and loading utilities.

| Function | Description |
|----------|-------------|
| `test_model_loader_initialization` | Tests ModelLoader initialization with environment config. |
| `test_model_loader_initialization_custom_root` | Tests initialization with custom project root. |
| `test_model_loader_no_cache_dir` | Verifies error when CACHE_DIR not set. |
| `test_get_model_path_huggingface` | Tests model path for HuggingFace models. |
| `test_get_model_path_cached` | Tests local path returned when model is cached. |
| `test_download_from_gdrive_not_exists` | Tests Google Drive download when model missing. |
| `test_download_from_gdrive_already_exists` | Tests skip download when model exists. |
| `test_download_from_gdrive_error` | Tests error handling for failed downloads. |
| `test_model_loader_different_models` | Tests loader with different model configurations. |

---

### test_prompts.py
Tests for `src/data/prompts.py` - Prompt loading and building utilities.

| Function | Description |
|----------|-------------|
| `test_load_prompts_from_directory` | Tests loading prompts from directory with sd_prompt_ prefix. |
| `test_load_prompts_empty_directory` | Verifies empty dict for empty directory. |
| `test_load_prompts_skip_empty_lines` | Tests that empty lines are ignored. |
| `test_load_prompts_skip_empty_prompts` | Tests that empty prompt entries are skipped. |
| `test_load_prompts_multiple_files` | Tests loading from multiple prompt files. |
| `test_load_prompts_sd_prompt_prefix` | Verifies sd_prompt_ prefix is stripped from names. |
| `test_load_prompts_unicode` | Tests Unicode character support in prompts. |
| `test_load_base_prompts` | Tests loading base prompt templates with IDs. |
| `test_load_base_prompts_simple` | Tests simple format with semicolon separator. |
| `test_load_base_prompts_no_id` | Tests auto-increment IDs when not specified. |
| `test_load_base_prompts_mixed_format` | Tests mixed ID and no-ID formats. |
| `test_load_base_prompts_invalid_id` | Tests fallback for invalid ID values. |
| `test_load_base_prompts_empty_lines` | Tests empty line handling. |
| `test_load_classes_file` | Tests loading class mapping file. |
| `test_load_classes_file_simple` | Tests simple class file with IDs. |
| `test_load_classes_file_no_id` | Tests auto-increment for classes without IDs. |
| `test_load_classes_file_mixed_format` | Tests mixed ID formats in class file. |
| `test_load_classes_file_invalid_id` | Tests invalid ID handling. |
| `test_load_classes_file_empty_lines` | Tests empty line handling. |
| `test_build_prompts_by_class` | Tests prompt building with placeholder replacement. |
| `test_build_prompts_by_class_simple` | Tests basic placeholder replacement. |
| `test_build_prompts_by_class_no_placeholder` | Tests class name appending when no placeholder. |
| `test_build_prompts_by_class_missing_class` | Tests handling of missing class IDs. |
| `test_build_prompts_by_class_empty_selected` | Tests empty selected IDs list. |
| `test_build_prompts_by_class_multiple_placeholders` | Tests multiple {} placeholder replacement. |
| `test_build_prompts_by_class_special_characters` | Tests special characters in class names. |

---

### test_sae_metrics.py
Tests for `src/models/sae/training/metrics.py` - SAE training metrics computation.

| Function | Description |
|----------|-------------|
| `test_compute_reconstruction_error_2d` | Tests R² computation for 2D tensors. |
| `test_compute_reconstruction_error_4d` | Tests R² computation for 4D image-like tensors. |
| `test_compute_reconstruction_error_3d` | Tests R² computation for 3D sequence tensors. |
| `test_compute_reconstruction_error_poor` | Tests detection of poor reconstruction quality. |
| `test_compute_sparsity_metrics_sparse` | Tests sparsity metrics for sparse activations. |
| `test_compute_sparsity_metrics_dense` | Tests sparsity metrics for dense activations. |
| `test_compute_sparsity_metrics_all_zeros` | Tests metrics for all-zero tensors. |
| `test_compute_dictionary_metrics` | Tests dictionary feature statistics. |
| `test_compute_avg_max_cosine_similarity_identical` | Tests cosine similarity for identical rows. |
| `test_compute_avg_max_cosine_similarity_orthogonal` | Tests cosine similarity for orthogonal vectors. |
| `test_compute_avg_max_cosine_similarity_random` | Tests cosine similarity for random matrices. |
| `test_compute_sparsity_metrics_inf_handling` | Tests graceful handling of infinite values. |

---

### test_sae_training.py
Tests for `src/models/sae/training.py` - SAE training utilities.

| Function | Description |
|----------|-------------|
| `test_criterion_laux` | Tests auxiliary loss criterion computation. |
| `test_compute_reconstruction_error` | Tests internal R² reconstruction error. |
| `test_log_metrics` | Tests metrics logging at different monitoring levels. |

---

### test_sae_utils.py
Tests for `src/models/sae/training/utils.py` - SAE training helper functions.

| Function | Description |
|----------|-------------|
| `test_extract_input_tuple` | Tests input extraction from tuple batch. |
| `test_extract_input_list` | Tests input extraction from list batch. |
| `test_extract_input_dict_with_data` | Tests input extraction from dict with 'data' key. |
| `test_extract_input_dict_without_data` | Tests handling of dict without 'data' key. |
| `test_extract_input_tensor` | Tests direct tensor passthrough. |
| `test_get_dictionary_property` | Tests dictionary retrieval via property. |
| `test_get_dictionary_method` | Tests dictionary retrieval via method call. |
| `test_create_warmup_cosine_scheduler_disabled` | Tests scheduler returns None when disabled. |
| `test_create_warmup_cosine_scheduler_no_warmup` | Tests scheduler with zero warmup steps. |
| `test_create_warmup_cosine_scheduler_valid` | Tests valid scheduler creation and LR progression. |
| `test_create_warmup_cosine_scheduler_warmup_too_large` | Tests warmup steps clamping to total steps. |
| `test_create_warmup_cosine_scheduler_min_lr_ratio` | Tests minimum learning rate enforcement. |

---

### test_visualization.py
Tests for `src/utils/visualization.py` - Image display utilities.

| Function | Description |
|----------|-------------|
| `test_display_image` | Tests single image display with matplotlib. |
| `test_display_sequence` | Tests image sequence grid display. |
| `test_display_gif` | Tests GIF display with IPython. |
| `test_display_image_invalid_input` | Tests graceful handling of invalid input. |
| `test_display_sequence_empty_list` | Tests handling of empty image list. |
| `test_display_sequence_sampling` | Tests image sequence with sampling rate. |

---

### test_dashboard.py
Tests for `dashboard/` modules - Dashboard state, layers, and concepts.

| Function | Description |
|----------|-------------|
| `test_state_initialization` | Verifies DashboardState initializes with correct defaults. |
| `test_log_message` | Tests logging functionality with log level. |
| `test_model_state_update` | Tests model state updates with load times. |
| `test_get_model_status_text` | Tests model status text generation. |
| `test_generation_progress` | Tests generation progress tracking and formatting. |
| `test_get_all_layers` | Tests getting all UNet layers organized by category. |
| `test_get_layer_choices` | Tests layer choices formatting for dropdowns. |
| `test_get_layer_info_single` | Tests layer info retrieval for single layer. |
| `test_get_layer_info_empty` | Tests layer info with empty selection. |
| `test_get_layer_info_multiple` | Tests layer info with multiple layers. |
| `test_get_flat_layer_list` | Tests getting flat list of all layers. |
| `test_is_recommended_layer` | Tests checking if layer is recommended. |
| `test_load_concepts` | Tests concept loading from file. |
| `test_get_concept_choices` | Tests concept choices formatting. |
| `test_get_concept_info_empty` | Tests concept info with empty selection. |
| `test_get_concept_info_single` | Tests concept info with single selection. |
| `test_get_concept_info_multiple` | Tests concept info with multiple selection. |
| `test_get_concept_label` | Tests getting single concept label. |
| `test_validate_concepts` | Tests concept validation. |
| `test_load_concepts_missing_file` | Tests concept loading with missing file. |
| `test_state_lifecycle` | Tests complete state lifecycle. |
| `test_layer_concept_workflow` | Tests layer and concept selection workflow. | |

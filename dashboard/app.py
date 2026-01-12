"""
SD Control & Representation - Interactive Dashboard
Local deployment with comprehensive state management and progress tracking
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

# Disable xformers memory efficient attention before importing torch
# This prevents CUDA kernel errors on unsupported GPU architectures
os.environ.setdefault("XFORMERS_DISABLED", "1")

import gradio as gr
import torch
from PIL import Image

# Add dashboard directory to path for imports when running directly
_dashboard_dir = Path(__file__).parent
_project_root = _dashboard_dir.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))
# Also add project root for src imports
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import CUDA utilities (provides GPU detection and compatibility checks)
from config.sae_config_loader import (  # noqa: E402
    get_concept_choices,
    get_feature_sums_path,
    get_layer_id,
    get_model_path,
    get_sae_hyperparameters,
    get_sae_model_choices,
    load_sae_config,
)
from core.model_loader import (  # noqa: E402
    load_nudenet_model as _load_nudenet_model,
)
from core.model_loader import (  # noqa: E402
    load_sae_model as _load_sae_model,
)

# Import model loading functions
from core.model_loader import (  # noqa: E402
    load_sd_model as _load_sd_model,
)

# Import state management
from core.state import (  # noqa: E402
    DashboardState,
    ModelLoadState,
    SystemState,
)
from utils.cuda import (  # noqa: E402
    CUDA_COMPATIBLE,
    CUDA_STATUS,
)

# Import detection utilities
from utils.detection import detect_content as _detect_content  # noqa: E402
from utils.detection import format_nudenet_comparison  # noqa: E402

# SAE and intervention imports
try:
    from overcomplete.sae import TopKSAE

    from src.models.sd_v1_5.layers import LayerPath
    from src.utils.RepresentationModifier import RepresentationModifier

    SAE_AVAILABLE = True
    SAE_IMPORT_ERROR = ""
    print("[OK] SAE dependencies loaded successfully")
except ImportError as e:
    SAE_AVAILABLE = False
    SAE_IMPORT_ERROR = str(e)
    print(f"[ERROR] SAE dependencies not available: {e}")

# NudeNet imports
try:
    from nudenet import NudeDetector

    NUDENET_AVAILABLE = True
    print("[OK] NudeNet loaded successfully")
except ImportError as e:
    NUDENET_AVAILABLE = False
    NUDENET_IMPORT_ERROR = str(e)
    print(f"[INFO] NudeNet not available: {e}")


# Global state
state = DashboardState()

# Load SAE configuration
try:
    SAE_CONFIG = load_sae_config()
    SAE_CONFIG_LOADED = True
except Exception as e:
    SAE_CONFIG = None
    SAE_CONFIG_LOADED = False
    print(f"Warning: Failed to load SAE config: {e}")


# Image Generation
def generate_image(prompt: str, steps: int, guidance: float, seed: int, progress=None):
    """Generate image from prompt with real-time progress updates"""
    if state.sd_pipe is None:
        return (
            None,
            "ERROR: SD model not loaded. Please load models first.",
            state.get_model_status_text(),
            state.log("Cannot generate: SD model not loaded", "error"),
            "**ERROR**\n\nSD model not loaded. Click 'Load Models' first.",
        )

    try:
        state.system_state = SystemState.GENERATING
        state.log(f"Starting generation with prompt: '{prompt[:50]}...'", "info")
        state.log(f"Parameters: steps={steps}, guidance={guidance}, seed={seed}", "info")

        # Set seed
        if seed == -1:
            import random

            seed = random.randint(0, 2**32 - 1)  # noqa: S311

        generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)

        # Start progress tracking
        state.start_generation(total_steps=steps, phase="original")

        # Progress callback - updates both state and Gradio progress bar
        def progress_callback(step, timestep, latents):
            state.update_generation(step)
            if progress is not None:
                progress((step, steps), desc=f"Generating... {step}/{steps}")

        # Generate image with callback
        start_time = time.time()

        # Check if pipeline supports callback
        try:
            output = state.sd_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback=progress_callback,
                callback_steps=1,
            )
        except TypeError:
            # Fallback without callback
            output = state.sd_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        generation_time = time.time() - start_time
        state.last_generation_time = generation_time

        image = output.images[0]

        state.system_state = SystemState.IDLE
        state.log(f"Generation complete in {generation_time:.2f}s", "success")

        # Update displays
        status_msg = f"GENERATION COMPLETE — {generation_time:.2f}s | Seed: {seed}"
        progress_msg = f"**GENERATION COMPLETE**\n\n✓ Image generated successfully\n**Time:** {generation_time:.2f}s | **Seed:** {seed}\n**Steps/sec:** {steps / generation_time:.2f}"

        return (
            image,
            status_msg,
            state.get_model_status_text(),
            state.log(f"Image generated successfully (seed: {seed})", "success"),
            progress_msg,
        )

    except Exception as e:
        state.system_state = SystemState.ERROR
        error_msg = f"Generation failed: {str(e)}"
        state.log(error_msg, "error")
        return (
            None,
            f"ERROR: {str(e)}",
            state.get_model_status_text(),
            state.log(error_msg, "error"),
            f"**ERROR**\n\n{str(e)}",
        )


def generate_with_intervention(
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    concept_configs: List[dict],
    layer_name: str = "UNET_UP_1_ATT_1",
    progress=None,
) -> Tuple[Any, str, str, str]:
    """
    Generate image with SAE intervention.

    Args:
        prompt: Text prompt for generation
        steps: Number of diffusion steps
        guidance: Guidance scale
        seed: Random seed
        concept_configs: List of concept configurations, each with:
            - id: Concept ID (e.g., 'exposed_breast')
            - strength: Intervention strength for this concept
            - neurons: Number of neurons to use for this concept
        layer_name: Layer to apply intervention on
        progress: Gradio progress tracker

    Returns:
        Tuple of (image, status_text, model_status, log_text)
    """
    print("\n" + "=" * 80)
    print("[DEBUG] generate_with_intervention() CALLED")
    print("=" * 80)
    print(f"[DEBUG] prompt: {prompt[:50]}...")
    print(f"[DEBUG] steps: {steps}, guidance: {guidance}, seed: {seed}")
    print(f"[DEBUG] concept_configs: {concept_configs}")
    print(f"[DEBUG] layer_name: {layer_name}")

    state.log("Starting generation WITH intervention...", "info")

    # Check SD pipeline
    print(f"[DEBUG] state.sd_pipe is None: {state.sd_pipe is None}")
    if state.sd_pipe is None:
        error_msg = "SD model not loaded. Please load model first."
        print(f"[DEBUG] ERROR: {error_msg}")
        state.log(error_msg, "error")
        return (
            None,
            f"ERROR: {error_msg}",
            state.get_model_status_text(),
            state.log(error_msg, "error"),
        )

    # Check SAE model
    print(f"[DEBUG] state.sae_model is None: {state.sae_model is None}")
    if state.sae_model is None:
        error_msg = "SAE model not loaded. Please load SAE model first."
        print(f"[DEBUG] ERROR: {error_msg}")
        state.log(error_msg, "error")
        return (
            None,
            f"ERROR: {error_msg}",
            state.get_model_status_text(),
            state.log(error_msg, "error"),
        )

    # Check SAE stats
    print(f"[DEBUG] state.sae_stats is None: {state.sae_stats is None}")
    if state.sae_stats is not None:
        print(f"[DEBUG] state.sae_stats keys (first 10): {list(state.sae_stats.keys())[:10]}")
    if state.sae_stats is None:
        error_msg = "SAE statistics not loaded. Please reload SAE model."
        print(f"[DEBUG] ERROR: {error_msg}")
        state.log(error_msg, "error")
        return (
            None,
            f"ERROR: {error_msg}",
            state.get_model_status_text(),
            state.log(error_msg, "error"),
        )

    # Check concept_configs
    print(f"[DEBUG] concept_configs empty: {not concept_configs}")
    if not concept_configs:
        error_msg = "No concepts selected for intervention. Please select at least one concept."
        print(f"[DEBUG] WARNING: {error_msg}")
        state.log(error_msg, "warning")
        return (
            None,
            f"WARNING: {error_msg}",
            state.get_model_status_text(),
            state.log(error_msg, "warning"),
        )

    try:
        state.system_state = SystemState.GENERATING
        print("[DEBUG] SystemState set to GENERATING")

        # Progress tracking
        if progress is not None:
            progress(0.1, desc="Setting up intervention...")

        # Convert concept IDs to concept names and validate
        valid_concepts = []
        print("[DEBUG] Converting concept IDs to concept names...")
        for config in concept_configs:
            concept_id = config["id"]
            strength = config.get("strength", 1.0)
            neurons = config.get("neurons", 16)
            # Convert underscore to space: "exposed_breast" -> "exposed breast"
            concept_name = concept_id.replace("_", " ")
            print(f"[DEBUG]   Checking concept: '{concept_id}' -> '{concept_name}'")
            if concept_name in state.sae_stats:
                valid_concepts.append(
                    {
                        "name": concept_name,
                        "strength": strength,
                        "neurons": neurons,
                    }
                )
                print(f"[DEBUG]   FOUND in stats: {concept_name}")
                state.log(f"Concept: {concept_name} (str={strength}, n={neurons})", "info")
            else:
                print(f"[DEBUG]   NOT FOUND in stats: {concept_name}")
                state.log(f"Warning: Concept '{concept_name}' not found in stats", "warning")

        print(f"[DEBUG] Valid concepts: {len(valid_concepts)}")

        if not valid_concepts:
            error_msg = "No valid concepts found in statistics. Check concept configuration."
            print(f"[DEBUG] ERROR: {error_msg}")
            state.log(error_msg, "error")
            return (
                None,
                f"ERROR: {error_msg}",
                state.get_model_status_text(),
                state.log(error_msg, "error"),
            )

        # Determine device
        device = "cuda" if (torch.cuda.is_available() and CUDA_COMPATIBLE) else "cpu"
        print(f"[DEBUG] Using device: {device}")

        # Create RepresentationModifier
        print("[DEBUG] Creating RepresentationModifier...")
        modifier = RepresentationModifier(
            sae=state.sae_model,
            stats_dict=state.sae_stats,
            epsilon=1e-8,
            device=device,
            max_concepts_number=32,
        )
        print("[DEBUG] RepresentationModifier created successfully")

        # Add each concept to unlearn with its own parameters
        print("[DEBUG] Adding concepts to unlearn...")
        for concept in valid_concepts:
            concept_name = concept["name"]
            strength = concept["strength"]
            neurons = concept["neurons"]
            print(f"[DEBUG]   Adding: {concept_name} (strength={strength}, neurons={neurons})")
            modifier.add_concept_to_unlearn(
                concept_name=concept_name,
                influence_factor=strength,
                features_number=neurons,
                per_timestep=True,
            )
            state.log(
                f"Added concept: {concept_name} (str={strength}, neurons={neurons})",
                "info",
            )

        # Get selected layer
        print(f"[DEBUG] Getting layer path for: {layer_name}")
        try:
            layer_path = LayerPath[layer_name]
            print(f"[DEBUG] Layer path found: {layer_path.name} = {layer_path.value}")
            state.log(f"Using layer: {layer_path.name}", "info")
        except (KeyError, AttributeError) as e:
            print(f"[DEBUG] Layer not found ({e}), using default")
            layer_path = LayerPath.UNET_UP_1_ATT_1
            state.log(f"Layer not found, using default: {layer_path.name}", "warning")

        # Reset timestep counter before generation (IMPORTANT!)
        print("[DEBUG] Resetting timestep counter...")
        modifier.reset_timestep()

        # Attach modifier to pipeline
        print(f"[DEBUG] Attaching modifier to pipeline at layer: {layer_path.value}")
        modifier.attach_to(state.sd_pipe, layer_path)
        print("[DEBUG] Modifier attached successfully")
        state.log(f"Modifier attached to layer: {layer_path.name} ({layer_path.value})", "info")

        if progress is not None:
            progress(0.2, desc="Generating with intervention...")

        # Progress callback for step-by-step updates
        def progress_callback(step, timestep, latents):
            if progress is not None:
                progress((step, steps), desc=f"Intervention... {step}/{steps}")

        # Generate with intervention using context manager
        print(f"[DEBUG] Creating generator with seed: {seed}")
        generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
        start_time = time.time()

        print("[DEBUG] Starting pipeline generation with modifier...")
        state.log("Starting pipeline generation with modifier...", "info")

        with modifier:
            print("[DEBUG] Inside modifier context manager")
            state.log("Inside modifier context, calling pipeline...", "info")
            try:
                print("[DEBUG] Calling sd_pipe with callback...")
                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1,
                )
                print("[DEBUG] Pipeline call completed successfully (with callback)")
                state.log("Pipeline generation completed inside context", "info")
            except TypeError as e:
                print(f"[DEBUG] Callback error: {e}")
                print("[DEBUG] Retrying without callback...")
                state.log(f"Callback error, retrying without callback: {e}", "warning")
                # Fallback without callback
                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                )
                print("[DEBUG] Pipeline call completed successfully (no callback)")
                state.log("Pipeline generation completed (no callback)", "info")

        generation_time = time.time() - start_time
        print(f"[DEBUG] Generation time: {generation_time:.2f}s")

        image = output.images[0]
        print(
            f"[DEBUG] Image generated: {type(image)}, size: {image.size if hasattr(image, 'size') else 'N/A'}"
        )

        # Detach modifier
        print("[DEBUG] Detaching modifier...")
        modifier.detach()
        print("[DEBUG] Modifier detached")
        state.log("Modifier detached", "info")

        if progress is not None:
            progress(1.0, desc="Complete!")

        state.system_state = SystemState.IDLE
        print("[DEBUG] SystemState set to IDLE")
        state.log(f"Generation with intervention complete in {generation_time:.2f}s", "success")

        # Update displays
        concept_count = len(valid_concepts)
        status_msg = (
            f"INTERVENTION COMPLETE - {generation_time:.2f}s | "
            f"Seed: {seed} | Concepts: {concept_count}"
        )
        print(f"[DEBUG] SUCCESS: {status_msg}")
        print("=" * 80 + "\n")

        return (
            image,
            status_msg,
            state.get_model_status_text(),
            state.log(f"Image generated with intervention (seed: {seed})", "success"),
        )

    except Exception as e:
        import traceback

        print(f"[DEBUG] EXCEPTION in generate_with_intervention: {e}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()

        state.system_state = SystemState.ERROR
        error_msg = f"Generation with intervention failed: {str(e)}"
        state.log(error_msg, "error")
        return (
            None,
            f"ERROR: {str(e)}",
            state.get_model_status_text(),
            state.log(error_msg, "error"),
        )


def generate_comparison_images(
    prompt: str,
    steps: int,
    guidance: float,
    seed: int,
    concept_configs: List[dict],
    layer_name: str = "UNET_UP_1_ATT_1",
    progress=None,
) -> Tuple[Any, Any, str, str]:
    """
    Generate both original and intervention images for comparison.

    This function generates two images sequentially with the SAME seed:
    1. Original image (without SAE intervention)
    2. Intervention image (with SAE concept suppression)

    NOTE: Parallel generation is NOT possible because the current hook-based
    implementation modifies all images in a batch uniformly. Sequential
    generation with the same seed ensures fair comparison.

    Args:
        prompt: Text prompt for generation
        steps: Number of diffusion steps
        guidance: Guidance scale
        seed: Random seed (same for both images)
        concept_configs: List of concept configurations, each with id, strength, neurons
        layer_name: Layer to apply intervention on
        progress: Gradio progress tracker

    Returns:
        Tuple of (original_image, intervention_image, status_text, log_text)
    """
    print("\n" + "#" * 80)
    print("[DEBUG] generate_comparison_images() CALLED")
    print("#" * 80)
    print(f"[DEBUG] prompt: {prompt[:50]}...")
    print(f"[DEBUG] concept_configs: {concept_configs}")
    print(f"[DEBUG] layer_name: {layer_name}")

    state.log("Starting comparison generation (original + intervention)...", "info")

    # Resolve random seed ONCE before generating either image
    # This ensures both images use the exact same seed
    if seed == -1:
        import random

        seed = random.randint(0, 2**32 - 1)  # noqa: S311
        print(f"[DEBUG] Resolved random seed to: {seed}")

    original_image = None
    intervention_image = None
    total_time = 0.0

    # =========================================================================
    # STEP 1: Generate ORIGINAL image (without SAE)
    # =========================================================================
    print("[DEBUG] STEP 1: Generating ORIGINAL image...")
    if progress is not None:
        progress(0.0, desc="Generating original image...")

    state.log(f"Generating original image (seed={seed})...", "info")
    start_time = time.time()

    result_original = generate_image(prompt, steps, guidance, seed, progress=None)
    original_image = result_original[0] if result_original else None
    original_time = time.time() - start_time

    print(f"[DEBUG] Original image result: {original_image is not None}")
    if original_image is None:
        error_msg = "Failed to generate original image"
        print(f"[DEBUG] ERROR: {error_msg}")
        state.log(error_msg, "error")
        return (None, None, f"ERROR: {error_msg}", state.log(error_msg, "error"))

    print(f"[DEBUG] Original image generated in {original_time:.2f}s")
    state.log(f"Original image generated in {original_time:.2f}s", "success")
    total_time += original_time

    # =========================================================================
    # STEP 2: Generate INTERVENTION image (with SAE)
    # =========================================================================
    print("[DEBUG] STEP 2: Generating INTERVENTION image...")
    if progress is not None:
        progress(0.5, desc="Generating intervention image...")

    state.log(f"Generating intervention image (seed={seed}, same as original)...", "info")
    start_time = time.time()

    print("[DEBUG] Calling generate_with_intervention()...")
    result_intervention = generate_with_intervention(
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        seed=seed,  # SAME seed for fair comparison
        concept_configs=concept_configs,
        layer_name=layer_name,
        progress=None,
    )
    intervention_image = result_intervention[0] if result_intervention else None
    intervention_time = time.time() - start_time

    print(f"[DEBUG] Intervention image result: {intervention_image is not None}")
    if intervention_image is None:
        # Still return original even if intervention failed
        error_msg = "Intervention generation failed, returning original only"
        print(f"[DEBUG] WARNING: {error_msg}")
        state.log(error_msg, "warning")
        return (
            original_image,
            None,
            f"WARNING: {error_msg}",
            state.log(error_msg, "warning"),
        )

    print(f"[DEBUG] Intervention image generated in {intervention_time:.2f}s")
    state.log(f"Intervention image generated in {intervention_time:.2f}s", "success")
    total_time += intervention_time

    # =========================================================================
    # STEP 3: Return comparison results
    # =========================================================================
    print("[DEBUG] STEP 3: Returning comparison results")
    if progress is not None:
        progress(1.0, desc="Comparison complete!")

    status_msg = (
        f"COMPARISON COMPLETE - Total: {total_time:.2f}s | "
        f"Original: {original_time:.2f}s | Intervention: {intervention_time:.2f}s | "
        f"Seed: {seed}"
    )
    print(f"[DEBUG] {status_msg}")
    print("#" * 80 + "\n")

    state.log(
        f"Comparison generation complete: {len(concept_configs)} concepts suppressed", "success"
    )

    return (
        original_image,
        intervention_image,
        status_msg,
        state.log(f"Both images generated for comparison (seed: {seed})", "success"),
    )


# CSS Hot Reload Support
def reload_css():
    """Reload CSS from file and return as injectable HTML"""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        log_msg = state.log("CSS reloaded successfully", "success")
        css_html = f"<style>{css_content}</style>"
        return css_html, log_msg
    else:
        log_msg = state.log("CSS file not found", "error")
        return "", log_msg


# Model loading functions - wrappers around core.model_loader
def load_sd_model(use_gpu=True):
    """Load Stable Diffusion v1.5 model (wrapper for core.model_loader)."""
    return _load_sd_model(state, use_gpu=use_gpu)


def load_sae_model(sae_model_id: str):
    """Load SAE model with weights and feature sums from config (wrapper for core.model_loader)."""
    if not SAE_AVAILABLE:
        state.log(f"SAE libraries not available: {SAE_IMPORT_ERROR}", "error")
        raise ImportError("SAE dependencies not installed")

    if not SAE_CONFIG_LOADED or not SAE_CONFIG:
        state.log("SAE config not loaded", "error")
        raise RuntimeError("SAE configuration not loaded")

    return _load_sae_model(
        state,
        sae_model_id,
        SAE_CONFIG,
        get_sae_hyperparameters,
        get_feature_sums_path,
        get_model_path,
    )


def detect_content(image: Image.Image) -> tuple[dict, Image.Image | None]:
    """Wrapper for detect_content from utils.detection that uses global state."""
    return _detect_content(image, state)


def load_nudenet_model():
    """Load NudeNet detector (wrapper for core.model_loader)."""
    if not NUDENET_AVAILABLE:
        error_msg = f"NudeNet not available: {NUDENET_IMPORT_ERROR}"
        state.log(error_msg, "error")
        raise ImportError(error_msg)
    return _load_nudenet_model(state)


# Dashboard UI
def create_dashboard():
    """Create the main dashboard interface"""

    with gr.Blocks(title="SD Control") as app:
        # 1. HEADER - Title and Description
        gr.HTML("""
<div style="margin-bottom: 3rem; text-align: center;">
    <h1 style="font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.5rem; background: linear-gradient(135deg, #00d4ff, #b24bf3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        Stable Diffusion Control Dashboard
    </h1>
    <p style="font-family: 'Inter', sans-serif; font-size: 1rem; color: #8fa3c4; max-width: 800px; margin: 0 auto;">
        Interactive dashboard for controlling Stable Diffusion's internal representations through Sparse Autoencoders.
        Load a base model, apply concept interventions, and visualize the results.
    </p>
</div>
""")

        # 2. BASE IMAGE GENERATION SECTION
        with gr.Accordion(
            "Base Image Generation",
            open=True,
            elem_classes=["main-section", "section-not-loaded"],
            elem_id="section-base-generation",
        ) as section_base_gen:
            with gr.Row(elem_classes=["section-content", "columns-gap-large"]):
                # Base Model Selection
                with gr.Column(scale=1, elem_classes=["column-base-model"]):
                    gr.Markdown("### Base Model")
                    with gr.Column(elem_classes=["form-group-consistent"]):
                        sd_model_dropdown = gr.Dropdown(
                            choices=["sd-legacy/stable-diffusion-v1-5"],
                            value="sd-legacy/stable-diffusion-v1-5",
                            label="Model",
                            info="Stable Diffusion model",
                        )
                        use_gpu_checkbox = gr.Checkbox(
                            label="Use GPU" + ("" if CUDA_COMPATIBLE else " - Not Available"),
                            value=CUDA_COMPATIBLE,
                            interactive=CUDA_COMPATIBLE,
                            info=CUDA_STATUS,
                        )
                        load_sd_btn = gr.Button("Load Model", variant="primary", size="sm")

                # Generation Parameters
                with gr.Column(scale=1, elem_classes=["column-gen-params"]):
                    gr.Markdown("### Generation Parameters")
                    with gr.Column(elem_classes=["form-group-consistent"]):
                        steps_input = gr.Slider(
                            minimum=10, maximum=100, value=50, step=1, label="Denoising Steps"
                        )
                        guidance_input = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                        )
                        seed_input = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

        # 3. INTERVENTION SECTION
        with gr.Accordion(
            "Concept Intervention",
            open=True,
            elem_classes=["main-section", "section-not-loaded"],
            elem_id="section-intervention",
        ) as section_intervention:
            with gr.Column(elem_classes=["section-content", "intervention-single-column"]):
                # Row 1: SAE Model Selection
                gr.Markdown("### SAE Model")

                # Get SAE model choices from config
                sae_choices = []
                if SAE_CONFIG_LOADED and SAE_CONFIG:
                    sae_choices = get_sae_model_choices(SAE_CONFIG)

                sae_model_dropdown = gr.Dropdown(
                    choices=[("None", "none")] + sae_choices,
                    value="none",
                    label="Select SAE Model",
                    interactive=True,
                    info="Choose a trained SAE model for concept intervention",
                )
                load_sae_btn = gr.Button("Load SAE", size="sm", variant="primary")

                # Hidden checkbox to maintain state compatibility
                enable_intervention = gr.Checkbox(
                    label="Enable Concept Intervention",
                    value=False,
                    visible=False,
                )

                # Row 3: Concept Selection & Strength (visible after layer selected)
                with gr.Column(
                    elem_classes=["concept-selection-section"], visible=False
                ) as concept_section:
                    gr.Markdown("### Concept Selection & Strength")
                    gr.Markdown(
                        "*Select concepts to suppress. Each concept can have its own strength and neuron count.*",
                        elem_classes=["concept-help-text"],
                    )

                    # Create individual concept rows (max 20 concepts)
                    MAX_CONCEPTS = 20
                    concept_components = {
                        "checkboxes": [],
                        "strengths": [],
                        "neurons": [],
                        "rows": [],
                    }

                    for i in range(MAX_CONCEPTS):
                        with gr.Row(
                            visible=False,
                            elem_classes=["concept-row"],
                        ) as concept_row:
                            concept_cb = gr.Checkbox(
                                label=f"Concept {i + 1}",
                                value=False,
                                scale=2,
                                elem_classes=["concept-checkbox"],
                            )
                            concept_strength = gr.Slider(
                                minimum=-10.0,
                                maximum=10.0,
                                value=1.0,
                                step=0.1,
                                label="Strength",
                                scale=2,
                                interactive=False,
                                elem_classes=["concept-slider"],
                            )
                            concept_neurons = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=16,
                                step=1,
                                label="Neurons",
                                scale=2,
                                interactive=False,
                                elem_classes=["concept-slider"],
                            )

                        concept_components["rows"].append(concept_row)
                        concept_components["checkboxes"].append(concept_cb)
                        concept_components["strengths"].append(concept_strength)
                        concept_components["neurons"].append(concept_neurons)

                        # Wire up checkbox to enable/disable sliders
                        def make_toggle_handler(strength_slider, neurons_slider):
                            def toggle_sliders(checked):
                                return (
                                    gr.update(interactive=checked),
                                    gr.update(interactive=checked),
                                )

                            return toggle_sliders

                        concept_cb.change(
                            fn=make_toggle_handler(concept_strength, concept_neurons),
                            inputs=[concept_cb],
                            outputs=[concept_strength, concept_neurons],
                        )

                    # Store concept metadata (populated when SAE loads)
                    concept_metadata = gr.State(value=[])  # List of (id, name) tuples

        # 4. IMAGE GENERATION
        with gr.Accordion(
            "Image Generation",
            open=True,
            elem_classes=["main-section"],
            elem_id="section-results",
        ):
            with gr.Column(elem_classes=["section-content"]):
                # Prompt and NudeNet checkbox row
                with gr.Row(elem_classes=["prompt-generate-row"]):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="a beautiful landscape with mountains and sunset",
                        lines=2,
                        scale=4,
                    )
                    nudenet_checkbox = gr.Checkbox(
                        label="Enable NudeNet Detection",
                        value=True,
                        info="Run safety detection on generated images",
                        scale=1,
                    )
                # Generate button row
                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg",
                    interactive=False,  # Disabled until both models are loaded
                    elem_id="btn-generate",
                )

                with gr.Row(elem_classes=["results-images-row"]):
                    with gr.Column(elem_classes=["result-image-container"]):
                        img_original = gr.Image(
                            label="Original Output",
                            type="pil",
                            interactive=False,
                            show_label=True,
                            elem_classes=["result-image-bordered"],
                        )
                    with gr.Column(elem_classes=["result-image-container"]):
                        img_unlearned = gr.Image(
                            label="With Intervention",
                            type="pil",
                            interactive=False,
                            show_label=True,
                            elem_classes=["result-image-bordered"],
                        )

                # NudeNet Scores Comparison Table (below images)
                nudenet_scores_comparison = gr.Markdown(
                    value="",
                    elem_classes=["nudenet-scores-comparison"],
                )

        # Event handlers

        def handle_sae_model_change(sae_model_id):
            """Handle SAE model selection - SAE needs to be loaded."""
            # When SAE selection changes, SAE needs to be loaded again - disable generate
            if sae_model_id == "none" or not SAE_CONFIG_LOADED or not SAE_CONFIG:
                return {
                    concept_section: gr.update(visible=False),
                    enable_intervention: gr.update(value=False),
                    section_intervention: gr.update(
                        elem_classes=["main-section", "section-not-loaded"]
                    ),
                    generate_btn: gr.update(interactive=False),
                }

            # SAE selected but not loaded yet
            return {
                concept_section: gr.update(visible=False),
                enable_intervention: gr.update(value=False),
                section_intervention: gr.update(
                    elem_classes=["main-section", "section-not-loaded"]
                ),
                generate_btn: gr.update(interactive=False),
            }

        def toggle_intervention(enabled):
            """Enable/disable intervention controls"""
            return {
                sae_model_dropdown: gr.update(interactive=True),
                load_sae_btn: gr.update(interactive=enabled),
            }

        def handle_load_sd(model_name, use_gpu):
            """Load SD model with UI state updates"""
            # First, indicate loading state
            device_type = "GPU" if (use_gpu and CUDA_COMPATIBLE) else "CPU"
            gr.Info(
                f"Loading Stable Diffusion v1.5 on {device_type}... This may take a moment.",
                duration=5,
            )

            yield (
                gr.update(interactive=False),  # Disable load button during loading
                gr.update(interactive=False),  # Keep generate disabled
                gr.update(elem_classes=["main-section", "section-loading"]),  # Blue pulsing
            )

            try:
                state.set_model_state("sd_base", ModelLoadState.LOADING)

                start_time = time.time()
                load_sd_model(use_gpu=use_gpu)
                load_time = time.time() - start_time
                state.set_model_state("sd_base", ModelLoadState.LOADED, load_time)

                state.log(f"SD v1.5 loaded on {device_type} in {load_time:.1f}s", "success")
                gr.Info(
                    f"✓ Stable Diffusion v1.5 loaded successfully on {device_type} ({load_time:.1f}s)",
                    duration=15,
                )
                # Enable generate only if SAE is also loaded
                sae_loaded = state.model_states["sae"] == ModelLoadState.LOADED
                yield (
                    gr.update(interactive=True),  # Re-enable load button
                    gr.update(interactive=sae_loaded),  # Enable generate only if SAE loaded
                    gr.update(elem_classes=["main-section", "section-loaded"]),  # Green
                )

            except Exception as e:
                state.set_model_state("sd_base", ModelLoadState.FAILED)
                state.system_state = SystemState.ERROR
                state.log(f"SD loading failed: {str(e)}", "error")
                gr.Warning(f"Failed to load Stable Diffusion: {str(e)}", duration=15)
                yield (
                    gr.update(interactive=True),  # Re-enable load button to retry
                    gr.update(interactive=False),  # Keep generate disabled
                    gr.update(elem_classes=["main-section", "section-not-loaded"]),  # Yellow
                )

        def handle_load_sae(sae_model_id):
            """Load SAE model and show concepts on success"""
            MAX_CONCEPTS = 20

            # Build empty result for all concept components
            # Order: rows (20), checkboxes (20), strengths (20), neurons (20)
            def build_empty_result():
                result = [
                    gr.update(visible=False),  # concept_section
                    gr.update(value=False),  # enable_intervention
                    gr.update(elem_classes=["main-section", "section-not-loaded"]),
                    gr.update(interactive=False),  # generate_btn
                    [],  # concept_metadata
                ]
                # Add updates for rows
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(visible=False))
                # Add updates for checkboxes
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(label="Concept", value=False))
                # Add updates for strengths
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=1.0, interactive=False))
                # Add updates for neurons
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=16, interactive=False))
                return tuple(result)

            if sae_model_id == "none":
                yield build_empty_result()
                return

            # Indicate loading state
            loading_result = [
                gr.update(visible=False),
                gr.update(value=False),
                gr.update(elem_classes=["main-section", "section-loading"]),
                gr.update(interactive=False),
                [],
            ]
            # Rows
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update(visible=False))
            # Checkboxes
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            # Strengths
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            # Neurons
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            yield tuple(loading_result)

            try:
                state.set_model_state("sae", ModelLoadState.LOADING)

                # Get SAE model info from config
                sae_name = sae_model_id
                layer_id = ""
                if SAE_CONFIG_LOADED and SAE_CONFIG:
                    sae_model_config = SAE_CONFIG.get_sae_model(sae_model_id)
                    if sae_model_config:
                        sae_name = sae_model_config.name
                        layer_id = sae_model_config.layer_id

                gr.Info(
                    f"Loading SAE model: {sae_name} ({layer_id})... This may take a moment.",
                    duration=5,
                )
                state.log(f"Loading {sae_name} ({layer_id})...", "loading")

                start_time = time.time()
                load_sae_model(sae_model_id)
                load_time = time.time() - start_time
                state.set_model_state("sae", ModelLoadState.LOADED, load_time)

                state.log(f"SAE loaded in {load_time:.1f}s", "success")
                gr.Info(f"✓ SAE model loaded successfully ({load_time:.1f}s)", duration=15)

                # Get concepts for this SAE model from config
                concept_choices = []
                if SAE_CONFIG_LOADED and SAE_CONFIG:
                    concept_choices = get_concept_choices(SAE_CONFIG, sae_model_id)

                # Enable generate only if SD base is also loaded
                sd_loaded = state.model_states["sd_base"] == ModelLoadState.LOADED

                # Build success result with concept components
                # Order: rows (20), checkboxes (20), strengths (20), neurons (20)
                result = [
                    gr.update(visible=True),  # concept_section
                    gr.update(value=True),  # enable_intervention
                    gr.update(elem_classes=["main-section", "section-loaded"]),
                    gr.update(interactive=sd_loaded),  # generate_btn
                    concept_choices,  # concept_metadata
                ]

                # Rows
                for i in range(MAX_CONCEPTS):
                    if i < len(concept_choices):
                        result.append(gr.update(visible=True))
                    else:
                        result.append(gr.update(visible=False))

                # Checkboxes
                for i in range(MAX_CONCEPTS):
                    if i < len(concept_choices):
                        name, concept_id, description = concept_choices[i]
                        result.append(gr.update(label=name, value=False, info=description))
                    else:
                        result.append(gr.update(label="Concept", value=False, info=None))

                # Strengths
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=1.0, interactive=False))

                # Neurons
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=16, interactive=False))

                yield tuple(result)

            except Exception as e:
                state.set_model_state("sae", ModelLoadState.FAILED)
                state.log(f"SAE loading failed: {str(e)}", "error")
                gr.Warning(f"Failed to load SAE model: {str(e)}", duration=15)
                yield build_empty_result()

        # SAE model selection
        sae_model_dropdown.change(
            fn=handle_sae_model_change,
            inputs=[sae_model_dropdown],
            outputs=[
                concept_section,
                enable_intervention,
                section_intervention,
                generate_btn,
            ],
        )

        # Intervention toggle
        enable_intervention.change(
            fn=toggle_intervention,
            inputs=[enable_intervention],
            outputs=[
                sae_model_dropdown,
                load_sae_btn,
            ],
        )

        # Generate button - handles both normal and intervention generation
        def handle_generate(
            prompt,
            steps,
            guidance,
            seed,
            intervention_enabled,
            sae_model_id,
            nudenet_enabled,
            concept_meta,  # List of (name, id) tuples
            *concept_values,  # All checkbox, strength, neurons values flattened
            progress=gr.Progress(),
        ):
            """
            Generate original and intervened images with progress tracking.

            IMPORTANT: Parallel generation of both images is NOT possible because
            the hook-based SAE intervention modifies ALL images in a batch uniformly.
            We generate sequentially with the SAME seed for fair comparison.

            Flow:
            1. Generate original image (without SAE)
            2. If intervention enabled: Generate intervention image (with SAE, same seed)
            3. Run NudeNet detection on both (if enabled)
            4. Return both images for side-by-side comparison
            """
            print("\n" + "*" * 80)
            print("[DEBUG] handle_generate() CALLED")
            print("*" * 80)
            print(f"[DEBUG] prompt: {prompt[:50] if prompt else 'None'}...")
            print(f"[DEBUG] steps: {steps}, guidance: {guidance}, seed: {seed}")
            print(f"[DEBUG] intervention_enabled: {intervention_enabled}")
            print(f"[DEBUG] sae_model_id: {sae_model_id}")
            print(f"[DEBUG] nudenet_enabled: {nudenet_enabled}")
            print(f"[DEBUG] concept_meta: {concept_meta}")
            print(f"[DEBUG] concept_values count: {len(concept_values)}")

            # Parse concept values - they come in groups of 3 (checkbox, strength, neurons)
            MAX_CONCEPTS = 20
            concept_configs = []
            if concept_meta:
                for i in range(min(len(concept_meta), MAX_CONCEPTS)):
                    idx = i * 3
                    if idx + 2 < len(concept_values):
                        is_selected = concept_values[idx]
                        strength = concept_values[idx + 1]
                        neurons = concept_values[idx + 2]
                        if is_selected:
                            name, concept_id, _description = concept_meta[i]
                            concept_configs.append(
                                {
                                    "id": concept_id,
                                    "strength": strength,
                                    "neurons": int(neurons),
                                }
                            )

            print(f"[DEBUG] Built concept_configs: {concept_configs}")
            print(f"[DEBUG] state.sd_pipe is None: {state.sd_pipe is None}")
            print(f"[DEBUG] state.sae_model is None: {state.sae_model is None}")
            print(f"[DEBUG] state.sae_stats is None: {state.sae_stats is None}")

            # Check if model is loaded
            if state.sd_pipe is None:
                print("[DEBUG] ERROR: SD pipe not loaded, returning None")
                yield None, None, ""
                return

            # Resolve random seed ONCE before generating either image
            if seed == -1:
                import random

                seed = random.randint(0, 2**32 - 1)  # noqa: S311
                print(f"[DEBUG] Resolved random seed to: {seed}")

            total_steps = int(steps)
            display_original = None
            display_intervened = None

            # =====================================================================
            # Use generate_comparison_images when intervention is enabled
            # =====================================================================
            print("[DEBUG] Checking intervention conditions...")
            print(f"[DEBUG]   intervention_enabled: {intervention_enabled}")
            print(f"[DEBUG]   state.sae_model is not None: {state.sae_model is not None}")
            print(f"[DEBUG]   concept_configs (truthy): {bool(concept_configs)}")

            if intervention_enabled and state.sae_model is not None and concept_configs:
                print("[DEBUG] INTERVENTION MODE: Generating comparison images")
                state.log("Using comparison mode: generating both images sequentially", "info")

                # Get layer ID from config
                layer = (
                    get_layer_id(SAE_CONFIG, sae_model_id)
                    if SAE_CONFIG_LOADED and SAE_CONFIG
                    else "UNET_UP_1_ATT_1"
                )
                print(f"[DEBUG] Using layer: {layer}")

                # Total steps for both images combined
                combined_total = total_steps * 2

                # Progress callback for original image (steps 1 to total_steps)
                def original_callback(pipe, step, timestep, callback_kwargs):
                    current = step + 1  # step is 0-indexed
                    if progress is not None:
                        progress(
                            (current, combined_total),
                            desc=f"🖼️ Original: Step {current}/{total_steps}",
                        )
                    return callback_kwargs

                state.log(f"Generating original image (seed={seed})...", "info")
                state.system_state = SystemState.GENERATING
                state.start_generation(total_steps=total_steps, phase="original")

                generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                start_time = time.time()

                # Generate original image with callback
                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=total_steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=original_callback,
                )

                original_image = output.images[0]
                original_time = time.time() - start_time
                state.log(f"Original image generated in {original_time:.2f}s", "success")

                # Progress callback for intervention image (steps total_steps+1 to combined_total)
                def intervention_callback(pipe, step, timestep, callback_kwargs):
                    current = total_steps + step + 1  # Offset by original steps
                    if progress is not None:
                        progress(
                            (current, combined_total),
                            desc=f"🔧 Intervention: Step {step + 1}/{total_steps}",
                        )
                    return callback_kwargs

                state.log(f"Generating intervention image (seed={seed})...", "info")

                # Generate with intervention
                try:
                    device = "cuda" if (torch.cuda.is_available() and CUDA_COMPATIBLE) else "cpu"

                    valid_concepts = []
                    for config in concept_configs:
                        concept_id = config["id"]
                        strength = config.get("strength", 1.0)
                        neurons = config.get("neurons", 16)
                        concept_name = concept_id.replace("_", " ")
                        if concept_name in state.sae_stats:
                            valid_concepts.append(
                                {
                                    "name": concept_name,
                                    "strength": strength,
                                    "neurons": neurons,
                                }
                            )

                    if valid_concepts:
                        modifier = RepresentationModifier(
                            sae=state.sae_model,
                            stats_dict=state.sae_stats,
                            epsilon=1e-8,
                            device=device,
                            max_concepts_number=32,
                        )

                        for concept in valid_concepts:
                            modifier.add_concept_to_unlearn(
                                concept_name=concept["name"],
                                influence_factor=concept["strength"],
                                features_number=concept["neurons"],
                                per_timestep=True,
                            )

                        layer_path = LayerPath[layer] if layer else LayerPath.UNET_UP_1_ATT_1
                        modifier.reset_timestep()
                        modifier.attach_to(state.sd_pipe, layer_path)

                        generator2 = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                        start_time2 = time.time()

                        with modifier:
                            output2 = state.sd_pipe(
                                prompt=prompt,
                                num_inference_steps=total_steps,
                                guidance_scale=guidance,
                                generator=generator2,
                                callback_on_step_end=intervention_callback,
                            )

                        modifier.detach()
                        intervened_image = output2.images[0]
                        intervention_time = time.time() - start_time2
                        state.log(
                            f"Intervention image generated in {intervention_time:.2f}s", "success"
                        )
                    else:
                        intervened_image = None
                        state.log("No valid concepts found", "warning")

                except Exception as e:
                    print(f"[DEBUG] Intervention error: {e}")
                    intervened_image = None
                    state.log(f"Intervention failed: {str(e)}", "error")

                # Show completion
                if progress is not None:
                    progress((combined_total, combined_total), desc="✅ Both images complete!")

                display_original = original_image
                display_intervened = intervened_image

                # Run NudeNet detection on both images
                detection_orig = None
                detection_interv = None

                if original_image is not None and state.nudenet_detector is not None:
                    detection_orig, censored_orig = detect_content(original_image)
                    if (
                        nudenet_enabled
                        and censored_orig is not None
                        and detection_orig.get("has_unsafe", False)
                    ):
                        display_original = censored_orig
                        state.log("Displaying censored version of original image", "info")

                if intervened_image is not None and state.nudenet_detector is not None:
                    detection_interv, censored_interv = detect_content(intervened_image)
                    if (
                        nudenet_enabled
                        and censored_interv is not None
                        and detection_interv.get("has_unsafe", False)
                    ):
                        display_intervened = censored_interv
                        state.log("Displaying censored version of intervened image", "info")

                # Format combined comparison table
                scores_comparison = format_nudenet_comparison(detection_orig, detection_interv)

                state.system_state = SystemState.IDLE

                # Final yield with both images and combined scores
                print("[DEBUG] Yielding both images and scores for display")
                print("*" * 80 + "\n")
                yield display_original, display_intervened, scores_comparison

            else:
                # =====================================================================
                # No intervention - generate original only
                # =====================================================================
                print("[DEBUG] NO INTERVENTION MODE: Generating original only")
                if not intervention_enabled:
                    print("[DEBUG]   Reason: intervention_enabled is False")
                if state.sae_model is None:
                    print("[DEBUG]   Reason: state.sae_model is None")
                if not concept_configs:
                    print("[DEBUG]   Reason: no concepts selected")

                state.log("Generating original image only (no intervention)", "info")

                # Progress callback for solo generation
                def solo_callback(pipe, step, timestep, callback_kwargs):
                    current = step + 1
                    if progress is not None:
                        progress(
                            (current, total_steps),
                            desc=f"🖼️ Generating: Step {current}/{total_steps}",
                        )
                    return callback_kwargs

                state.system_state = SystemState.GENERATING
                state.start_generation(total_steps=total_steps, phase="original")

                generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                start_time = time.time()

                # Generate with callback
                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=total_steps,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=solo_callback,
                )

                original_image = output.images[0]
                generation_time = time.time() - start_time
                state.log(f"Original image generated in {generation_time:.2f}s", "success")

                # Show completion
                if progress is not None:
                    progress(
                        (total_steps, total_steps), desc=f"✅ Complete ({generation_time:.1f}s)"
                    )

                display_original = original_image
                print(f"[DEBUG] Original image generated: {original_image is not None}")

                # Run NudeNet detection on original
                detection_orig = None
                if original_image is not None and state.nudenet_detector is not None:
                    detection_orig, censored_orig = detect_content(original_image)
                    if (
                        nudenet_enabled
                        and censored_orig is not None
                        and detection_orig.get("has_unsafe", False)
                    ):
                        display_original = censored_orig
                        state.log("Displaying censored version of original image", "info")

                # Format comparison table (intervention column will show dashes)
                scores_comparison = format_nudenet_comparison(detection_orig, None)

                state.system_state = SystemState.IDLE

                # Yield original only (no intervention image)
                print("[DEBUG] Yielding original image only (no intervention)")
                print("*" * 80 + "\n")
                yield display_original, None, scores_comparison

        # Build inputs list for generate button
        generate_inputs = [
            prompt_input,
            steps_input,
            guidance_input,
            seed_input,
            enable_intervention,
            sae_model_dropdown,
            nudenet_checkbox,
            concept_metadata,
        ]
        # Add all concept components in interleaved order (checkbox, strength, neurons for each)
        for i in range(MAX_CONCEPTS):
            generate_inputs.append(concept_components["checkboxes"][i])
            generate_inputs.append(concept_components["strengths"][i])
            generate_inputs.append(concept_components["neurons"][i])

        generate_btn.click(
            fn=handle_generate,
            inputs=generate_inputs,
            outputs=[
                img_original,
                img_unlearned,
                nudenet_scores_comparison,
            ],
        )

        # Model loading buttons
        load_sd_btn.click(
            fn=handle_load_sd,
            inputs=[sd_model_dropdown, use_gpu_checkbox],
            outputs=[
                load_sd_btn,
                generate_btn,
                section_base_gen,
            ],
        )

        load_sae_btn.click(
            fn=handle_load_sae,
            inputs=[sae_model_dropdown],
            outputs=[
                concept_section,
                enable_intervention,
                section_intervention,
                generate_btn,
                concept_metadata,
            ]
            + concept_components["rows"]
            + concept_components["checkboxes"]
            + concept_components["strengths"]
            + concept_components["neurons"],
        )

    return app


# Launch
if __name__ == "__main__":
    # Auto-load NudeNet detector for scoring
    if NUDENET_AVAILABLE and state.nudenet_detector is None:
        try:
            print("[INFO] Auto-loading NudeNet detector for scoring...")
            load_nudenet_model()
            print("[OK] NudeNet detector loaded")
        except Exception as e:
            print(f"[WARNING] Could not auto-load NudeNet: {e}")

    app = create_dashboard()

    css_path = Path(__file__).parent / "style.css"
    custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        css=custom_css,
        favicon_path=Path(__file__).parent / "favicon.ico",
    )

"""Interactive dashboard for controlling Stable Diffusion via Sparse Autoencoders."""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")

import gradio as gr
import torch

_dashboard_dir = Path(__file__).parent
_project_root = _dashboard_dir.parent
if str(_dashboard_dir) not in sys.path:
    sys.path.insert(0, str(_dashboard_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    load_nudenet_model,
    load_sae_model,
    load_sd_model,
)
from core.state import (  # noqa: E402
    DashboardState,
    ModelLoadState,
    SystemState,
)
from utils.clip_score import (  # noqa: E402
    calculate_clip_scores,
    format_clip_scores,
)
from utils.cuda import (  # noqa: E402
    CUDA_COMPATIBLE,
    CUDA_STATUS,
)
from utils.detection import (  # noqa: E402
    detect_content,
    format_nudenet_comparison,
)
from utils.heatmap import (  # noqa: E402
    collect_activations_from_representations,
    decode_latent_to_image,
    generate_heatmap_gallery,
)

from src.models.sd_v1_5.hooks import capture_layer_representations  # noqa: E402
from src.models.sd_v1_5.layers import LayerPath  # noqa: E402
from src.utils.RepresentationModifier import RepresentationModifier  # noqa: E402

SAE_CONFIG = load_sae_config()
MAX_CONCEPTS = 20
DENOISING_STEPS = 50

state = DashboardState()


def create_dashboard():
    """Create the main dashboard interface"""

    with gr.Blocks(title="SD Control") as app:
        gr.HTML("""
<div style="margin-bottom: 3rem; align-items:center; display: flex;">
    <h1 style="font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; letter-spacing: 0.05em; background: linear-gradient(135deg, #5b9bd5, #c87bf3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-align: left;">
        Stable Diffusion <br/>
        Control Dashboard
    </h1>
    <p style="font-family: 'Inter', sans-serif; font-size: 1rem; color: #8fa3c4; max-width: 800px; margin-left: auto; text-align: right;">
        Interactive dashboard for controlling Stable Diffusion's internal representations through Sparse Autoencoders.
        Load a base model, apply concept interventions, and visualize the results.
    </p>
</div>
""")

        with gr.Accordion(
            "Base Model",
            open=True,
            elem_classes=["main-section", "section-not-loaded"],
            elem_id="section-base-generation",
        ) as section_base_gen:
            with gr.Row(elem_classes=["section-content", "columns-gap-large"]):
                with gr.Column(scale=1, elem_classes=["column-base-model"]):
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

        with gr.Accordion(
            "Concept Intervention",
            open=True,
            elem_classes=["main-section", "section-not-loaded"],
            elem_id="section-intervention",
        ) as section_intervention:
            with gr.Column(elem_classes=["section-content", "intervention-single-column"]):
                gr.Markdown("### SAE Model")

                sae_choices = []
                sae_choices = get_sae_model_choices(SAE_CONFIG)

                sae_model_dropdown = gr.Dropdown(
                    choices=sae_choices,
                    value=sae_choices[0][1] if sae_choices else None,
                    label="Select SAE Model",
                    interactive=True,
                    info="Choose a trained SAE model for concept intervention",
                )
                load_sae_btn = gr.Button("Load SAE", size="sm", variant="primary")

                enable_intervention = gr.Checkbox(
                    label="Enable Concept Intervention",
                    value=False,
                    visible=False,
                )

                with gr.Column(
                    elem_classes=["concept-selection-section"], visible=False
                ) as concept_section:
                    gr.Markdown("### Concept Selection")
                    gr.Markdown(
                        """Select concepts to eliminate from generated images.  
                        **Strength** controls the intensity of concept suppression (higher = stronger elimination).  
                        **Neurons** determines how many SAE features are used (higher values improve elimination but may affect other image aspects).""",  # noqa: E501, W291
                        elem_classes=["concept-help-text"],
                    )

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
                                minimum=-0.0,
                                maximum=32.0,
                                value=4.0,
                                step=0.1,
                                label="Strength",
                                scale=2,
                                interactive=False,
                                elem_classes=["concept-slider"],
                            )
                            concept_neurons = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=8,
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

                        def toggle_sliders(checked):
                            return (
                                gr.update(interactive=checked),
                                gr.update(interactive=checked),
                            )

                        concept_cb.change(
                            fn=toggle_sliders,
                            inputs=[concept_cb],
                            outputs=[concept_strength, concept_neurons],
                        )

                    concept_metadata = gr.State(value=[])

        with gr.Accordion(
            "Image Generation",
            open=True,
            elem_classes=["main-section"],
            elem_id="section-results",
        ):
            with gr.Column(elem_classes=["section-content"]):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="a beautiful landscape with mountains and sunset",
                    lines=2,
                    elem_classes=["no-status-tracker"],
                )

                with gr.Row(
                    elem_classes=["no-status-tracker"],
                ):
                    guidance_input = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    seed_input = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                with gr.Row(
                    elem_classes=["no-status-tracker"],
                ):
                    nudenet_checkbox = gr.Checkbox(
                        label="Censor Adult Content",
                        value=True,
                        info="⚠️ This model can generate adult content. Disabling this option is at your own risk.",
                    )
                    intervention_mode = gr.Radio(
                        choices=[
                            ("Per-Timestep", "per_timestep"),
                            ("Global", "global"),
                        ],
                        value="per_timestep",
                        label="Neuron Selection Mode",
                        info="Per-Timestep: different neurons each step. Global: same neurons all steps.",
                    )

                gr.Markdown(
                    """**⚠️ Content Warning:** These images are generated to analyze
                    intervention results. While NudeNet censoring is applied,
                    the content may still contain adult, disturbing, or unexpected
                    imagery. **View at your own risk.**""",
                    elem_classes=["warning-text"],
                )

                generate_btn = gr.Button(
                    "Generate Image",
                    variant="primary",
                    size="lg",
                    interactive=False,
                    elem_id="btn-generate",
                )

                generation_progress = gr.HTML(
                    value="",
                    elem_classes=["progress-display"],
                )

                with gr.Accordion(
                    "🖼️ Generated Images (⚠️ May contain adult content)",
                    open=False,
                    visible=False,
                    elem_classes=["inner-section", "results-accordion"],
                ) as images_accordion:
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

                with gr.Accordion(
                    "📊 NudeNet Detection Score",
                    open=True,
                    visible=False,
                    elem_classes=["inner-section", "analysis-accordion"],
                ) as nudenet_accordion:
                    nudenet_scores_comparison = gr.HTML(
                        value="",
                        elem_classes=["nudenet-scores-comparison"],
                    )

                with gr.Accordion(
                    "📐 CLIP Score Analysis",
                    open=True,
                    visible=False,
                    elem_classes=["inner-section", "analysis-accordion"],
                ) as clip_accordion:
                    clip_scores_display = gr.HTML(
                        value="",
                        elem_classes=["clip-scores-display"],
                    )

                with gr.Accordion(
                    "🔥 SAE Feature Heatmaps",
                    open=True,
                    visible=False,
                    elem_classes=["inner-section", "analysis-accordion"],
                ) as heatmap_accordion:
                    gr.Markdown(
                        """Visualize which spatial regions activate specific SAE features.
                        Select a concept from intervention and specify timesteps to analyze.""",
                        elem_classes=["heatmap-description"],
                    )
                    with gr.Row(elem_classes=["heatmap-controls-row"]):
                        heatmap_timesteps = gr.Textbox(
                            label="Timesteps",
                            value="",
                            placeholder="e.g., 10,25,40",
                            info="Comma-separated (0-49)",
                            interactive=True,
                            scale=2,
                        )
                        heatmap_concept_dropdown = gr.Dropdown(
                            label="Concept",
                            choices=[],
                            value=None,
                            info="From intervention",
                            interactive=True,
                            scale=2,
                        )
                        generate_heatmap_btn = gr.Button(
                            "Generate Heatmaps",
                            variant="primary",
                            size="sm",
                            interactive=False,
                            elem_id="btn-generate-heatmap",
                        )

                    heatmap_progress = gr.HTML(
                        value="",
                        elem_classes=["progress-display"],
                    )

                    heatmap_section_html = gr.HTML(
                        value="",
                        elem_classes=["heatmap-section-display"],
                    )
                    heatmap_section_gallery = gr.Gallery(
                        label="Feature Activation Heatmaps",
                        columns=8,
                        rows=3,
                        height="auto",
                        object_fit="contain",
                        elem_classes=["heatmap-gallery"],
                        visible=False,
                    )

        def update_heatmap_concept_dropdown(concept_meta, *checkbox_values):
            """Update available concepts for heatmap visualization dropdown"""
            if not concept_meta:
                return gr.update(choices=[], value=None)

            selected = []
            for i, is_checked in enumerate(checkbox_values[: len(concept_meta)]):
                if is_checked:
                    name, concept_id, _ = concept_meta[i]
                    selected.append(name)

            return gr.update(
                choices=selected,
                value=selected[0] if selected else None,
            )

        for cb in concept_components["checkboxes"]:
            cb.change(
                fn=update_heatmap_concept_dropdown,
                inputs=[concept_metadata] + concept_components["checkboxes"],
                outputs=[heatmap_concept_dropdown],
            )

        def update_heatmap_btn_state(*args):
            """Enable heatmap button only if SD, SAE loaded, concept selected, and valid timesteps"""
            sd_loaded = state.model_states.get("sd_base") == ModelLoadState.LOADED
            sae_loaded = state.model_states.get("sae") == ModelLoadState.LOADED

            timesteps_input = args[0] if len(args) > 0 else ""
            concept_selected = args[1] if len(args) > 1 else None

            valid_timesteps = False
            if timesteps_input and timesteps_input.strip():
                try:
                    parsed = [int(t.strip()) for t in timesteps_input.split(",") if t.strip()]
                    valid_timesteps = len(parsed) > 0
                except ValueError:
                    valid_timesteps = False

            can_generate = (
                sd_loaded and sae_loaded and concept_selected is not None and valid_timesteps
            )
            return gr.update(interactive=can_generate)

        heatmap_timesteps.change(
            fn=update_heatmap_btn_state,
            inputs=[heatmap_timesteps, heatmap_concept_dropdown],
            outputs=[generate_heatmap_btn],
        )

        heatmap_concept_dropdown.change(
            fn=update_heatmap_btn_state,
            inputs=[heatmap_timesteps, heatmap_concept_dropdown],
            outputs=[generate_heatmap_btn],
        )

        def handle_generate_heatmaps(
            prompt,
            guidance,
            seed,
            sae_model_id,
            nudenet_enabled,
            intervention_mode_value,
            heatmap_timesteps_input,
            heatmap_concept_selected,
            concept_meta,
            *concept_values,
            progress=gr.Progress(),
        ):
            """
            Generate heatmaps for the selected concept.
            Uses the neuron count from the intervention section.
            """
            print("\n" + "=" * 80)
            print("[DEBUG] handle_generate_heatmaps() CALLED")
            print("=" * 80)
            print(f"[DEBUG] concept_selected: {heatmap_concept_selected}")
            print(f"[DEBUG] timesteps_input: {heatmap_timesteps_input}")

            yield (
                "",
                gr.update(value=[], visible=False),
                '<div style="text-align: center; padding: 0.5rem; color: #5b9bd5;">🎨 Starting generation...</div>',
            )

            if state.sd_pipe is None or state.sae_model is None:
                print("[DEBUG] ERROR: Models not loaded")
                gr.Warning("Please load both SD and SAE models first.")
                yield "", gr.update(value=[], visible=False), ""
                return

            if not heatmap_concept_selected:
                print("[DEBUG] ERROR: No concept selected")
                gr.Warning("Please select a concept from the intervention section.")
                yield "", gr.update(value=[], visible=False), ""
                return

            neurons_count = 8
            per_timestep = intervention_mode_value == "per_timestep"

            if concept_meta:
                for i in range(min(len(concept_meta), MAX_CONCEPTS)):
                    idx = i * 3
                    if idx + 2 < len(concept_values):
                        is_selected = concept_values[idx]
                        neurons = concept_values[idx + 2]
                        if is_selected:
                            name, concept_id, _ = concept_meta[i]
                            if name == heatmap_concept_selected:
                                neurons_count = int(neurons)
                                print(
                                    f"[DEBUG] Found concept '{name}' with {neurons_count} neurons"
                                )
                                break

            if seed == -1:
                import random

                seed = random.randint(0, 2**32 - 1)  # noqa: S311

            try:
                heatmap_timesteps_list = [
                    int(t.strip()) for t in heatmap_timesteps_input.split(",") if t.strip()
                ]
            except ValueError:
                heatmap_timesteps_list = [10, 25, 40]
                state.log("Invalid timesteps format, using default [10,25,40]", "warning")

            state.log(
                f"Generating heatmaps for '{heatmap_concept_selected}' "
                f"at timesteps {heatmap_timesteps_list}...",
                "info",
            )
            gr.Info(f"Generating heatmaps for '{heatmap_concept_selected}'...", duration=3)

            try:
                layer = get_layer_id(SAE_CONFIG, sae_model_id)
                layer_path = LayerPath[layer] if layer else LayerPath.UNET_UP_1_ATT_1
                device = str(state.sd_pipe.device)

                generator = torch.Generator(device=device).manual_seed(seed)

                if progress:
                    progress(0, desc="🎨 Starting generation...")

                representations, final_image, latents = capture_layer_representations(
                    pipe=state.sd_pipe,
                    prompt=prompt,
                    layer_paths=[layer_path],
                    num_inference_steps=DENOISING_STEPS,
                    guidance_scale=guidance,
                    generator=generator,
                    capture_latents=True,
                )

                if representations[0] is None:
                    state.log("Failed to capture representations", "error")
                    yield "", gr.update(value=[], visible=False), ""
                    return

                yield (
                    "",
                    gr.update(value=[], visible=False),
                    '<div style="text-align: center; padding: 0.5rem; color: #5b9bd5;">🧠 Processing activations...</div>',
                )

                if progress:
                    progress(0.5, desc="🧠 Processing activations...")

                intermediate_images = {}
                if latents is not None:
                    for timestep in heatmap_timesteps_list:
                        if timestep < latents.shape[0]:
                            latent = latents[timestep]
                            intermediate_img = decode_latent_to_image(latent, state.sd_pipe, device)
                            intermediate_images[timestep] = intermediate_img

                stats_key = heatmap_concept_selected

                if concept_meta:
                    for name, concept_id, _ in concept_meta:
                        if name == heatmap_concept_selected:
                            stats_key = concept_id.replace("_", " ")
                            break

                print(f"[DEBUG] Looking up stats_key: '{stats_key}'")

                if stats_key not in state.sae_stats:
                    state.log(f"Concept '{stats_key}' not found in SAE stats", "error")
                    yield "", gr.update(value=[], visible=False), ""
                    return

                modifier = RepresentationModifier(
                    sae=state.sae_model,
                    stats_dict=state.sae_stats,
                    epsilon=1e-8,
                    device=device,
                    max_concepts_number=32,
                )

                feature_scores = modifier.calculate_scores_for_concept(
                    concept_name=stats_key,
                    per_timestep=per_timestep,
                )

                yield (
                    "",
                    gr.update(value=[], visible=False),
                    '<div style="text-align: center; padding: 0.5rem; color: #5b9bd5;">🔥 Generating heatmaps...</div>',
                )

                activations = collect_activations_from_representations(
                    representations=representations[0],
                    sae=state.sae_model,
                    timesteps=heatmap_timesteps_list,
                    top_k_features=neurons_count,
                    device=device,
                    feature_scores=feature_scores,
                )

                heatmap_gallery_data = generate_heatmap_gallery(
                    activations=activations,
                    image=final_image,
                    alpha=0.4,
                    max_features=neurons_count,
                    intermediate_images=intermediate_images,
                    nudenet_detector=state.nudenet_detector,
                    apply_censorship=nudenet_enabled,
                    state=state,
                    concept_name=heatmap_concept_selected,
                )

                num_timesteps = len(heatmap_timesteps_list)
                section_header_html = f"""
<div class="analysis-container heatmap-header-container">
    <div class="analysis-header">
        <p class="analysis-description">
            Visualizing <strong>{neurons_count}</strong> top features for concept
            "<strong>{heatmap_concept_selected}</strong>" across
            <strong>{num_timesteps}</strong> timesteps.
            Each row shows one timestep, columns show individual feature activations.
        </p>
    </div>
</div>
"""

                if progress:
                    progress(1.0, desc="✅ Complete!")

                state.log(
                    f"Generated {len(heatmap_gallery_data)} heatmaps "
                    f"for '{heatmap_concept_selected}'",
                    "success",
                )
                gr.Info(f"✓ Generated {len(heatmap_gallery_data)} heatmaps", duration=5)

                yield (
                    section_header_html,
                    gr.update(value=heatmap_gallery_data, columns=neurons_count, visible=True),
                    "",
                )

            except Exception as e:
                state.log(f"Heatmap generation failed: {str(e)}", "error")
                print(f"[DEBUG] Heatmap error: {e}")
                import traceback

                traceback.print_exc()
                gr.Warning(f"Heatmap generation failed: {str(e)}")
                yield "", gr.update(value=[], visible=False), ""

        heatmap_generate_inputs = [
            prompt_input,
            guidance_input,
            seed_input,
            sae_model_dropdown,
            nudenet_checkbox,
            intervention_mode,
            heatmap_timesteps,
            heatmap_concept_dropdown,
            concept_metadata,
        ]
        for i in range(MAX_CONCEPTS):
            heatmap_generate_inputs.append(concept_components["checkboxes"][i])
            heatmap_generate_inputs.append(concept_components["strengths"][i])
            heatmap_generate_inputs.append(concept_components["neurons"][i])

        def update_generate_button_state(*checkbox_values):
            """Enable generate button only if SD loaded, SAE loaded, and at least one concept selected"""
            sd_loaded = state.model_states.get("sd_base") == ModelLoadState.LOADED
            sae_loaded = state.model_states.get("sae") == ModelLoadState.LOADED

            any_concept_selected = any(checkbox_values[:MAX_CONCEPTS]) if checkbox_values else False

            can_generate = sd_loaded and sae_loaded and any_concept_selected
            return gr.update(interactive=can_generate)

        def handle_sae_model_change(sae_model_id):
            """Handle SAE model selection - SAE needs to be loaded."""

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
                load_sd_model(state, use_gpu=use_gpu)
                load_time = time.time() - start_time
                state.set_model_state("sd_base", ModelLoadState.LOADED, load_time)

                state.log(f"SD v1.5 loaded on {device_type} in {load_time:.1f}s", "success")
                gr.Info(
                    f"✓ Stable Diffusion v1.5 loaded successfully on {device_type} ({load_time:.1f}s)",
                    duration=15,
                )
                yield (
                    gr.update(interactive=True),  # Re-enable load button
                    gr.update(interactive=False),  # Generate button managed by state checker
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

            def build_empty_result():
                result = [
                    gr.update(visible=False),  # concept_section
                    gr.update(value=False),  # enable_intervention
                    gr.update(elem_classes=["main-section", "section-not-loaded"]),
                    gr.update(interactive=False),  # generate_btn
                    [],  # concept_metadata
                ]
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(visible=False))
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(label="Concept", value=False))
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=4.0, interactive=False))
                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=8, interactive=False))
                return tuple(result)

            if sae_model_id == "none":
                yield build_empty_result()
                return

            loading_result = [
                gr.update(visible=False),
                gr.update(value=False),
                gr.update(elem_classes=["main-section", "section-loading"]),
                gr.update(interactive=False),
                [],
            ]
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update(visible=False))
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            for _ in range(MAX_CONCEPTS):
                loading_result.append(gr.update())
            yield tuple(loading_result)

            try:
                state.set_model_state("sae", ModelLoadState.LOADING)

                sae_name = sae_model_id
                layer_id = ""
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

                load_sae_model(
                    state,
                    sae_model_id,
                    SAE_CONFIG,
                    get_sae_hyperparameters,
                    get_feature_sums_path,
                    get_model_path,
                )
                load_time = time.time() - start_time
                state.set_model_state("sae", ModelLoadState.LOADED, load_time)

                state.log(f"SAE loaded in {load_time:.1f}s", "success")
                gr.Info(f"✓ SAE model loaded successfully ({load_time:.1f}s)", duration=15)

                concept_choices = []
                concept_choices = get_concept_choices(SAE_CONFIG, sae_model_id)

                result = [
                    gr.update(visible=True),
                    gr.update(value=True),
                    gr.update(elem_classes=["main-section", "section-loaded"]),
                    gr.update(interactive=False),
                    concept_choices,
                ]

                for i in range(MAX_CONCEPTS):
                    if i < len(concept_choices):
                        result.append(gr.update(visible=True))
                    else:
                        result.append(gr.update(visible=False))

                for i in range(MAX_CONCEPTS):
                    if i < len(concept_choices):
                        name, concept_id, description = concept_choices[i]
                        result.append(gr.update(label=name, value=False, info=description))
                    else:
                        result.append(gr.update(label="Concept", value=False, info=None))

                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=4.0, interactive=False))

                for _ in range(MAX_CONCEPTS):
                    result.append(gr.update(value=8, interactive=False))
                yield tuple(result)

            except Exception as e:
                state.set_model_state("sae", ModelLoadState.FAILED)
                state.log(f"SAE loading failed: {str(e)}", "error")
                gr.Warning(f"Failed to load SAE model: {str(e)}", duration=15)
                yield build_empty_result()

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

        for cb in concept_components["checkboxes"]:
            cb.change(
                fn=update_generate_button_state,
                inputs=concept_components["checkboxes"],
                outputs=[generate_btn],
            )

        enable_intervention.change(
            fn=toggle_intervention,
            inputs=[enable_intervention],
            outputs=[
                sae_model_dropdown,
                load_sae_btn,
            ],
        )

        def handle_generate(
            prompt,
            guidance,
            seed,
            intervention_enabled,
            sae_model_id,
            nudenet_enabled,
            intervention_mode_value,
            concept_meta,
            *concept_values,
            progress=gr.Progress(),
        ):
            """
            Generate original and intervened images with progress tracking.
            """
            per_timestep = intervention_mode_value == "per_timestep"

            print("\n" + "*" * 80)
            print("[DEBUG] handle_generate() CALLED")
            print("*" * 80)
            print(f"[DEBUG] prompt: {prompt[:50] if prompt else 'None'}...")
            print(f"[DEBUG] steps: {DENOISING_STEPS}, guidance: {guidance}, seed: {seed}")
            print(f"[DEBUG] intervention_enabled: {intervention_enabled}")
            print(f"[DEBUG] sae_model_id: {sae_model_id}")
            print(f"[DEBUG] nudenet_enabled: {nudenet_enabled}")
            print(
                f"[DEBUG] intervention_mode: {intervention_mode_value} (per_timestep={per_timestep})"
            )
            print(f"[DEBUG] concept_meta: {concept_meta}")
            print(f"[DEBUG] concept_values count: {len(concept_values)}")

            concept_configs = []
            if concept_meta:
                for i in range(min(len(concept_meta), MAX_CONCEPTS)):
                    idx = i * 3
                    if idx + 2 < len(concept_values):
                        is_selected = concept_values[idx]
                        strength = concept_values[idx + 1]
                        neurons = concept_values[idx + 2]
                        if is_selected:
                            _, concept_id, _ = concept_meta[i]
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

            if state.sd_pipe is None:
                print("[DEBUG] ERROR: SD pipe not loaded, returning None")
                yield (
                    None,
                    None,
                    "",
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
                return

            yield (
                None,
                None,
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(),
                "",
                gr.update(value=[], visible=False),
                '<div style="text-align: center; padding: 0.5rem; color: #5b9bd5;">🎨 Preparing generation...</div>',
            )

            if seed == -1:
                import random

                seed = random.randint(0, 2**32 - 1)  # noqa: S311
                print(f"[DEBUG] Resolved random seed to: {seed}")

            display_original = None
            display_intervened = None

            print("[DEBUG] Checking intervention conditions...")
            print(f"[DEBUG]   intervention_enabled: {intervention_enabled}")
            print(f"[DEBUG]   state.sae_model is not None: {state.sae_model is not None}")
            print(f"[DEBUG]   concept_configs (truthy): {bool(concept_configs)}")

            if intervention_enabled and state.sae_model is not None and concept_configs:
                print("[DEBUG] INTERVENTION MODE: Generating comparison images")
                state.log("Using comparison mode: generating both images sequentially", "info")

                layer = get_layer_id(SAE_CONFIG, sae_model_id)
                print(f"[DEBUG] Using layer: {layer}")

                combined_total = DENOISING_STEPS * 2

                def original_callback(pipe, step, timestep, callback_kwargs):
                    current = min(step + 1, combined_total)
                    if progress is not None:
                        progress(
                            (current, combined_total),
                            desc="🖼️ Original",
                        )
                    return callback_kwargs

                state.log(f"Generating original image (seed={seed})...", "info")
                state.system_state = SystemState.GENERATING
                state.start_generation(total_steps=DENOISING_STEPS, phase="original")

                generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                start_time = time.time()

                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=DENOISING_STEPS,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=original_callback,
                )

                original_image = output.images[0]
                original_time = time.time() - start_time
                state.log(f"Original image generated in {original_time:.2f}s", "success")

                def intervention_callback(pipe, step, timestep, callback_kwargs):
                    current = min(DENOISING_STEPS + step + 1, combined_total)
                    if progress is not None:
                        progress(
                            (current, combined_total),
                            desc="🔧 Intervention",
                        )
                    return callback_kwargs

                state.log(f"Generating intervention image (seed={seed})...", "info")

                try:
                    device = str(state.sd_pipe.device)

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
                                per_timestep=per_timestep,
                            )

                        layer_path = LayerPath[layer] if layer else LayerPath.UNET_UP_1_ATT_1
                        modifier.reset_timestep()
                        modifier.attach_to(state.sd_pipe, layer_path)

                        generator2 = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                        start_time2 = time.time()

                        with modifier:
                            output2 = state.sd_pipe(
                                prompt=prompt,
                                num_inference_steps=DENOISING_STEPS,
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

                if progress is not None:
                    progress((combined_total, combined_total), desc="📊 Calculating metrics...")

                display_original = original_image
                display_intervened = intervened_image

                detection_orig = None
                detection_interv = None

                if original_image is not None and state.nudenet_detector is not None:
                    detection_orig, censored_orig = detect_content(original_image, state)
                    if (
                        nudenet_enabled
                        and censored_orig is not None
                        and detection_orig.get("has_unsafe", False)
                    ):
                        display_original = censored_orig
                        state.log("Displaying censored version of original image", "info")

                if intervened_image is not None and state.nudenet_detector is not None:
                    detection_interv, censored_interv = detect_content(intervened_image, state)
                    if (
                        nudenet_enabled
                        and censored_interv is not None
                        and detection_interv.get("has_unsafe", False)
                    ):
                        display_intervened = censored_interv
                        state.log("Displaying censored version of intervened image", "info")

                scores_comparison = format_nudenet_comparison(detection_orig, detection_interv)

                clip_device = str(state.sd_pipe.device)
                clip_scores = calculate_clip_scores(
                    prompt, original_image, intervened_image, clip_device, state
                )
                clip_scores_html = format_clip_scores(clip_scores)

                state.system_state = SystemState.IDLE

                print("[DEBUG] Yielding both images and scores for display")
                print("*" * 80 + "\n")
                yield (
                    display_original,
                    display_intervened,
                    scores_comparison,
                    clip_scores_html,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=seed),
                    "",
                    gr.update(value=[], visible=False),
                    "",
                )

            else:
                print("[DEBUG] NO INTERVENTION MODE: Generating original only")
                if not intervention_enabled:
                    print("[DEBUG]   Reason: intervention_enabled is False")
                if state.sae_model is None:
                    print("[DEBUG]   Reason: state.sae_model is None")
                if not concept_configs:
                    print("[DEBUG]   Reason: no concepts selected")

                state.log("Generating original image only (no intervention)", "info")

                def solo_callback(pipe, step, timestep, callback_kwargs):
                    current = min(step + 1, DENOISING_STEPS)
                    if progress is not None:
                        progress(
                            (current, DENOISING_STEPS),
                            desc="🖼️ Generating",
                        )
                    return callback_kwargs

                state.system_state = SystemState.GENERATING
                state.start_generation(total_steps=DENOISING_STEPS, phase="original")

                generator = torch.Generator(device=state.sd_pipe.device).manual_seed(seed)
                start_time = time.time()

                output = state.sd_pipe(
                    prompt=prompt,
                    num_inference_steps=DENOISING_STEPS,
                    guidance_scale=guidance,
                    generator=generator,
                    callback_on_step_end=solo_callback,
                )

                original_image = output.images[0]
                generation_time = time.time() - start_time
                state.log(f"Original image generated in {generation_time:.2f}s", "success")

                if progress is not None:
                    progress(
                        (DENOISING_STEPS, DENOISING_STEPS),
                        desc=f"✅ Complete ({generation_time:.1f}s)",
                    )

                display_original = original_image
                print(f"[DEBUG] Original image generated: {original_image is not None}")

                detection_orig = None
                if original_image is not None and state.nudenet_detector is not None:
                    detection_orig, censored_orig = detect_content(original_image, state)
                    if (
                        nudenet_enabled
                        and censored_orig is not None
                        and detection_orig.get("has_unsafe", False)
                    ):
                        display_original = censored_orig
                        state.log("Displaying censored version of original image", "info")

                scores_comparison = format_nudenet_comparison(detection_orig, None)

                clip_device = str(state.sd_pipe.device)
                clip_scores = calculate_clip_scores(
                    prompt, original_image, None, clip_device, state
                )
                clip_scores_html = format_clip_scores(clip_scores)

                state.system_state = SystemState.IDLE

                print("[DEBUG] Yielding original image only (no intervention)")
                print("*" * 80 + "\n")
                yield (
                    display_original,
                    None,
                    scores_comparison,
                    clip_scores_html,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value=seed),
                    "",
                    gr.update(value=[], visible=False),
                    "",
                )

        generate_inputs = [
            prompt_input,
            guidance_input,
            seed_input,
            enable_intervention,
            sae_model_dropdown,
            nudenet_checkbox,
            intervention_mode,
            concept_metadata,
        ]
        for i in range(MAX_CONCEPTS):
            generate_inputs.append(concept_components["checkboxes"][i])
            generate_inputs.append(concept_components["strengths"][i])
            generate_inputs.append(concept_components["neurons"][i])

        generation_controls = [
            prompt_input,
            guidance_input,
            seed_input,
            nudenet_checkbox,
            intervention_mode,
            generate_btn,
            sd_model_dropdown,
            use_gpu_checkbox,
            load_sd_btn,
            sae_model_dropdown,
            load_sae_btn,
        ]
        generation_controls.extend(concept_components["checkboxes"])
        generation_controls.extend(concept_components["strengths"])
        generation_controls.extend(concept_components["neurons"])

        def disable_controls():
            """Disable all controls before generation starts"""
            result = [
                gr.update(interactive=False),  # prompt_input
                gr.update(interactive=False),  # guidance_input
                gr.update(interactive=False),  # seed_input
                gr.update(interactive=False),  # nudenet_checkbox
                gr.update(interactive=False),  # intervention_mode
                gr.update(interactive=False),  # generate_btn
                gr.update(interactive=False),  # sd_model_dropdown
                gr.update(interactive=False),  # use_gpu_checkbox
                gr.update(interactive=False),  # load_sd_btn
                gr.update(interactive=False),  # sae_model_dropdown
                gr.update(interactive=False),  # load_sae_btn
            ]
            for _ in range(MAX_CONCEPTS * 3):
                result.append(gr.update(interactive=False))
            return tuple(result)

        def enable_controls(*checkbox_values):
            """Re-enable all controls after generation completes"""
            result = [
                gr.update(interactive=True),  # prompt_input
                gr.update(interactive=True),  # guidance_input
                gr.update(interactive=True),  # seed_input
                gr.update(interactive=True),  # nudenet_checkbox
                gr.update(interactive=True),  # intervention_mode
                gr.update(interactive=True),  # generate_btn
                gr.update(interactive=True),  # sd_model_dropdown
                gr.update(interactive=CUDA_COMPATIBLE),  # use_gpu_checkbox
                gr.update(interactive=True),  # load_sd_btn
                gr.update(interactive=True),  # sae_model_dropdown
                gr.update(interactive=True),  # load_sae_btn
            ]
            for _ in range(MAX_CONCEPTS):
                result.append(gr.update(interactive=True))  # checkboxes
            for i in range(MAX_CONCEPTS):
                is_checked = checkbox_values[i] if i < len(checkbox_values) else False
                result.append(gr.update(interactive=is_checked))  # strengths
            for i in range(MAX_CONCEPTS):
                is_checked = checkbox_values[i] if i < len(checkbox_values) else False
                result.append(gr.update(interactive=is_checked))  # neurons
            return tuple(result)

        generate_btn.click(
            fn=disable_controls,
            inputs=None,
            outputs=generation_controls,
        ).then(
            fn=handle_generate,
            inputs=generate_inputs,
            outputs=[
                img_original,
                img_unlearned,
                nudenet_scores_comparison,
                clip_scores_display,
                images_accordion,
                nudenet_accordion,
                clip_accordion,
                heatmap_accordion,
                seed_input,
                heatmap_section_html,
                heatmap_section_gallery,
                generation_progress,
            ],
        ).then(
            fn=enable_controls,
            inputs=concept_components["checkboxes"],
            outputs=generation_controls,
        )

        load_sd_btn.click(
            fn=handle_load_sd,
            inputs=[sd_model_dropdown, use_gpu_checkbox],
            outputs=[
                load_sd_btn,
                generate_btn,
                section_base_gen,
            ],
        ).then(
            fn=update_generate_button_state,
            inputs=concept_components["checkboxes"],
            outputs=[generate_btn],
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

        generate_heatmap_btn.click(
            fn=disable_controls,
            inputs=None,
            outputs=generation_controls,
        ).then(
            fn=handle_generate_heatmaps,
            inputs=heatmap_generate_inputs,
            outputs=[heatmap_section_html, heatmap_section_gallery, heatmap_progress],
        ).then(
            fn=enable_controls,
            inputs=concept_components["checkboxes"],
            outputs=generation_controls,
        )

    return app


if __name__ == "__main__":
    load_nudenet_model(state)
    app = create_dashboard()

    css_path = Path(__file__).parent / "style.css"
    custom_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    app.queue()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        css=custom_css,
        favicon_path=Path(__file__).parent / "favicon.ico",
    )

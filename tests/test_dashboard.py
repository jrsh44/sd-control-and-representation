"""
Dashboard Tests - Phase 9: Testing & Validation
Comprehensive test suite for the SD Control & Representation dashboard

Tests the modular dashboard components:
- state.py: State management classes
- layers.py: UNet layer selection
- concepts.py: Concept loading and management
"""

import sys
from pathlib import Path

import pytest

# Add dashboard to path for imports
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Test DashboardState class from state.py"""

    def test_state_initialization(self):
        """Test state initializes correctly"""
        from core.state import DashboardState, ModelLoadState, SystemState

        state = DashboardState()
        assert state.system_state == SystemState.IDLE
        assert state.sd_pipe is None
        assert state.sae_model is None
        assert state.nudenet_detector is None
        assert len(state.model_states) == 3
        assert state.model_states["sd_base"] == ModelLoadState.NOT_LOADED

    def test_log_message(self):
        """Test logging functionality"""
        from core.state import DashboardState

        state = DashboardState()
        result = state.log("Test message", "info")
        assert "Test message" in result
        assert "[INFO]" in result
        assert len(state.logs) > 0

    def test_model_state_update(self):
        """Test model state updates"""
        from core.state import DashboardState, ModelLoadState

        state = DashboardState()
        state.set_model_state("sd_base", ModelLoadState.LOADED, 5.2)
        assert state.model_states["sd_base"] == ModelLoadState.LOADED
        assert state.load_times["sd_base"] == 5.2

    def test_get_model_status_text(self):
        """Test model status text generation"""
        from core.state import DashboardState

        state = DashboardState()
        status_text = state.get_model_status_text()
        assert isinstance(status_text, str)
        assert "MODEL STATUS" in status_text

    def test_generation_progress(self):
        """Test generation progress tracking"""
        import time

        from core.state import GenerationProgress

        progress = GenerationProgress(
            phase="original",
            current_step=25,
            total_steps=50,
            start_time=time.time() - 10,
        )
        assert progress.progress == 0.5
        assert progress.elapsed_time >= 10
        assert progress.format_time(65) == "01:05"


# =============================================================================
# Layer Management Tests
# =============================================================================


class TestLayerManagement:
    """Test layer selection and information functions from layers.py"""

    def test_get_all_layers(self):
        """Test getting all layers"""
        from core.layers import get_all_layers

        layers = get_all_layers()

        assert isinstance(layers, dict)
        assert len(layers) > 0
        assert "Recommended" in layers
        assert "Down Blocks" in layers
        assert "Mid Block" in layers
        assert "Up Blocks" in layers

    def test_get_layer_choices(self):
        """Test layer choices formatting"""
        from core.layers import get_layer_choices

        choices = get_layer_choices()

        assert isinstance(choices, list)
        assert len(choices) > 0
        assert all(isinstance(choice, tuple) for choice in choices)
        assert all(len(choice) == 2 for choice in choices)

    def test_get_layer_info_single(self):
        """Test layer information retrieval for single layer"""
        from core.layers import get_layer_info

        info = get_layer_info("UNET_MID_ATT")
        assert "UNET_MID_ATT" in info
        assert "Layer:" in info

    def test_get_layer_info_empty(self):
        """Test layer info with empty selection"""
        from core.layers import get_layer_info

        info = get_layer_info([])
        assert "No layer selected" in info

    def test_get_layer_info_multiple(self):
        """Test layer info with multiple layers"""
        from core.layers import get_layer_info

        info = get_layer_info(["UNET_MID_ATT", "UNET_UP_1_ATT_1"])
        assert "2 layers selected" in info

    def test_get_flat_layer_list(self):
        """Test getting flat list of layers"""
        from core.layers import get_flat_layer_list

        layers = get_flat_layer_list()
        assert isinstance(layers, list)
        assert "UNET_MID_ATT" in layers
        assert "UNET_UP_1_ATT_1" in layers

    def test_is_recommended_layer(self):
        """Test checking if layer is recommended"""
        from core.layers import is_recommended_layer

        assert is_recommended_layer("UNET_MID_ATT") is True
        assert is_recommended_layer("UNET_DOWN_0_RES_0") is False


# =============================================================================
# Concept Management Tests
# =============================================================================


class TestConceptManagement:
    """Test concept loading and management from concepts.py"""

    def test_load_concepts(self):
        """Test concept loading from file"""
        from core.concepts import load_concepts

        concepts = load_concepts()

        assert isinstance(concepts, dict)
        assert len(concepts) > 0
        # Should have at least some concepts (or fallback)
        assert all(isinstance(k, int) for k in concepts.keys())
        assert all(isinstance(v, str) for v in concepts.values())

    def test_get_concept_choices(self):
        """Test concept choices formatting"""
        from core.concepts import get_concept_choices

        choices = get_concept_choices()

        assert isinstance(choices, list)
        assert len(choices) > 0
        assert all(isinstance(choice, tuple) for choice in choices)
        # Format: ("ID: Label", ID)
        assert all(len(choice) == 2 for choice in choices)

    def test_get_concept_info_empty(self):
        """Test concept info with empty selection"""
        from core.concepts import get_concept_info

        info = get_concept_info([])
        assert "No concepts selected" in info

    def test_get_concept_info_single(self):
        """Test concept info with single selection"""
        from core.concepts import get_concept_info

        info = get_concept_info([0])
        assert "1 concept" in info

    def test_get_concept_info_multiple(self):
        """Test concept info with multiple selection"""
        from core.concepts import get_concept_info

        info = get_concept_info([0, 1, 2])
        assert "3 concept" in info

    def test_get_concept_label(self):
        """Test getting single concept label"""
        from core.concepts import get_concept_label

        label = get_concept_label(0)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_validate_concepts(self):
        """Test concept validation"""
        from core.concepts import validate_concepts

        valid, invalid = validate_concepts([0, 1, 999])
        assert 0 in valid
        assert 1 in valid
        assert 999 in invalid


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_load_concepts_missing_file(self):
        """Test concept loading with missing file"""
        from core.concepts import load_concepts

        # Should return fallback concepts
        concepts = load_concepts(Path("nonexistent/path/classes.txt"))
        assert isinstance(concepts, dict)
        assert len(concepts) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full workflows"""

    def test_state_lifecycle(self):
        """Test complete state lifecycle"""
        from core.state import DashboardState, ModelLoadState, SystemState

        state = DashboardState()

        # Initial state
        assert state.system_state == SystemState.IDLE

        # Update system state
        state.system_state = SystemState.LOADING_MODEL
        assert state.system_state == SystemState.LOADING_MODEL

        # Update model state
        state.set_model_state("sd_base", ModelLoadState.LOADING)
        state.set_model_state("sd_base", ModelLoadState.LOADED, 3.5)

        # Check status
        status = state.get_model_status_text()
        assert "Loaded" in status or "●" in status

    def test_layer_concept_workflow(self):
        """Test layer and concept selection workflow"""
        from core.concepts import get_concept_choices, get_concept_info
        from core.layers import get_layer_choices, get_layer_info

        # Get available layers
        layer_choices = get_layer_choices()
        assert len(layer_choices) > 0

        # Select a layer
        selected_layer = [layer_choices[0][1]]
        layer_info = get_layer_info(selected_layer)
        assert len(layer_info) > 0

        # Get available concepts
        concept_choices = get_concept_choices()
        assert len(concept_choices) > 0

        # Select concepts
        selected_concepts = [concept_choices[0][1], concept_choices[1][1]]
        concept_info = get_concept_info(selected_concepts)
        assert "2 concept" in concept_info


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

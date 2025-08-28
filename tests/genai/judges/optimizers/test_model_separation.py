"""Tests for model separation between optimizer and judge."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer


class TestOptimizerJudgeModelSeparation:
    """Test cases for model separation between optimizer and judge."""

    def test_optimizer_default_model_initialization(self):
        """Test that optimizer uses default model when none specified."""
        with patch("mlflow.genai.judges.utils.get_default_model") as mock_get_default:
            mock_get_default.return_value = "openai:/gpt-4.1-mini"

            optimizer = SIMBAAlignmentOptimizer()

            assert optimizer._model == "openai:/gpt-4.1-mini"
            mock_get_default.assert_called_once()

    def test_optimizer_custom_model_initialization(self):
        """Test that optimizer uses custom model when specified."""
        custom_model = "anthropic:/claude-3.5-sonnet"

        optimizer = SIMBAAlignmentOptimizer(model=custom_model)

        assert optimizer._model == custom_model

    @patch("dspy.LM")
    def test_setup_dspy_model_uses_optimizer_model(self, mock_lm):
        """Test DSPy model setup uses optimizer's model directly."""
        mock_lm_instance = Mock()
        mock_lm.return_value = mock_lm_instance

        optimizer = SIMBAAlignmentOptimizer(model="openai:/gpt-4")

        result = optimizer._setup_dspy_model()

        assert result == mock_lm_instance
        mock_lm.assert_called_once_with(model="openai:/gpt-4")

    @patch("dspy.LM")
    def test_setup_dspy_model_anthropic(self, mock_lm):
        """Test DSPy model setup for Anthropic models."""
        mock_lm_instance = Mock()
        mock_lm.return_value = mock_lm_instance

        optimizer = SIMBAAlignmentOptimizer(model="anthropic:/claude-3.5-sonnet")

        result = optimizer._setup_dspy_model()

        assert result == mock_lm_instance
        mock_lm.assert_called_once_with(model="anthropic:/claude-3.5-sonnet")

    @patch("dspy.LM")
    def test_setup_dspy_model_custom_format(self, mock_lm):
        """Test DSPy model setup with custom model format."""
        mock_lm_instance = Mock()
        mock_lm.return_value = mock_lm_instance

        optimizer = SIMBAAlignmentOptimizer(model="custom-model-format")

        result = optimizer._setup_dspy_model()

        assert result == mock_lm_instance
        mock_lm.assert_called_once_with(model="custom-model-format")

    def test_model_parameter_preserved_through_inheritance(self):
        """Test that model parameter is properly handled in inheritance chain."""
        # Test with custom model
        optimizer = SIMBAAlignmentOptimizer(model="custom:/model")
        assert optimizer._model == "custom:/model"

        # Test with default model
        with patch("mlflow.genai.judges.utils.get_default_model") as mock_get_default:
            mock_get_default.return_value = "default:/model"

            optimizer_default = SIMBAAlignmentOptimizer()
            assert optimizer_default._model == "default:/model"

    def test_kwargs_handling_with_model_parameter(self):
        """Test that kwargs are properly handled alongside model parameter."""
        optimizer = SIMBAAlignmentOptimizer(
            model="custom:/model", custom_param="value", another_param=42
        )

        assert optimizer._model == "custom:/model"
        assert optimizer._kwargs == {"custom_param": "value", "another_param": 42}

        # Verify SIMBA-specific parameters are still set correctly
        assert optimizer._bsize == SIMBAAlignmentOptimizer.DEFAULT_BSIZE
        assert optimizer._seed == SIMBAAlignmentOptimizer.DEFAULT_SEED


class TestDSPyContextUsage:
    """Test DSPy context manager usage for model separation."""

    @patch("dspy.context")
    def test_dspy_context_manager_called(self, mock_context):
        """Test that DSPy context manager is properly called during alignment."""
        # Setup mock context manager
        mock_context_manager = MagicMock()
        mock_context.return_value = mock_context_manager

        # Setup optimizer with model
        optimizer = SIMBAAlignmentOptimizer(model="openai:/gpt-4")

        # Mock the _setup_dspy_model method
        mock_dspy_model = Mock()
        with patch.object(optimizer, "_setup_dspy_model", return_value=mock_dspy_model):
            # Create minimal mocks to get through align method
            mock_judge = Mock()
            mock_judge.name = "test_judge"
            mock_judge.description = "Test description"

            # Mock trace that will fail to convert (to keep test simple)
            mock_trace = Mock()
            mock_trace.info.assessments = []

            with patch.object(optimizer, "_trace_to_dspy_example", return_value=None):
                # This should fail with "No valid examples" but first call context
                with pytest.raises(MlflowException, match="No valid examples"):
                    optimizer.align(mock_judge, [mock_trace])

                # Verify DSPy context was called with optimizer's model
                mock_context.assert_called_once_with(lm=mock_dspy_model)
                mock_context_manager.__enter__.assert_called_once()
                mock_context_manager.__exit__.assert_called_once()


class TestModelSeparationIntegration:
    """Integration tests for model separation scenarios."""

    def test_different_models_no_interference(self):
        """Test that different optimizers maintain separate models."""
        optimizer1 = SIMBAAlignmentOptimizer(model="openai:/gpt-3.5-turbo")
        optimizer2 = SIMBAAlignmentOptimizer(model="anthropic:/claude-3")

        assert optimizer1._model == "openai:/gpt-3.5-turbo"
        assert optimizer2._model == "anthropic:/claude-3"

        # Verify they don't interfere with each other
        assert optimizer1._model != optimizer2._model

        # Verify SIMBA parameters are independent
        assert optimizer1._bsize == optimizer2._bsize == SIMBAAlignmentOptimizer.DEFAULT_BSIZE
        assert optimizer1._seed == optimizer2._seed == SIMBAAlignmentOptimizer.DEFAULT_SEED

    def test_optimizer_model_separate_from_judge_model(self):
        """Test that optimizer model is separate from judge model."""
        # Create optimizer with specific model
        optimizer = SIMBAAlignmentOptimizer(model="anthropic:/claude-3.5-sonnet")

        # Create mock judge with different model
        mock_judge = Mock()
        mock_judge.name = "test_judge"
        mock_judge.description = "Test description"
        # Simulate judge having its own model
        mock_judge._model_uri = "openai:/gpt-4"

        # Verify they have different models
        assert optimizer._model == "anthropic:/claude-3.5-sonnet"
        assert mock_judge._model_uri == "openai:/gpt-4"
        assert optimizer._model != mock_judge._model_uri

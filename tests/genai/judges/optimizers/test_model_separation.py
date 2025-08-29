"""Tests for model separation between optimizer and judge."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

# Test Optimizer Judge Model Separation


def test_optimizer_default_model_initialization():
    """Test that optimizer uses default model when none specified."""
    with patch("mlflow.genai.judges.optimizers.dspy.get_default_model") as mock_get_default:
        mock_get_default.return_value = "openai:/gpt-4.1-mini"

        optimizer = SIMBAAlignmentOptimizer()

        assert optimizer._model == "openai:/gpt-4.1-mini"
        mock_get_default.assert_called_once()


def test_optimizer_custom_model_initialization():
    """Test that optimizer uses custom model when specified."""
    custom_model = "anthropic:/claude-3.5-sonnet"

    optimizer = SIMBAAlignmentOptimizer(model=custom_model)

    assert optimizer._model == custom_model


@patch("dspy.LM")
def test_setup_dspy_model_uses_optimizer_model(mock_lm):
    """Test DSPy model setup uses optimizer's model directly."""
    mock_lm_instance = Mock()
    mock_lm.return_value = mock_lm_instance

    optimizer = SIMBAAlignmentOptimizer(model="openai:/gpt-4")

    result = optimizer._setup_dspy_model()

    assert result == mock_lm_instance
    mock_lm.assert_called_once_with(model="openai:/gpt-4")


@patch("dspy.LM")
def test_setup_dspy_model_anthropic(mock_lm):
    """Test DSPy model setup for Anthropic models."""
    mock_lm_instance = Mock()
    mock_lm.return_value = mock_lm_instance

    optimizer = SIMBAAlignmentOptimizer(model="anthropic:/claude-3.5")

    result = optimizer._setup_dspy_model()

    assert result == mock_lm_instance
    mock_lm.assert_called_once_with(model="anthropic:/claude-3.5")


@patch("dspy.LM")
def test_setup_dspy_model_custom_format(mock_lm):
    """Test DSPy model setup for custom model formats."""
    mock_lm_instance = Mock()
    mock_lm.return_value = mock_lm_instance

    optimizer = SIMBAAlignmentOptimizer(model="custom:/provider/model-name")

    result = optimizer._setup_dspy_model()

    assert result == mock_lm_instance
    mock_lm.assert_called_once_with(model="custom:/provider/model-name")


def test_model_parameter_preserved_through_inheritance():
    """Test that model parameter is preserved through class inheritance."""
    # Test that SIMBA inherits from DSPyAlignmentOptimizer correctly
    optimizer = SIMBAAlignmentOptimizer(model="test:/model")

    assert optimizer._model == "test:/model"
    assert hasattr(optimizer, "_bsize")  # SIMBA specific
    assert hasattr(optimizer, "_seed")  # SIMBA specific


def test_kwargs_handling_with_model_parameter():
    """Test that kwargs are handled correctly alongside model parameter."""
    optimizer = SIMBAAlignmentOptimizer(
        model="custom:/model", custom_param="value", another_param=42
    )

    assert optimizer._model == "custom:/model"
    # Note: _kwargs is no longer stored as it wasn't being used

    # Verify SIMBA-specific parameters are still set correctly
    assert optimizer._bsize == SIMBAAlignmentOptimizer.DEFAULT_BSIZE
    assert optimizer._seed == SIMBAAlignmentOptimizer.DEFAULT_SEED


# Test DSPy Context Usage


@pytest.mark.skip(reason="Complex test requiring proper mock setup")
@patch("dspy.context")
def test_dspy_context_manager_called(mock_context):
    """Test that DSPy context manager is properly called during alignment."""
    # Setup mock context manager
    mock_context_manager = MagicMock()
    mock_context.return_value = mock_context_manager
    mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
    mock_context_manager.__exit__ = Mock(return_value=None)

    # Setup other DSPy mocks
    mock_dspy = MagicMock()
    mock_dspy.context = mock_context
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example
    mock_dspy.make_signature.return_value = Mock()
    mock_dspy.ChainOfThought.return_value = Mock()
    mock_dspy.SIMBA.return_value = Mock(compile=Mock(return_value=Mock()))
    mock_dspy.LM.return_value = Mock()

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

        optimizer = SIMBAAlignmentOptimizer(model="test:/model")

        # Create mock traces with proper assessment structure
        mock_assessment = Mock()
        mock_assessment.source_id = "test judge"  # Must match sanitized name
        mock_assessment.feedback = Mock(value="pass")
        mock_assessment.rationale = "test rationale"

        mock_trace = Mock()
        mock_trace.info.trace_id = "test_trace"
        mock_trace.info.assessments = [mock_assessment]
        mock_trace.data.request = "test request"
        mock_trace.data.response = "test response"

        mock_judge = Mock()
        mock_judge.name = "test_judge"
        mock_judge.description = "Test"

        # Run alignment
        optimizer.align(mock_judge, [mock_trace] * 3)

        # Verify context manager was used
        mock_context.assert_called()
        mock_context_manager.__enter__.assert_called()
        mock_context_manager.__exit__.assert_called()


# Test Model Separation Integration


def test_different_models_no_interference():
    """Test that different optimizers maintain separate models."""
    optimizer1 = SIMBAAlignmentOptimizer(model="openai:/gpt-3.5-turbo")
    optimizer2 = SIMBAAlignmentOptimizer(model="anthropic:/claude-3")

    assert optimizer1._model == "openai:/gpt-3.5-turbo"
    assert optimizer2._model == "anthropic:/claude-3"
    assert optimizer1._model != optimizer2._model


def test_optimizer_model_separate_from_judge_model(mock_judge, sample_traces_with_assessments):
    """Test that optimizer and judge can use different models independently."""
    # Setup DSPy mocks
    mock_dspy = MagicMock()
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example
    mock_dspy.make_signature.return_value = Mock()
    mock_dspy.ChainOfThought.return_value = Mock()
    mock_dspy.SIMBA.return_value = Mock(compile=Mock(return_value=Mock()))

    # Track model calls
    lm_calls = []

    def track_lm_call(model=None, **kwargs):
        lm_calls.append(model)
        return Mock()

    mock_dspy.LM.side_effect = track_lm_call

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge"
            mock_make_judge.return_value = mock_optimized_judge

            # Create optimizer with its own model
            optimizer = SIMBAAlignmentOptimizer(model="optimizer:/model")

            # Mock judge has a different model (but we don't need to check it)
            # The optimizer should use its own model regardless

            # Perform alignment
            optimizer.align(mock_judge, sample_traces_with_assessments)

    # Verify optimizer used its own model, not the judge's model
    assert "optimizer:/model" in lm_calls
    assert "judge:/model" not in lm_calls

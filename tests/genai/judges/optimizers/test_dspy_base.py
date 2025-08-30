"""Tests for DSPyAlignmentOptimizer base class."""

from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer

from tests.genai.judges.optimizers.conftest import MockDSPyLM


class ConcreteDSPyOptimizer(DSPyAlignmentOptimizer):
    """Concrete implementation for testing."""

    def _dspy_optimize(self, program, examples, metric_fn):
        # Mock implementation for testing
        mock_program = Mock()
        mock_program.signature = Mock()
        mock_program.signature.instructions = (
            "Optimized instructions with {{inputs}} and {{outputs}}"
        )
        return mock_program


def test_dspy_optimizer_abstract():
    """Test that DSPyAlignmentOptimizer cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        DSPyAlignmentOptimizer()


def test_concrete_implementation_required():
    """Test that concrete classes must implement _dspy_optimize method."""

    class IncompleteDSPyOptimizer(DSPyAlignmentOptimizer):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteDSPyOptimizer()


def test_concrete_implementation_works():
    """Test that concrete implementation can be instantiated."""
    optimizer = ConcreteDSPyOptimizer()
    assert optimizer is not None


def test_align_success(sample_traces_with_assessments):
    """Test successful alignment process."""
    # Create a mock judge with model attribute
    from tests.genai.judges.optimizers.conftest import MockJudge

    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")

    with patch("dspy.LM", MagicMock()):
        # Setup concrete optimizer
        optimizer = ConcreteDSPyOptimizer()

        result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result.model == mock_judge.model
    assert result.instructions == "Optimized instructions with {{inputs}} and {{outputs}}"


def test_align_no_traces(mock_judge):
    """Test alignment with no traces provided."""
    optimizer = ConcreteDSPyOptimizer()

    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(mock_judge, [])


def test_align_no_valid_examples(mock_judge, sample_trace_without_assessment):
    """Test alignment when no valid examples can be created."""
    with patch("dspy.LM", MagicMock()):
        optimizer = ConcreteDSPyOptimizer()
        with pytest.raises(MlflowException, match="No valid examples could be created"):
            optimizer.align(mock_judge, [sample_trace_without_assessment])


def test_align_insufficient_examples(mock_judge, sample_trace_with_assessment):
    """Test alignment with insufficient examples."""
    optimizer = ConcreteDSPyOptimizer()

    # Mock dspy first to avoid import errors
    with patch("dspy.LM", MagicMock()):
        with pytest.raises(MlflowException, match="At least 2 valid examples are required"):
            optimizer.align(mock_judge, [sample_trace_with_assessment])


def test_align_no_dspy(mock_judge, sample_traces_with_assessments):
    """Test alignment when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = ConcreteDSPyOptimizer()

        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer.align(mock_judge, sample_traces_with_assessments)


def _create_mock_dspy_lm_factory(optimizer_lm, judge_lm):
    """Factory function to create MockDSPyLM instances that track calls to LMs."""

    def mock_lm_factory(model=None, **kwargs):
        """Internal factory method to carry the input models"""
        # Choose the appropriate tracking list based on model
        if model == optimizer_lm.model:
            return optimizer_lm
        elif model == judge_lm.model:
            return judge_lm
        else:
            raise ValueError(f"Invalid model: {model}")

    return mock_lm_factory


def test_optimizer_and_judge_use_different_models(sample_traces_with_assessments):
    """Test that optimizer uses its own model while judge program uses judge's model."""
    from tests.genai.judges.optimizers.conftest import MockJudge

    # Setup models
    judge_model = "openai:/gpt-4"
    optimizer_model = "anthropic:/claude-3"

    # Create judge and traces
    mock_judge = MockJudge(name="mock_judge", model=judge_model)
    traces = sample_traces_with_assessments

    # Track LM calls and what models they use in context
    optimizer_lm = MockDSPyLM(optimizer_model)
    judge_lm = MockDSPyLM(judge_model)

    # Create LM factory that tracks calls to the underlying mocked LMs
    mock_lm_factory = _create_mock_dspy_lm_factory(optimizer_lm, judge_lm)

    # Direct patching approach: patch only LM, use real DSPy otherwise
    with patch.object(dspy, "LM", side_effect=mock_lm_factory):
        # Override ConcreteDSPyOptimizer's _dspy_optimize to call the program
        class TestDSPyOptimizer(ConcreteDSPyOptimizer):
            def _dspy_optimize(self, program, examples, metric_fn):
                lm_in_context = dspy.settings.lm
                assert lm_in_context == optimizer_lm

                # Simulate calling the program (which represents the judge)
                # This should happen with the judge's model context
                program(inputs=examples[0].inputs)

                # Return optimized program as usual
                return super()._dspy_optimize(program, examples, metric_fn)

        # Create optimizer with different model
        optimizer = TestDSPyOptimizer(model=optimizer_model)

        # Run alignment
        optimizer.align(mock_judge, traces)

        # Verify that the judge's LM was actually called during program execution
        # This ensures that the program call used the judge's model
        assert len(judge_lm.context_calls) > 0, (
            f"Expected judge LM to be called, but got {len(judge_lm.context_calls)} calls. "
            f"Optimizer calls: {len(optimizer_lm.context_calls)}"
        )

        # Verify that the optimizer's LM was not called
        assert len(optimizer_lm.context_calls) == 0, (
            f"Expected optimizer LM to not be called, but got "
            f"{len(optimizer_lm.context_calls)} calls. "
            f"Judge calls: {len(judge_lm.context_calls)}"
        )


def test_optimizer_default_model_initialization():
    """Test that optimizer uses default model when none specified."""
    with patch("mlflow.genai.judges.optimizers.dspy.get_default_model") as mock_get_default:
        mock_get_default.return_value = "openai:/gpt-4.1-mini"

        optimizer = ConcreteDSPyOptimizer()

        assert optimizer.model == "openai:/gpt-4.1-mini"
        mock_get_default.assert_called_once()


def test_optimizer_custom_model_initialization():
    """Test that optimizer uses custom model when specified."""
    custom_model = "anthropic:/claude-3.5-sonnet"

    optimizer = ConcreteDSPyOptimizer(model=custom_model)

    assert optimizer.model == custom_model


def test_different_models_no_interference():
    """Test that different optimizers maintain separate models."""
    optimizer1 = ConcreteDSPyOptimizer(model="openai:/gpt-3.5-turbo")
    optimizer2 = ConcreteDSPyOptimizer(model="anthropic:/claude-3")

    assert optimizer1.model == "openai:/gpt-3.5-turbo"
    assert optimizer2.model == "anthropic:/claude-3"
    assert optimizer1.model != optimizer2.model

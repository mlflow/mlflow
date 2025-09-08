"""Tests for DSPyAlignmentOptimizer base class."""

from typing import Any, Callable, Collection
from unittest.mock import MagicMock, Mock, patch

import dspy
import litellm
import pytest

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer

from tests.genai.judges.optimizers.conftest import MockDSPyLM, MockJudge


class ConcreteDSPyOptimizer(DSPyAlignmentOptimizer):
    """Concrete implementation for testing."""

    def _dspy_optimize(
        self,
        program: "dspy.Module",
        examples: Collection["dspy.Example"],
        metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
    ) -> "dspy.Module":
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
    # The instructions are wrapped by make_judge with a header and formatting
    assert "Optimized instructions with {{inputs}} and {{outputs}}" in result.instructions


def test_align_no_traces(mock_judge):
    """Test alignment with no traces provided."""
    optimizer = ConcreteDSPyOptimizer()

    with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
        optimizer.align(mock_judge, [])

    # Check that the main error message includes the exception details
    assert "No traces provided" in str(exc_info.value)
    # Check that the chained exception has the expected message
    assert exc_info.value.__cause__ is not None
    assert "No traces provided" in str(exc_info.value.__cause__)


def test_align_no_valid_examples(mock_judge, sample_trace_without_assessment):
    """Test alignment when no valid examples can be created."""
    with patch("dspy.LM", MagicMock()):
        optimizer = ConcreteDSPyOptimizer()
        with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
            optimizer.align(mock_judge, [sample_trace_without_assessment])

        # Check that the main error message includes the exception details
        assert "No valid examples could be created" in str(exc_info.value)
        # Check that the chained exception has the expected message
        assert exc_info.value.__cause__ is not None
        assert "No valid examples could be created" in str(exc_info.value.__cause__)


def test_align_insufficient_examples(mock_judge, sample_trace_with_assessment):
    """Test alignment with insufficient examples."""
    optimizer = ConcreteDSPyOptimizer()
    min_traces = optimizer.get_min_traces_required()

    # Mock dspy first to avoid import errors
    with patch("dspy.LM", MagicMock()):
        with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
            optimizer.align(mock_judge, [sample_trace_with_assessment])

        # Check that the main error message includes the exception details
        assert f"At least {min_traces} valid traces are required" in str(exc_info.value)
        # Check that the chained exception has the expected message
        assert exc_info.value.__cause__ is not None
        assert f"At least {min_traces} valid traces are required" in str(exc_info.value.__cause__)


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
    from mlflow.genai.judges.optimizers.dspy_utils import convert_mlflow_uri_to_litellm

    from tests.genai.judges.optimizers.conftest import MockJudge

    # Setup models
    judge_model = "openai:/gpt-4"
    optimizer_model = "anthropic:/claude-3"

    # Create judge and traces
    mock_judge = MockJudge(name="mock_judge", model=judge_model)
    traces = sample_traces_with_assessments

    # Track LM calls and what models they use in context
    # The MockDSPyLM should be initialized with the converted LiteLLM format
    # since that's what will be passed to the mock factory
    optimizer_lm = MockDSPyLM(convert_mlflow_uri_to_litellm(optimizer_model))
    judge_lm = MockDSPyLM(convert_mlflow_uri_to_litellm(judge_model))

    # Create LM factory that tracks calls to the underlying mocked LMs
    mock_lm_factory = _create_mock_dspy_lm_factory(optimizer_lm, judge_lm)

    # Direct patching approach: patch only LM, use real DSPy otherwise
    with patch.object(dspy, "LM", side_effect=mock_lm_factory):
        # Override ConcreteDSPyOptimizer's _dspy_optimize to call the program
        class TestDSPyOptimizer(ConcreteDSPyOptimizer):
            def _dspy_optimize(
                self,
                program: "dspy.Module",
                examples: Collection["dspy.Example"],
                metric_fn: Callable[["dspy.Example", Any, Any | None], bool],
            ) -> "dspy.Module":
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
        mock_get_default.return_value = "whichever default model is used"

        optimizer = ConcreteDSPyOptimizer()

        assert optimizer.model == "whichever default model is used"
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


def test_mlflow_to_litellm_uri_conversion_in_optimizer(sample_traces_with_assessments):
    """Test that MLflow URIs are correctly converted to LiteLLM format in optimizer."""
    from tests.genai.judges.optimizers.conftest import MockJudge

    # Setup models with MLflow URI format
    judge_model = "openai:/gpt-4"
    optimizer_model = "anthropic:/claude-3.5-sonnet"

    mock_judge = MockJudge(name="mock_judge", model=judge_model)

    # Track what models are passed to dspy.LM
    lm_calls = []

    def mock_lm_init(model=None, **kwargs):
        lm_calls.append(model)
        return MagicMock()

    with patch("dspy.LM", side_effect=mock_lm_init):
        optimizer = ConcreteDSPyOptimizer(model=optimizer_model)
        optimizer.align(mock_judge, sample_traces_with_assessments)

    # Check that URIs were converted to LiteLLM format (slash instead of colon-slash)
    assert "anthropic/claude-3.5-sonnet" in lm_calls
    assert "openai/gpt-4" in lm_calls
    # Original MLflow format should not be passed to dspy.LM
    assert "anthropic:/claude-3.5-sonnet" not in lm_calls
    assert "openai:/gpt-4" not in lm_calls


def test_mlflow_to_litellm_uri_conversion_in_judge_program():
    """Test that judge's model URI is converted when creating DSPy program."""
    from tests.genai.judges.optimizers.conftest import MockJudge

    # Create mock judge with MLflow URI format
    mock_judge = MockJudge(name="test_judge", model="openai:/gpt-4o-mini")

    optimizer = ConcreteDSPyOptimizer()

    # Track what model is passed to dspy.LM when creating judge program
    lm_calls = []

    def mock_lm_init(model=None, **kwargs):
        lm_calls.append(model)
        return MagicMock()

    with patch("dspy.LM", side_effect=mock_lm_init):
        program = optimizer._get_dspy_program_from_judge(mock_judge)
        # Force initialization of the LM by accessing the internal _lm
        _ = program._lm

    # Should have converted the URI
    assert "openai/gpt-4o-mini" in lm_calls
    assert "openai:/gpt-4o-mini" not in lm_calls


def test_dspy_align_litellm_nonfatal_error_messages_suppressed():
    """Test that LiteLLM nonfatal error messages are suppressed during DSPy align method."""
    suppression_state_during_call = {}

    def mock_dspy_optimize(program, examples, metric_fn):
        # Capture the state of litellm flags during the DSPy optimization call
        suppression_state_during_call["set_verbose"] = litellm.set_verbose
        suppression_state_during_call["suppress_debug_info"] = litellm.suppress_debug_info

        # Return a mock optimized program
        mock_program = Mock()
        mock_program.signature = Mock()
        mock_program.signature.instructions = "Optimized instructions"
        return mock_program

    mock_traces = [Mock(spec=Trace) for _ in range(10)]
    mock_judge = MockJudge(name="test_judge", model="openai:/gpt-4o-mini")
    optimizer = ConcreteDSPyOptimizer()

    with (
        patch("dspy.LM"),
        patch("mlflow.genai.judges.optimizers.dspy.trace_to_dspy_example", return_value=Mock()),
        patch("mlflow.genai.judges.optimizers.dspy.make_judge", return_value=Mock()),
        patch.object(optimizer, "_dspy_optimize", mock_dspy_optimize),
    ):
        optimizer.align(mock_judge, mock_traces)

        # Verify suppression was active during the DSPy optimization call
        assert suppression_state_during_call["set_verbose"] is False
        assert suppression_state_during_call["suppress_debug_info"] is True

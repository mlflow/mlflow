"""Tests for DSPyAlignmentOptimizer base class."""

from typing import Any, Callable, Collection
from unittest.mock import MagicMock, Mock, patch

import dspy
import litellm
import pytest

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.genai.judges.optimizers.dspy_utils import AgentEvalLM, convert_mlflow_uri_to_litellm

from tests.genai.judges.optimizers.conftest import MockDSPyLM, MockJudge


def create_mock_make_judge(expected_model=None, track_calls=None):
    """Create a mock make_judge function for testing.

    Args:
        expected_model: If provided, track calls with this model to track_calls list
        track_calls: List to append model to when make_judge is called with expected_model

    Returns:
        Mock function that can be used to patch make_judge
    """
    mock_feedback = MagicMock()
    mock_feedback.value = "pass"
    mock_feedback.rationale = "Test rationale"

    def mock_make_judge(name, instructions, model):
        # Track which model was used if tracking is enabled
        if track_calls is not None and model == expected_model:
            track_calls.append(model)
        return MagicMock(return_value=mock_feedback)

    return mock_make_judge


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
    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")

    with patch("dspy.LM", MagicMock()):
        # Setup concrete optimizer
        optimizer = ConcreteDSPyOptimizer()

        # Mock get_min_traces_required to work with 5 traces from fixture
        with patch.object(ConcreteDSPyOptimizer, "get_min_traces_required", return_value=5):
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

    # Use the utility function to create mock make_judge
    mock_make_judge = create_mock_make_judge(
        expected_model=judge_model, track_calls=judge_lm.context_calls
    )

    # Direct patching approach: patch only LM, use real DSPy otherwise
    with (
        patch.object(dspy, "LM", side_effect=mock_lm_factory),
        patch("mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=mock_make_judge),
    ):
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
                # Note: Must provide outputs parameter as it's required
                program(inputs=examples[0].inputs, outputs=examples[0].outputs)

                # Return optimized program as usual
                return super()._dspy_optimize(program, examples, metric_fn)

        # Create optimizer with different model
        optimizer = TestDSPyOptimizer(model=optimizer_model)

        # Run alignment with mocked min traces requirement
        with patch.object(TestDSPyOptimizer, "get_min_traces_required", return_value=5):
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
        # Mock get_min_traces_required to work with 5 traces from fixture
        with patch.object(ConcreteDSPyOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

    # Check that URIs were converted to LiteLLM format (slash instead of colon-slash)
    assert lm_calls == ["anthropic/claude-3.5-sonnet"]


def test_mlflow_to_litellm_uri_conversion_in_judge_program():
    """Test that judge's model URI is converted when creating DSPy program."""
    # Create mock judge with MLflow URI format
    mock_judge = MockJudge(name="test_judge", model="openai:/gpt-4o-mini")

    optimizer = ConcreteDSPyOptimizer()

    # Track what models are passed to make_judge
    make_judge_calls = []
    mock_make_judge = create_mock_make_judge(
        expected_model=mock_judge.model, track_calls=make_judge_calls
    )

    # Create the program
    program = optimizer._get_dspy_program_from_judge(mock_judge)

    # Call the program to trigger judge creation
    with patch("mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=mock_make_judge):
        # Create a mock dspy.LM with the expected converted model
        mock_lm = MagicMock()
        mock_lm.model = convert_mlflow_uri_to_litellm(mock_judge.model)

        # Call the program with the mock LM
        program.forward(inputs="test", outputs="test", lm=mock_lm)

    # Verify that make_judge was called with the converted URI
    assert len(make_judge_calls) == 1
    assert make_judge_calls[0] == mock_judge.model  # Should use original MLflow URI format


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

    optimizer = ConcreteDSPyOptimizer()
    min_traces = optimizer.get_min_traces_required()
    mock_traces = [Mock(spec=Trace) for _ in range(min_traces)]
    mock_judge = MockJudge(name="test_judge", model="openai:/gpt-4o-mini")

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


def test_align_configures_databricks_lm_in_context(sample_traces_with_assessments):
    """Test that align method configures AgentEvalLM in dspy.settings when using databricks model"""
    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")
    optimizer = ConcreteDSPyOptimizer(model="databricks")

    # This is necessary because the LM is set in a specific context within `align`
    def check_context(*args, **kwargs):
        assert isinstance(dspy.settings["lm"], AgentEvalLM)
        return MagicMock()

    with (
        patch("mlflow.genai.judges.optimizers.dspy.make_judge", return_value=MagicMock()),
        patch.object(optimizer, "_dspy_optimize", side_effect=check_context),
        patch.object(optimizer, "get_min_traces_required", return_value=0),
    ):
        optimizer.align(mock_judge, sample_traces_with_assessments)


def test_align_configures_openai_lm_in_context(sample_traces_with_assessments):
    """Test that align method configures dspy.LM in dspy.settings when using OpenAI model."""
    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")
    optimizer = ConcreteDSPyOptimizer(model="openai:/gpt-4.1")

    # This is necessary because the LM is set in a specific context within `align`
    def check_context(*args, **kwargs):
        assert isinstance(dspy.settings["lm"], dspy.LM)
        assert dspy.settings["lm"].model == "openai/gpt-4.1"
        return MagicMock()

    with (
        patch(
            "mlflow.genai.judges.optimizers.dspy.trace_to_dspy_example", return_value=MagicMock()
        ),
        patch("mlflow.genai.judges.optimizers.dspy.make_judge", return_value=MagicMock()),
        patch.object(optimizer, "_dspy_optimize", side_effect=check_context),
        patch.object(optimizer, "get_min_traces_required", return_value=0),
    ):
        optimizer.align(mock_judge, sample_traces_with_assessments)


@pytest.mark.parametrize(
    ("lm_value", "lm_model", "expected_judge_model", "test_description"),
    [
        (None, None, "openai:/gpt-4", "No lm parameter - should use original judge model"),
        (
            "mock_lm",
            "anthropic/claude-3",
            "anthropic:/claude-3",
            "Regular model - should convert from LiteLLM to MLflow format",
        ),
        (
            "mock_lm",
            "databricks",
            "databricks",
            "Databricks default - should use directly without conversion",
        ),
    ],
)
def test_custom_predict_forward_lm_parameter_handling(
    lm_value, lm_model, expected_judge_model, test_description
):
    """Test that CustomPredict.forward handles the lm parameter correctly in all cases.

    Args:
        lm_value: Whether to pass an lm parameter (None or "mock_lm")
        lm_model: The model string to set on the mock lm object
        expected_judge_model: The expected model that should be passed to make_judge
        test_description: Description of what this test case is testing
    """
    from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL

    # Ensure databricks constant matches our test expectation
    assert _DATABRICKS_DEFAULT_JUDGE_MODEL == "databricks"

    # Create a mock judge with a default model
    original_judge_model = "openai:/gpt-4"
    mock_judge = MockJudge(name="test_judge", model=original_judge_model)

    # Create optimizer and program
    optimizer = ConcreteDSPyOptimizer()
    program = optimizer._get_dspy_program_from_judge(mock_judge)

    # Track what models are passed to make_judge
    make_judge_calls = []

    def track_make_judge(name, instructions, model):
        make_judge_calls.append(model)
        mock_feedback = MagicMock()
        mock_feedback.value = "pass"
        mock_feedback.rationale = "Test"
        return MagicMock(return_value=mock_feedback)

    with patch("mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=track_make_judge):
        # Prepare the lm parameter based on test case
        kwargs = {"inputs": "test", "outputs": "test"}
        if lm_value == "mock_lm":
            mock_lm = MagicMock()
            mock_lm.model = lm_model
            kwargs["lm"] = mock_lm

        # Call forward with or without lm parameter
        program.forward(**kwargs)

        # Verify the correct model was passed to make_judge
        assert len(make_judge_calls) == 1, (
            f"Expected 1 call to make_judge, got {len(make_judge_calls)}"
        )
        assert make_judge_calls[0] == expected_judge_model, (
            f"{test_description}: Expected {expected_judge_model}, got {make_judge_calls[0]}"
        )

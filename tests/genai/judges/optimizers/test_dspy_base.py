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


def make_judge_mock_builder(
    expected_model: str | None = None, track_calls: list[str] | None = None
) -> Callable[[str, str, str], MagicMock]:
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
    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")

    with patch("dspy.LM", MagicMock()):
        optimizer = ConcreteDSPyOptimizer()

        with patch.object(ConcreteDSPyOptimizer, "get_min_traces_required", return_value=5):
            result = optimizer.align(mock_judge, sample_traces_with_assessments)

    assert result is not None
    assert result.model == mock_judge.model
    assert "Optimized instructions with {{inputs}} and {{outputs}}" in result.instructions


def test_align_no_traces(mock_judge):
    """Test alignment with no traces provided."""
    optimizer = ConcreteDSPyOptimizer()

    with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
        optimizer.align(mock_judge, [])

    assert "No traces provided" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "No traces provided" in str(exc_info.value.__cause__)


def test_align_no_valid_examples(mock_judge, sample_trace_without_assessment):
    """Test alignment when no valid examples can be created."""
    with patch("dspy.LM", MagicMock()):
        optimizer = ConcreteDSPyOptimizer()
        with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
            optimizer.align(mock_judge, [sample_trace_without_assessment])

        assert "No valid examples could be created" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert "No valid examples could be created" in str(exc_info.value.__cause__)


def test_align_insufficient_examples(mock_judge, sample_trace_with_assessment):
    """Test alignment with insufficient examples."""
    optimizer = ConcreteDSPyOptimizer()
    min_traces = optimizer.get_min_traces_required()

    with patch("dspy.LM", MagicMock()):
        with pytest.raises(MlflowException, match="Alignment optimization failed") as exc_info:
            optimizer.align(mock_judge, [sample_trace_with_assessment])

        assert f"At least {min_traces} valid traces are required" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert f"At least {min_traces} valid traces are required" in str(exc_info.value.__cause__)


def _create_mock_dspy_lm_factory(optimizer_lm, judge_lm):
    """Factory function to create MockDSPyLM instances that track calls to LMs."""

    def mock_lm_factory(model=None, **kwargs):
        """Internal factory method to carry the input models"""
        if model == optimizer_lm.model:
            return optimizer_lm
        elif model == judge_lm.model:
            return judge_lm
        else:
            raise ValueError(f"Invalid model: {model}")

    return mock_lm_factory


def test_optimizer_and_judge_use_different_models(sample_traces_with_assessments):
    """Test that optimizer uses its own model while judge program uses judge's model."""
    judge_model = "openai:/gpt-4"
    optimizer_model = "anthropic:/claude-3"

    mock_judge = MockJudge(name="mock_judge", model=judge_model)
    traces = sample_traces_with_assessments

    optimizer_lm = MockDSPyLM(convert_mlflow_uri_to_litellm(optimizer_model))
    judge_lm = MockDSPyLM(convert_mlflow_uri_to_litellm(judge_model))

    mock_lm_factory = _create_mock_dspy_lm_factory(optimizer_lm, judge_lm)
    mock_make_judge = make_judge_mock_builder(
        expected_model=judge_model, track_calls=judge_lm.context_calls
    )

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

                program(inputs=examples[0].inputs, outputs=examples[0].outputs)

                return super()._dspy_optimize(program, examples, metric_fn)

        optimizer = TestDSPyOptimizer(model=optimizer_model)

        with patch.object(TestDSPyOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, traces)

        assert len(judge_lm.context_calls) > 0, (
            f"Expected judge LM to be called, but got {len(judge_lm.context_calls)} calls. "
            f"Optimizer calls: {len(optimizer_lm.context_calls)}"
        )

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

    lm_calls = []

    def mock_lm_init(model=None, **kwargs):
        lm_calls.append(model)
        return MagicMock()

    with patch("dspy.LM", side_effect=mock_lm_init):
        optimizer = ConcreteDSPyOptimizer(model=optimizer_model)
        with patch.object(ConcreteDSPyOptimizer, "get_min_traces_required", return_value=5):
            optimizer.align(mock_judge, sample_traces_with_assessments)

    assert lm_calls == ["anthropic/claude-3.5-sonnet"]


def test_mlflow_to_litellm_uri_conversion_in_judge_program():
    """Test that judge's model URI is converted when creating DSPy program."""
    mock_judge = MockJudge(name="test_judge", model="openai:/gpt-4o-mini")

    optimizer = ConcreteDSPyOptimizer()

    make_judge_calls = []
    mock_make_judge = make_judge_mock_builder(
        expected_model=mock_judge.model, track_calls=make_judge_calls
    )

    program = optimizer._get_dspy_program_from_judge(mock_judge)

    with patch("mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=mock_make_judge):
        mock_lm = MagicMock()
        mock_lm.model = convert_mlflow_uri_to_litellm(mock_judge.model)

        program.forward(inputs="test", outputs="test", lm=mock_lm)

    assert len(make_judge_calls) == 1
    assert make_judge_calls[0] == mock_judge.model


def test_dspy_align_litellm_nonfatal_error_messages_suppressed():
    """Test that LiteLLM nonfatal error messages are suppressed during DSPy align method."""
    suppression_state_during_call = {}

    def mock_dspy_optimize(program, examples, metric_fn):
        suppression_state_during_call["set_verbose"] = litellm.set_verbose
        suppression_state_during_call["suppress_debug_info"] = litellm.suppress_debug_info

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

        assert suppression_state_during_call["set_verbose"] is False
        assert suppression_state_during_call["suppress_debug_info"] is True


def test_align_configures_databricks_lm_in_context(sample_traces_with_assessments):
    """Test that align method configures AgentEvalLM in dspy.settings when using databricks model"""
    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")
    optimizer = ConcreteDSPyOptimizer(model="databricks")

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
    ("lm_value", "lm_model", "expected_judge_model"),
    [
        (None, None, "openai:/gpt-4"),
        ("mock_lm", "anthropic/claude-3", "anthropic:/claude-3"),
        ("mock_lm", "databricks", "databricks"),
    ],
)
def test_dspy_program_forward_lm_parameter_handling(lm_value, lm_model, expected_judge_model):
    original_judge_model = "openai:/gpt-4"
    mock_judge = MockJudge(name="test_judge", model=original_judge_model)

    optimizer = ConcreteDSPyOptimizer()
    program = optimizer._get_dspy_program_from_judge(mock_judge)

    make_judge_calls = []
    captured_args = {}

    def track_make_judge(name, instructions, model):
        make_judge_calls.append(model)
        captured_args["name"] = name
        captured_args["instructions"] = instructions
        mock_feedback = MagicMock()
        mock_feedback.value = "pass"
        mock_feedback.rationale = "Test"
        return MagicMock(return_value=mock_feedback)

    with patch("mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=track_make_judge):
        kwargs = {"inputs": "test", "outputs": "test"}
        if lm_value == "mock_lm":
            mock_lm = MagicMock()
            mock_lm.model = lm_model
            kwargs["lm"] = mock_lm

        program.forward(**kwargs)

        assert len(make_judge_calls) == 1
        assert make_judge_calls[0] == expected_judge_model
        # Verify instructions from judge are passed through
        assert captured_args["name"] == "test_judge"
        assert captured_args["instructions"] == mock_judge.instructions


def test_dspy_program_uses_make_judge_with_optimized_instructions(sample_traces_with_assessments):
    """Test that CustomPredict uses optimized instructions after DSPy optimization."""
    original_instructions = (
        "Original judge instructions for evaluation of {{inputs}} and {{outputs}}"
    )
    optimized_instructions = (
        "Optimized instructions after DSPy alignment for {{inputs}} and {{outputs}}"
    )
    mock_judge = MockJudge(
        name="mock_judge", model="openai:/gpt-4", instructions=original_instructions
    )
    captured_instructions = None

    def capture_make_judge(name, instructions, model):
        nonlocal captured_instructions
        captured_instructions = instructions
        mock_feedback = MagicMock()
        mock_feedback.value = "pass"
        mock_feedback.rationale = "Test"
        return MagicMock(return_value=mock_feedback)

    class TestOptimizer(ConcreteDSPyOptimizer):
        def _dspy_optimize(self, program, examples, metric_fn):
            program.signature.instructions = optimized_instructions

            with patch(
                "mlflow.genai.judges.optimizers.dspy.make_judge", side_effect=capture_make_judge
            ):
                program.forward(inputs="test input", outputs="test output")

            return program

    optimizer = TestOptimizer()
    with (
        patch("dspy.LM", MagicMock()),
        patch.object(TestOptimizer, "get_min_traces_required", return_value=5),
    ):
        optimizer.align(mock_judge, sample_traces_with_assessments)
        assert captured_instructions == optimized_instructions

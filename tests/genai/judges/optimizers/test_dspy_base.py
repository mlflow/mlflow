"""Tests for DSPyAlignmentOptimizer base class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import dspy

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer
from mlflow.genai.judges.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_text_from_data,
)
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


def test_extract_text_from_data_string():
    """Test extracting text from string data."""
    result = extract_text_from_data("simple string", "request")
    assert result == "simple string"


def test_extract_text_from_data_dict_request():
    """Test extracting request text from dictionary data."""
    data = {"prompt": "test input", "other": "ignored"}
    result = extract_text_from_data(data, "request")
    assert result == "test input"


def test_extract_text_from_data_dict_response():
    """Test extracting response text from dictionary data."""
    data = {"content": "test output", "other": "ignored"}
    result = extract_text_from_data(data, "response")
    assert result == "test output"


def test_extract_text_from_data_none():
    """Test extracting text from None data."""
    result = extract_text_from_data(None, "request")
    assert result == ""


@pytest.mark.parametrize(
    ("trace_fixture", "expected_content"),
    [
        ("sample_trace_with_assessment", "test input"),
        ("trace_with_nested_request_response", "nested input"),
        ("trace_with_list_request_response", "item"),
        ("trace_with_string_request_response", "capital of France"),
        ("trace_with_mixed_types", "test"),
    ],
)
def test_extract_request_from_trace(trace_fixture, expected_content, request):
    """Test extracting request from various trace types."""
    trace = request.getfixturevalue(trace_fixture)
    result = extract_request_from_trace(trace)
    assert expected_content in result


@pytest.mark.parametrize(
    ("trace_fixture", "expected_content"),
    [
        ("sample_trace_with_assessment", "test output"),
        ("trace_with_nested_request_response", "nested output"),
        ("trace_with_list_request_response", "result"),
        ("trace_with_string_request_response", "Paris"),
        ("trace_with_mixed_types", "response"),
    ],
)
def test_extract_response_from_trace(trace_fixture, expected_content, request):
    """Test extracting response from various trace types."""
    trace = request.getfixturevalue(trace_fixture)
    result = extract_response_from_trace(trace)
    assert expected_content in result


def test_align_success(sample_traces_with_assessments):
    """Test successful alignment process."""
    # Create a mock judge with model attribute
    from tests.genai.judges.optimizers.conftest import MockJudge

    mock_judge = MockJudge(name="mock_judge", model="openai:/gpt-4")

    mock_dspy = MagicMock()
    # Setup mock DSPy components
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    mock_signature = Mock()
    mock_dspy.make_signature.return_value = mock_signature

    mock_program = Mock()
    mock_program.signature = Mock()
    mock_program.signature.instructions = "Optimized instructions with {{inputs}} and {{outputs}}"
    mock_dspy.Predict.return_value = mock_program

    # Mock context manager
    mock_dspy.context.return_value.__enter__ = Mock(return_value=None)
    mock_dspy.context.return_value.__exit__ = Mock(return_value=None)

    # Mock LM
    mock_dspy.LM.return_value = MagicMock()

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.optimizers.dspy.make_judge") as mock_make_judge:
            # Mock the optimized judge
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "mock_judge_optimized"
            mock_make_judge.return_value = mock_optimized_judge

            # Setup concrete optimizer
            optimizer = ConcreteDSPyOptimizer()

            result = optimizer.align(mock_judge, sample_traces_with_assessments)

    # Should return an optimized judge
    assert result is not None
    assert result == mock_optimized_judge

    # Verify make_judge was called with correct parameters (from the _create_optimized_judge test)
    mock_make_judge.assert_called_once_with(
        name="mock_judge",
        instructions="Optimized instructions with {{inputs}} and {{outputs}}",
        model="openai:/gpt-4",
    )


def test_align_no_traces(mock_judge):
    """Test alignment with no traces provided."""
    optimizer = ConcreteDSPyOptimizer()

    with pytest.raises(MlflowException, match="No traces provided"):
        optimizer.align(mock_judge, [])


def test_align_no_valid_examples(mock_judge):
    """Test alignment when no valid examples can be created."""
    mock_dspy = MagicMock()
    # Setup DSPy mocks
    mock_dspy.Example.side_effect = Exception("Failed to create example")

    # Create trace without proper assessment
    mock_trace = Mock()
    mock_trace.info.assessments = []
    mock_trace.info.request_preview = "test"
    mock_trace.info.response_preview = "test"
    mock_trace.data.request = "test"
    mock_trace.data.response = "test"

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        optimizer = ConcreteDSPyOptimizer()

        with pytest.raises(MlflowException, match="No valid examples could be created"):
            optimizer.align(mock_judge, [mock_trace])


def test_align_insufficient_examples(mock_judge):
    """Test alignment with insufficient examples."""
    optimizer = ConcreteDSPyOptimizer()

    # Mock dspy first to avoid import errors
    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example
    mock_dspy.LM.return_value = MagicMock()
    mock_dspy.Predict.return_value = MagicMock()
    mock_dspy.context.return_value.__enter__ = Mock(return_value=None)
    mock_dspy.context.return_value.__exit__ = Mock(return_value=None)

    mock_signature = Mock()

    # Mock the create_dspy_signature and trace_to_dspy_example functions
    with patch("mlflow.genai.judges.optimizers.dspy.create_dspy_signature") as mock_create_sig:
        mock_create_sig.return_value = mock_signature
        with patch(
            "mlflow.genai.judges.optimizers.dspy.trace_to_dspy_example"
        ) as mock_trace_convert:
            # Return a valid example (not None) so we get past the "no valid examples" check
            mock_trace_convert.return_value = mock_example

            with patch.dict("sys.modules", {"dspy": mock_dspy}):
                # Create a single trace - should result in 1 valid example
                mock_trace = Mock()

                with pytest.raises(MlflowException, match="At least 2 valid examples are required"):
                    optimizer.align(mock_judge, [mock_trace])


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
        # Mock the optimized judge creation to avoid extra calls to the mocked LM
        with patch("mlflow.genai.judges.optimizers.dspy.make_judge") as mock_make_judge:
            
            # Override ConcreteDSPyOptimizer's _dspy_optimize to call the program
            class TestDSPyOptimizer(ConcreteDSPyOptimizer):
                def _dspy_optimize(self, program, examples, metric_fn):
                    # Simulate calling the program (which represents the judge)
                    # This should happen with the judge's model context
                    # We need to recreate the judge model context here
                    
                    lm_in_context = dspy.settings.lm
                    assert lm_in_context == optimizer_lm

                    program(inputs=examples[0].inputs)
                    
                    # Return optimized program as usual
                    return super()._dspy_optimize(program, examples, metric_fn)

            # Create optimizer with different model
            optimizer = TestDSPyOptimizer(model=optimizer_model)

            # Run alignment
            result = optimizer.align(mock_judge, traces)

            # Verify that the judge's LM was actually called during program execution
            # This ensures that the program call used the judge's model
            assert len(judge_lm.context_calls) > 0, (
                f"Expected judge LM to be called, but got {len(judge_lm.context_calls)} calls. "
                f"Optimizer calls: {len(optimizer_lm.context_calls)}"
            )

            # Verify that the optimizer's LM was not called
            assert len(optimizer_lm.context_calls) == 0, (
                f"Expected optimizer LM to not be called, but got {len(optimizer_lm.context_calls)} calls. "
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

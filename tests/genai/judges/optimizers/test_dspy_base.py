"""Tests for DSPyAlignmentOptimizer base class."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.dspy import DSPyAlignmentOptimizer


class ConcreteDSPyOptimizer(DSPyAlignmentOptimizer):
    """Concrete implementation for testing."""

    def _dspy_optimize(self, program, examples, metric_fn):
        # Mock implementation for testing
        mock_program = Mock()
        mock_program.signature = Mock()
        mock_program.signature.instructions = "Optimized instructions from DSPy"
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
    optimizer = ConcreteDSPyOptimizer()
    result = optimizer._extract_text_from_data("simple string", "request")
    assert result == "simple string"


def test_extract_text_from_data_dict_request():
    """Test extracting request text from dictionary data."""
    optimizer = ConcreteDSPyOptimizer()
    data = {"inputs": "test input", "other": "ignored"}
    result = optimizer._extract_text_from_data(data, "request")
    assert result == "test input"


def test_extract_text_from_data_dict_response():
    """Test extracting response text from dictionary data."""
    optimizer = ConcreteDSPyOptimizer()
    data = {"outputs": "test output", "other": "ignored"}
    result = optimizer._extract_text_from_data(data, "response")
    assert result == "test output"


def test_extract_text_from_data_none():
    """Test extracting text from None data."""
    optimizer = ConcreteDSPyOptimizer()
    result = optimizer._extract_text_from_data(None, "request")
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
    optimizer = ConcreteDSPyOptimizer()
    result = optimizer._extract_request_from_trace(trace)
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
    optimizer = ConcreteDSPyOptimizer()
    result = optimizer._extract_response_from_trace(trace)
    assert expected_content in result


def test_sanitize_judge_name():
    """Test judge name sanitization."""
    optimizer = ConcreteDSPyOptimizer()
    assert optimizer._sanitize_judge_name("  Test Judge  ") == "test judge"
    assert optimizer._sanitize_judge_name("UPPERCASE") == "uppercase"


@pytest.mark.parametrize(
    "trace_fixture",
    [
        "sample_trace_with_assessment",
        "trace_with_nested_request_response",
        "trace_with_list_request_response",
        "trace_with_string_request_response",
        "trace_with_mixed_types",
    ],
)
def test_trace_to_dspy_example_success(trace_fixture, request):
    """Test successful conversion of various trace types to DSPy example."""
    trace = request.getfixturevalue(trace_fixture)
    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._trace_to_dspy_example(trace, "mock_judge")

    assert result is not None
    mock_dspy.Example.assert_called_once()
    mock_example.with_inputs.assert_called_once_with("inputs", "outputs")


def test_trace_to_dspy_example_no_assessment():
    """Test trace conversion with no matching assessment."""
    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_dspy.Example.return_value = mock_example

    # Create trace without assessments
    mock_trace = Mock()
    mock_trace.info.trace_id = "test"
    mock_trace.info.assessments = []
    mock_trace.info.request_preview = "test"
    mock_trace.info.response_preview = "test"
    mock_trace.data.request = "test"
    mock_trace.data.response = "test"

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        optimizer = ConcreteDSPyOptimizer()
        result = optimizer._trace_to_dspy_example(mock_trace, "mock_judge")

    assert result is None


def test_trace_to_dspy_example_no_dspy():
    """Test trace conversion when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = ConcreteDSPyOptimizer()
        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer._trace_to_dspy_example(Mock(), "judge")


def test_extract_judge_instructions(mock_judge):
    """Test extracting instructions from judge."""
    optimizer = ConcreteDSPyOptimizer()
    result = optimizer._extract_judge_instructions(mock_judge)
    assert result == mock_judge.description


def test_create_dspy_signature():
    """Test creating DSPy signature."""
    mock_dspy = MagicMock()
    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        optimizer = ConcreteDSPyOptimizer()
        instructions = "Test instructions"

        optimizer._create_dspy_signature(instructions)

    mock_dspy.make_signature.assert_called_once()
    args, kwargs = mock_dspy.make_signature.call_args
    assert args[1] == instructions  # instructions passed as second argument


def test_create_dspy_signature_no_dspy():
    """Test signature creation when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = ConcreteDSPyOptimizer()
        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer._create_dspy_signature("test")


def test_create_agreement_metric():
    """Test creating agreement metric function."""
    optimizer = ConcreteDSPyOptimizer()
    metric_fn = optimizer._create_agreement_metric()

    # Test metric with matching results
    example = Mock()
    example.result = "pass"
    pred = Mock()
    pred.result = "pass"

    assert metric_fn(example, pred) == 1.0

    # Test metric with different results
    pred.result = "fail"
    assert metric_fn(example, pred) == 0.0


def test_create_agreement_metric_error_handling():
    """Test agreement metric error handling."""
    optimizer = ConcreteDSPyOptimizer()
    metric_fn = optimizer._create_agreement_metric()

    # Test with invalid inputs
    result = metric_fn(None, None)
    assert result == 0.0


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
    mock_program.signature.instructions = "Optimized instructions from DSPy"
    mock_dspy.Predict.return_value = mock_program

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
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
        name="mock_judge", instructions="Optimized instructions from DSPy", model="openai:/gpt-4"
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

    # Mock _trace_to_dspy_example to return only one example
    with patch.object(optimizer, "_trace_to_dspy_example") as mock_trace_convert:
        mock_trace_convert.return_value = Mock()  # Only one example

        mock_trace = Mock()

        with pytest.raises(MlflowException, match="At least 2 valid examples are required"):
            optimizer.align(mock_judge, [mock_trace])


def test_align_no_dspy(mock_judge, sample_traces_with_assessments):
    """Test alignment when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        optimizer = ConcreteDSPyOptimizer()

        with pytest.raises(MlflowException, match="DSPy library is required"):
            optimizer.align(mock_judge, sample_traces_with_assessments)


def _create_traces_with_judge_assessments(judge_name, num_traces=3):
    """Helper to create mock traces with assessments for a specific judge."""
    traces = []
    for i in range(num_traces):
        # Mock assessment with matching judge name
        mock_assessment = Mock()
        mock_assessment.name = judge_name
        mock_assessment.source.source_type = "HUMAN"
        mock_assessment.feedback.value = "pass" if i % 2 == 0 else "fail"
        mock_assessment.rationale = f"Rationale for trace {i}"

        # Mock trace info
        mock_trace_info = Mock()
        mock_trace_info.trace_id = f"test_trace_{i:03d}"
        mock_trace_info.assessments = [mock_assessment]
        mock_trace_info.request_preview = f'{{"inputs": "test input {i}"}}'
        mock_trace_info.response_preview = f'{{"outputs": "test output {i}"}}'

        # Mock trace data
        mock_trace_data = Mock()
        mock_trace_data.request = {"inputs": f"test input {i}"}
        mock_trace_data.response = {"outputs": f"test output {i}"}

        # Mock trace
        mock_trace = Mock()
        mock_trace.info = mock_trace_info
        mock_trace.data = mock_trace_data

        traces.append(mock_trace)
    return traces


def test_optimizer_and_judge_use_different_models():
    """Test that optimizer uses its own model while judge program uses judge's model."""
    from tests.genai.judges.optimizers.conftest import MockJudge

    # Setup models
    judge_model = "openai:/gpt-4"
    optimizer_model = "anthropic:/claude-3"

    # Create judge and traces
    mock_judge = MockJudge(name="test_judge", model=judge_model)
    traces = _create_traces_with_judge_assessments("test_judge")

    # Mock DSPy components
    mock_dspy = MagicMock()
    mock_example = Mock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    mock_signature = Mock()
    mock_dspy.make_signature.return_value = mock_signature

    # Create a mock program that we can track calls to
    mock_program = Mock()
    mock_program.signature = Mock()
    mock_program.signature.instructions = "Optimized instructions"

    # Track what model is used when the program is called
    program_call_model = None

    def program_call_tracker(*args, **kwargs):
        # Capture the current DSPy context's model when program is called
        nonlocal program_call_model
        # In real DSPy, the program would use the model from dspy.settings
        # We simulate this by checking what model was set in the context
        program_call_model = getattr(mock_dspy.settings, "lm", None)
        return Mock(result="pass", rationale="test")

    mock_program.side_effect = program_call_tracker
    mock_dspy.Predict.return_value = mock_program

    # Mock DSPy settings to track model changes
    mock_dspy.settings = Mock()
    mock_dspy.settings.configure = Mock()

    # Mock context manager for DSPy
    mock_context = MagicMock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=None)

    def context_side_effect(lm=None, **kwargs):
        # When context is entered, set the model in settings
        if lm:
            mock_dspy.settings.lm = lm
        return mock_context

    mock_dspy.context = Mock(side_effect=context_side_effect)

    # Mock LiteLLM for both optimizer and judge models
    mock_optimizer_lm = Mock(name="optimizer_lm")
    mock_judge_lm = Mock(name="judge_lm")

    def litellm_side_effect(model=None, **kwargs):
        if "anthropic" in model:
            return mock_optimizer_lm
        elif "openai" in model:
            return mock_judge_lm
        return Mock()

    # Mock both LiteLLM and LM (DSPy uses both)
    mock_dspy.LiteLLM = Mock(side_effect=litellm_side_effect)
    mock_dspy.LM = Mock(side_effect=litellm_side_effect)

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        with patch("mlflow.genai.judges.make_judge") as mock_make_judge:
            # Mock the optimized judge
            mock_optimized_judge = Mock()
            mock_optimized_judge.name = "test_judge_optimized"
            mock_optimized_judge.model = judge_model
            mock_make_judge.return_value = mock_optimized_judge

            # Override ConcreteDSPyOptimizer's _dspy_optimize to call the program
            class TestDSPyOptimizer(ConcreteDSPyOptimizer):
                def _dspy_optimize(self, program, examples, metric_fn):
                    # Simulate calling the program (which represents the judge)
                    # This should happen with the judge's model context
                    for example in examples[:1]:  # Test with at least one example
                        # The program call simulates judge evaluation
                        program(inputs=example.inputs, outputs=example.outputs)

                    # Return optimized program as usual
                    return super()._dspy_optimize(program, examples, metric_fn)

            # Create optimizer with different model
            optimizer = TestDSPyOptimizer(model=optimizer_model)

            # Verify optimizer has the correct model
            assert optimizer._model == optimizer_model

            # Verify judge model before alignment
            assert mock_judge.model == judge_model

            # Run alignment
            result = optimizer.align(mock_judge, traces)

            # Verify that when the program (judge simulation) was called during optimization,
            # it used the judge's model context (not the optimizer's model)
            assert program_call_model == mock_judge_lm, (
                f"Program should use judge's model context, but got {program_call_model}"
            )

            # Verify DSPy.LM was called with both models
            assert mock_dspy.LM.call_count >= 2
            model_calls = [
                call[1].get("model") or call[0][0]
                for call in mock_dspy.LM.call_args_list
                if call[0] or call[1].get("model")
            ]
            assert any("anthropic" in str(m) for m in model_calls), "Optimizer model should be used"
            assert any("openai" in str(m) for m in model_calls), "Judge model should be used"

            # Verify judge model hasn't changed after alignment
            assert mock_judge.model == judge_model

            # Verify the returned optimized judge uses the original judge's model
            mock_make_judge.assert_called_once_with(
                name="test_judge",
                instructions="Optimized instructions from DSPy",
                model=judge_model,
            )

            # Verify result is the optimized judge with correct model
            assert result == mock_optimized_judge
            assert result.model == judge_model

"""Tests for DSPy utility functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    AgentEvalLM,
    agreement_metric,
    construct_dspy_lm,
    convert_litellm_to_mlflow_uri,
    convert_mlflow_uri_to_litellm,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)

from tests.genai.judges.optimizers.conftest import MockJudge


def test_sanitize_judge_name(sample_trace_with_assessment, mock_judge):
    """Test judge name sanitization in trace_to_dspy_example."""
    # The sanitization is now done inside trace_to_dspy_example
    # Test that it correctly handles different judge name formats

    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        judge1 = MockJudge(name="  mock_judge  ")
        judge2 = MockJudge(name="Mock_Judge")
        judge3 = MockJudge(name="MOCK_JUDGE")
        assert trace_to_dspy_example(sample_trace_with_assessment, judge1) is not None
        assert trace_to_dspy_example(sample_trace_with_assessment, judge2) is not None
        assert trace_to_dspy_example(sample_trace_with_assessment, judge3) is not None


def test_trace_to_dspy_example_two_human_assessments(trace_with_two_human_assessments, mock_judge):
    """Test that most recent HUMAN assessment is used when there are multiple HUMAN assessments."""
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    trace = trace_with_two_human_assessments
    result = trace_to_dspy_example(trace, mock_judge)

    assert isinstance(result, dspy.Example)
    # Should use the newer assessment with value="pass" and specific rationale
    assert result["result"] == "pass"
    assert result["rationale"] == "Second assessment - should be used (more recent)"


def test_trace_to_dspy_example_human_vs_llm_priority(
    trace_with_human_and_llm_assessments, mock_judge
):
    """Test that HUMAN assessment is prioritized over LLM_JUDGE even when LLM_JUDGE is newer."""
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    trace = trace_with_human_and_llm_assessments
    result = trace_to_dspy_example(trace, mock_judge)

    assert isinstance(result, dspy.Example)
    # Should use the HUMAN assessment despite being older
    assert result["result"] == "fail"
    assert result["rationale"] == "Human assessment - should be prioritized"


@pytest.mark.parametrize(
    ("trace_fixture", "required_fields", "expected_inputs"),
    [
        # Test different combinations of required fields
        ("sample_trace_with_assessment", ["inputs"], ["inputs"]),
        ("sample_trace_with_assessment", ["outputs"], ["outputs"]),
        ("sample_trace_with_assessment", ["inputs", "outputs"], ["inputs", "outputs"]),
        (
            "sample_trace_with_assessment",
            ["trace", "inputs", "outputs"],
            ["trace", "inputs", "outputs"],
        ),
        ("trace_with_expectations", ["expectations"], ["expectations"]),
        ("trace_with_expectations", ["inputs", "expectations"], ["inputs", "expectations"]),
        ("trace_with_expectations", ["outputs", "expectations"], ["outputs", "expectations"]),
        (
            "trace_with_expectations",
            ["inputs", "outputs", "expectations"],
            ["inputs", "outputs", "expectations"],
        ),
        (
            "trace_with_expectations",
            ["trace", "inputs", "outputs", "expectations"],
            ["trace", "inputs", "outputs", "expectations"],
        ),
    ],
)
def test_trace_to_dspy_example_success(request, trace_fixture, required_fields, expected_inputs):
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    trace = request.getfixturevalue(trace_fixture)

    class TestJudge(MockJudge):
        def __init__(self, fields):
            super().__init__(name="mock_judge")
            self._fields = fields

        def get_input_fields(self):
            return [JudgeField(name=field, description=f"Test {field}") for field in self._fields]

    judge = TestJudge(required_fields)

    # Use real DSPy since we've skipped if it's not available
    result = trace_to_dspy_example(trace, judge)

    assert isinstance(result, dspy.Example)

    # Build expected kwargs based on required fields
    expected_kwargs = {}
    if "trace" in required_fields:
        expected_kwargs["trace"] = trace
    if "inputs" in required_fields:
        expected_kwargs["inputs"] = extract_request_from_trace(trace)
    if "outputs" in required_fields:
        expected_kwargs["outputs"] = extract_response_from_trace(trace)
    if "expectations" in required_fields:
        expected_kwargs["expectations"] = extract_expectations_from_trace(trace)

    # Determine expected rationale based on fixture
    if trace_fixture == "trace_with_expectations":
        expected_rationale = "Meets expectations"
    else:
        expected_rationale = "This looks good"

    # Construct an expected example and assert that the result is the same
    expected_example = dspy.Example(
        result="pass",
        rationale=expected_rationale,
        **expected_kwargs,
    ).with_inputs(*expected_inputs)

    # Compare the examples
    assert result == expected_example


@pytest.mark.parametrize(
    ("trace_fixture", "required_fields"),
    [
        ("sample_trace_with_assessment", ["expectations"]),
        ("sample_trace_with_assessment", ["inputs", "expectations"]),
        ("sample_trace_with_assessment", ["outputs", "expectations"]),
        ("sample_trace_with_assessment", ["inputs", "outputs", "expectations"]),
        ("sample_trace_with_assessment", ["trace", "inputs", "outputs", "expectations"]),
    ],
)
def test_trace_to_dspy_example_missing_required_fields(request, trace_fixture, required_fields):
    trace = request.getfixturevalue(trace_fixture)

    class TestJudge(MockJudge):
        def __init__(self, fields):
            super().__init__(name="mock_judge")
            self._fields = fields

        def get_input_fields(self):
            return [JudgeField(name=field, description=f"Test {field}") for field in self._fields]

    judge = TestJudge(required_fields)

    result = trace_to_dspy_example(trace, judge)
    assert result is None


def test_trace_to_dspy_example_no_assessment(sample_trace_without_assessment, mock_judge):
    """Test trace conversion with no matching assessment."""
    # Use the fixture for trace without assessment
    trace = sample_trace_without_assessment

    # This should return None since there's no matching assessment
    result = trace_to_dspy_example(trace, mock_judge)

    assert result is None


def test_create_dspy_signature(mock_judge):
    """Test creating DSPy signature."""
    pytest.importorskip("dspy", reason="DSPy not installed")

    signature = create_dspy_signature(mock_judge)

    assert signature.instructions == mock_judge.instructions

    judge_input_fields = mock_judge.get_input_fields()
    for field in judge_input_fields:
        assert field.name in signature.input_fields
        assert signature.input_fields[field.name].json_schema_extra["desc"] == field.description

    judge_output_fields = mock_judge.get_output_fields()
    for field in judge_output_fields:
        assert field.name in signature.output_fields
        assert signature.output_fields[field.name].json_schema_extra["desc"] == field.description


def test_agreement_metric():
    """Test agreement metric function."""
    # Test metric with matching results
    example = Mock()
    example.result = "pass"
    pred = Mock()
    pred.result = "pass"

    assert agreement_metric(example, pred) is True

    # Test metric with different results
    pred.result = "fail"
    assert agreement_metric(example, pred) is False


def test_agreement_metric_error_handling():
    """Test agreement metric error handling."""
    # Test with invalid inputs
    result = agreement_metric(None, None)
    assert result is False


@pytest.mark.parametrize(
    ("mlflow_uri", "expected_litellm_uri"),
    [
        ("openai:/gpt-4", "openai/gpt-4"),
        ("openai:/gpt-3.5-turbo", "openai/gpt-3.5-turbo"),
        ("anthropic:/claude-3", "anthropic/claude-3"),
        ("anthropic:/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet"),
        ("cohere:/command", "cohere/command"),
        ("databricks:/dbrx", "databricks/dbrx"),
    ],
)
def test_convert_mlflow_uri_to_litellm(mlflow_uri, expected_litellm_uri):
    """Test conversion of MLflow URI to LiteLLM format."""
    assert convert_mlflow_uri_to_litellm(mlflow_uri) == expected_litellm_uri


@pytest.mark.parametrize(
    "invalid_uri",
    [
        "openai-gpt-4",  # Invalid format (missing colon-slash)
        "",  # Empty string
        None,  # None value
    ],
)
def test_convert_mlflow_uri_to_litellm_invalid(invalid_uri):
    """Test conversion with invalid URIs."""
    with pytest.raises(MlflowException, match="Failed to convert MLflow URI"):
        convert_mlflow_uri_to_litellm(invalid_uri)


@pytest.mark.parametrize(
    ("litellm_model", "expected_uri"),
    [
        ("openai/gpt-4", "openai:/gpt-4"),
        ("openai/gpt-3.5-turbo", "openai:/gpt-3.5-turbo"),
        ("anthropic/claude-3", "anthropic:/claude-3"),
        ("anthropic/claude-3.5-sonnet", "anthropic:/claude-3.5-sonnet"),
        ("cohere/command", "cohere:/command"),
        ("databricks/dbrx", "databricks:/dbrx"),
    ],
)
def test_convert_litellm_to_mlflow_uri(litellm_model, expected_uri):
    result = convert_litellm_to_mlflow_uri(litellm_model)
    assert result == expected_uri


@pytest.mark.parametrize(
    "invalid_model",
    [
        "openai-gpt-4",  # Missing slash
        "",  # Empty string
        None,  # None value
        "openai/",  # Missing model name
        "/gpt-4",  # Missing provider
        "//",  # Empty provider and model
    ],
)
def test_convert_litellm_to_mlflow_uri_invalid(invalid_model):
    with pytest.raises(MlflowException, match="LiteLLM|empty|None") as exc_info:
        convert_litellm_to_mlflow_uri(invalid_model)

    if invalid_model is None or invalid_model == "":
        assert "cannot be empty or None" in str(exc_info.value)
    elif "/" not in invalid_model:
        assert "Expected format: 'provider/model'" in str(exc_info.value)


@pytest.mark.parametrize(
    "mlflow_uri",
    [
        "openai:/gpt-4",
        "anthropic:/claude-3.5-sonnet",
        "cohere:/command",
        "databricks:/dbrx",
    ],
)
def test_mlflow_to_litellm_uri_round_trip_conversion(mlflow_uri):
    # Convert MLflow -> LiteLLM
    litellm_format = convert_mlflow_uri_to_litellm(mlflow_uri)
    # Convert LiteLLM -> MLflow
    result = convert_litellm_to_mlflow_uri(litellm_format)
    # Should get back the original
    assert result == mlflow_uri, f"Round-trip failed for {mlflow_uri}"


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        ("databricks", "AgentEvalLM"),
        ("openai:/gpt-4", "dspy.LM"),
        ("anthropic:/claude-3", "dspy.LM"),
    ],
)
def test_construct_dspy_lm_utility_method(model, expected_type):
    """Test the construct_dspy_lm utility method with different model types."""
    import dspy

    result = construct_dspy_lm(model)

    if expected_type == "AgentEvalLM":
        assert isinstance(result, AgentEvalLM)
    elif expected_type == "dspy.LM":
        assert isinstance(result, dspy.LM)
        # Ensure MLflow URI format is converted (no :/ in the model)
        assert ":/" not in result.model

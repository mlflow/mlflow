"""Tests for DSPy utility functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.genai.judges.optimizers.dspy_utils import (
    agreement_metric,
    convert_mlflow_uri_to_litellm,
    create_dspy_signature,
    trace_to_dspy_example,
)


def test_sanitize_judge_name(sample_trace_with_assessment):
    """Test judge name sanitization in trace_to_dspy_example."""
    # The sanitization is now done inside trace_to_dspy_example
    # Test that it correctly handles different judge name formats
    from mlflow.genai.judges.optimizers.dspy_utils import trace_to_dspy_example

    # Mock dspy module
    mock_dspy = MagicMock()
    mock_example = MagicMock()
    mock_example.with_inputs.return_value = mock_example
    mock_dspy.Example.return_value = mock_example

    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        # Test with different case variations - they should all find the assessment
        # The assessment name in the fixture is "mock_judge" in lowercase
        assert trace_to_dspy_example(sample_trace_with_assessment, "  mock_judge  ") is not None
        assert trace_to_dspy_example(sample_trace_with_assessment, "Mock_Judge") is not None
        assert trace_to_dspy_example(sample_trace_with_assessment, "MOCK_JUDGE") is not None


def test_trace_to_dspy_example_two_human_assessments(trace_with_two_human_assessments):
    """Test that most recent HUMAN assessment is used when there are multiple HUMAN assessments."""
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    trace = trace_with_two_human_assessments
    result = trace_to_dspy_example(trace, "mock_judge")

    assert isinstance(result, dspy.Example)
    # Should use the newer assessment with value="pass" and specific rationale
    assert result["result"] == "pass"
    assert result["rationale"] == "Second assessment - should be used (more recent)"


def test_trace_to_dspy_example_human_vs_llm_priority(trace_with_human_and_llm_assessments):
    """Test that HUMAN assessment is prioritized over LLM_JUDGE even when LLM_JUDGE is newer."""
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    trace = trace_with_human_and_llm_assessments
    result = trace_to_dspy_example(trace, "mock_judge")

    assert isinstance(result, dspy.Example)
    # Should use the HUMAN assessment despite being older
    assert result["result"] == "fail"
    assert result["rationale"] == "Human assessment - should be prioritized"


def test_trace_to_dspy_example_success(sample_trace_with_assessment):
    """Test successful conversion of trace to DSPy example."""
    dspy = pytest.importorskip("dspy", reason="DSPy not installed")

    # Use the fixture directly
    trace = sample_trace_with_assessment

    # Use real DSPy since we've skipped if it's not available
    result = trace_to_dspy_example(trace, "mock_judge")

    # Assert that the result is an instance of dspy.Example
    assert isinstance(result, dspy.Example)

    # Construct an expected example and assert that the result is the same
    expected_example = dspy.Example(
        inputs=extract_request_from_trace(trace),
        outputs=extract_response_from_trace(trace),
        result="pass",
        rationale="This looks good",
    ).with_inputs("inputs", "outputs")

    # Compare the examples
    assert result == expected_example


def test_trace_to_dspy_example_no_assessment(sample_trace_without_assessment):
    """Test trace conversion with no matching assessment."""
    # Use the fixture for trace without assessment
    trace = sample_trace_without_assessment

    # This should return None since there's no matching assessment
    result = trace_to_dspy_example(trace, "mock_judge")

    assert result is None


def test_create_dspy_signature(mock_judge):
    """Test creating DSPy signature."""
    pytest.importorskip("dspy", reason="DSPy not installed")

    signature = create_dspy_signature(mock_judge)

    assert signature.instructions == mock_judge.instructions

    # Check that the input fields of the signature are the same as the input fields of the judge
    judge_input_fields = mock_judge.get_input_fields()
    for field in judge_input_fields:
        # Check that the field exists in the signature's input_fields dictionary
        assert field.name in signature.input_fields
        # Verify the field description matches
        assert signature.input_fields[field.name].json_schema_extra["desc"] == field.description

    # Check that the output fields of the signature are the same as the output fields of the judge
    judge_output_fields = mock_judge.get_output_fields()
    for field in judge_output_fields:
        # Check that the field exists in the signature's output_fields dictionary
        assert field.name in signature.output_fields
        # Verify the field description matches
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


def test_convert_mlflow_uri_to_litellm():
    """Test conversion of MLflow URI to LiteLLM format."""
    # Test OpenAI model
    assert convert_mlflow_uri_to_litellm("openai:/gpt-4") == "openai/gpt-4"
    assert convert_mlflow_uri_to_litellm("openai:/gpt-3.5-turbo") == "openai/gpt-3.5-turbo"

    # Test Anthropic model
    assert convert_mlflow_uri_to_litellm("anthropic:/claude-3") == "anthropic/claude-3"
    assert (
        convert_mlflow_uri_to_litellm("anthropic:/claude-3.5-sonnet")
        == "anthropic/claude-3.5-sonnet"
    )

    # Test other providers
    assert convert_mlflow_uri_to_litellm("cohere:/command") == "cohere/command"
    assert convert_mlflow_uri_to_litellm("databricks:/dbrx") == "databricks/dbrx"


def test_convert_mlflow_uri_to_litellm_invalid():
    """Test conversion with invalid URIs."""
    from mlflow.exceptions import MlflowException

    # Test invalid format (missing colon-slash)
    with pytest.raises(MlflowException, match="Failed to convert MLflow URI"):
        convert_mlflow_uri_to_litellm("openai-gpt-4")

    # Test empty string
    with pytest.raises(MlflowException, match="Failed to convert MLflow URI"):
        convert_mlflow_uri_to_litellm("")

    # Test None (would cause an exception in _parse_model_uri)
    with pytest.raises(MlflowException, match="Failed to convert MLflow URI"):
        convert_mlflow_uri_to_litellm(None)

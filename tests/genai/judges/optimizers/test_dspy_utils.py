"""Tests for DSPy utility functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.genai.judges.judge_trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.genai.judges.optimizers.dspy_utils import (
    agreement_metric,
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
    assert result["inputs"] == expected_example["inputs"]
    assert result["outputs"] == expected_example["outputs"]
    assert result["result"] == expected_example["result"]
    assert result["rationale"] == expected_example["rationale"]


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

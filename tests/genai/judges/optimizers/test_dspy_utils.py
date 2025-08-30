"""Tests for DSPy utility functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.exceptions import MlflowException
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
    pytest.importorskip("dspy", reason="DSPy not installed")

    # Use the fixture directly
    trace = sample_trace_with_assessment

    # Use real DSPy since we've skipped if it's not available
    result = trace_to_dspy_example(trace, "mock_judge")

    assert result is not None
    # Verify the result has the expected DSPy structure
    assert hasattr(result, "inputs")
    assert hasattr(result, "outputs")
    assert hasattr(result, "result")
    assert hasattr(result, "rationale")


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
        result = trace_to_dspy_example(mock_trace, "mock_judge")

    assert result is None


def test_trace_to_dspy_example_no_dspy():
    """Test trace conversion when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(MlflowException, match="DSPy library is required"):
            trace_to_dspy_example(Mock(), "judge")


def test_create_dspy_signature(mock_judge):
    """Test creating DSPy signature."""
    #ALKIS: Do not mock the the dspy module. Use directly the dspy Signature class.
    mock_dspy = MagicMock()
    with patch.dict("sys.modules", {"dspy": mock_dspy}):
        instructions = "Test instructions"

        create_dspy_signature(mock_judge, instructions)

    mock_dspy.make_signature.assert_called_once()
    args, kwargs = mock_dspy.make_signature.call_args
    # ALKIS: Check that the arguments are passed correctly in the make_signature call
    assert args[1] == instructions  # instructions passed as second argument


def test_create_dspy_signature_no_dspy(mock_judge):
    """Test signature creation when DSPy is not available."""
    with patch.dict("sys.modules", {"dspy": None}):
        with pytest.raises(MlflowException, match="DSPy library is required"):
            create_dspy_signature(mock_judge, "test")


def test_create_agreement_metric():
    """Test agreement metric function."""
    # Test metric with matching results
    example = Mock()
    example.result = "pass"
    pred = Mock()
    pred.result = "pass"

    assert agreement_metric(example, pred) == True

    # Test metric with different results
    pred.result = "fail"
    assert agreement_metric(example, pred) == False


def test_create_agreement_metric_error_handling():
    """Test agreement metric error handling."""
    # Test with invalid inputs
    result = agreement_metric(None, None)
    assert result == False
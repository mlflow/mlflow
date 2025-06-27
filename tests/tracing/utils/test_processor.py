from unittest.mock import patch

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.processor import validate_span_processors

from tests.tracing.helper import get_traces


@mlflow.trace
def predict(text: str):
    return "Answer: " + text


@pytest.fixture(autouse=True)
def reset_tracing_config():
    """Reset tracing configuration before each test."""
    mlflow.tracing.reset()


def test_span_processors_no_processors_configured():
    """Test that function returns early when no processors are configured."""
    mlflow.tracing.configure(span_processors=[])

    predict("test")

    span = get_traces()[0].data.spans[0]
    assert span.inputs == {"text": "test"}
    assert span.outputs == "Answer: test"


def test_span_processors_single_processor_success():
    """Test successful execution of a single processor."""

    def test_processor(span):
        span.set_outputs("overridden_output")
        span.set_attribute("test_attribute", "test_value")

    mlflow.tracing.configure(span_processors=[test_processor])

    predict("test")

    span = get_traces()[0].data.spans[0]
    assert span.inputs == {"text": "test"}
    assert span.outputs == "overridden_output"
    assert span.attributes["test_attribute"] == "test_value"


def test_apply_span_processors_multiple_processors_success():
    """Test successful execution of multiple processors in sequence."""

    def processor1(span):
        span.set_outputs("overridden_output_1")
        span.set_attribute("attr_1", "value_1")

    def processor2(span):
        span.set_outputs("overridden_output_2")
        span.set_attribute("attr_2", "value_2")

    mlflow.tracing.configure(span_processors=[processor1, processor2])

    predict("test")

    span = get_traces()[0].data.spans[0]
    assert span.inputs == {"text": "test"}
    assert span.outputs == "overridden_output_2"
    assert span.attributes["attr_1"] == "value_1"
    assert span.attributes["attr_2"] == "value_2"


@patch("mlflow.tracing.utils.processor._logger")
def test_apply_span_processors_returns_non_none_warning(mock_logger):
    """Test warning is logged when processor returns a non-None value."""

    def bad_processor(span):
        return "some_value"  # Should return nothing

    def good_processor(span):
        span.set_outputs("overridden_output")

    mlflow.tracing.configure(span_processors=[bad_processor, good_processor])

    predict("test")

    mock_logger.warning.assert_called_once()
    message = mock_logger.warning.call_args[0][0]
    assert message.startswith("Span processors ['bad_processor'] returned a non-null value")

    # Other processors should still be applied
    span = get_traces()[0].data.spans[0]
    assert span.outputs == "overridden_output"


@patch("mlflow.tracing.utils.processor._logger")
def test_apply_span_processors_exception_handling(mock_logger):
    """Test that processor exceptions are caught and logged."""

    def failing_processor(span):
        raise ValueError("Test error")

    def good_processor(span):
        span.set_outputs("overridden_output")

    mlflow.tracing.configure(span_processors=[failing_processor, good_processor])

    predict("test")

    span = get_traces()[0].data.spans[0]
    assert span.outputs == "overridden_output"
    mock_logger.warning.assert_called_once()
    message = mock_logger.warning.call_args[0][0]
    assert message.startswith("Span processor failing_processor failed")


def test_validate_span_processors_empty_input():
    assert validate_span_processors(None) == []
    assert validate_span_processors([]) == []


def test_validate_span_processors_valid_processors():
    def processor1(span):
        return None

    def processor2(span):
        return None

    result = validate_span_processors([processor1, processor2])
    assert result == [processor1, processor2]


def test_validate_span_processors_non_callable_raises_exception():
    """Test that non-callable processor raises MlflowException."""
    non_callable_processor = "not_a_function"

    with pytest.raises(MlflowException, match=r"Span processor must be"):
        validate_span_processors([non_callable_processor])


def test_validate_span_processors_invalid_arguments_raises_exception():
    """Test that processor with no arguments raises MlflowException."""

    def processor_no_args():
        return None

    with pytest.raises(MlflowException, match=r"Span processor must take"):
        validate_span_processors([processor_no_args])

    def processor_extra_args(span, extra_arg):
        return None

    with pytest.raises(MlflowException, match=r"Span processor must take"):
        validate_span_processors([processor_extra_args])

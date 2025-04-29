import logging
from unittest import mock

import mlflow
from mlflow.tracing.utils.warning import suppress_warning


def test_suppress_token_detach_warning(caplog):
    logger = logging.getLogger("opentelemetry.context")
    logger.setLevel(logging.INFO)
    logger.removeFilter(logger.filters[0])

    logger.exception("Failed to detach context")
    assert caplog.records[0].message == "Failed to detach context"
    assert caplog.records[0].levelname == "ERROR"

    suppress_warning("opentelemetry.context", "Failed to detach context")

    logger.exception("Failed to detach context")
    assert len(caplog.records) == 1  # If the log level is not debug, the log shouldn't be recorded

    logger.exception("Another error")  # Other type of error log should still be recorded
    assert len(caplog.records) == 2
    assert caplog.records[1].message == "Another error"

    # If we change the log level to debug, the log should be recorded at the debug level
    logger.setLevel(logging.DEBUG)

    logger.exception("Failed to detach context")

    assert caplog.records[2].message == "Failed to detach context"
    assert caplog.records[2].levelname == "DEBUG"


@mock.patch("mlflow.tracing.utils.warning.warnings")
def test_request_id_backward_compatible(mock_warnings):
    client = mlflow.MlflowClient()

    # Invalid usage with deprecated request_id -> warning
    parent_span = client.start_trace(name="test")
    child_span = client.start_span(
        request_id=parent_span.trace_id,
        name="child",
        parent_id=parent_span.span_id,
    )
    assert child_span.trace_id == parent_span.trace_id
    mock_warnings.warn.assert_called_once()
    warning_msg = mock_warnings.warn.call_args[0][0]
    assert "start_span" in warning_msg
    mock_warnings.reset_mock()

    client.end_span(request_id=parent_span.trace_id, span_id=child_span.span_id)
    client.end_trace(request_id=parent_span.trace_id)

    assert mock_warnings.warn.call_count == 2
    mock_warnings.reset_mock()

    # Valid usage without request_id -> no warning
    trace = mlflow.get_trace(parent_span.trace_id)
    mock_warnings.warn.assert_not_called()
    assert trace.info.trace_id == parent_span.trace_id

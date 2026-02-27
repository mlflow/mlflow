import logging
import warnings

import pytest

import mlflow
from mlflow.tracing.utils.warning import suppress_warning

from tests.tracing.helper import skip_when_testing_trace_sdk


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


def _filter_request_id_warnings(
    warnings_list: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Filter warnings to only include FutureWarning about request_id deprecation."""
    return [
        w
        for w in warnings_list
        if issubclass(w.category, FutureWarning)
        and "request_id" in str(w.message)
        and "deprecated" in str(w.message).lower()
        and "trace_id" in str(w.message)
    ]


@skip_when_testing_trace_sdk
def test_request_id_backward_compatible():
    client = mlflow.MlflowClient()

    parent_span = client.start_trace(name="test")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        child_span = client.start_span(
            request_id=parent_span.trace_id,
            name="child",
            parent_id=parent_span.span_id,
        )

        request_id_warnings = _filter_request_id_warnings(w)
        assert len(request_id_warnings) == 1
        assert child_span.trace_id == parent_span.trace_id

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        client.end_span(request_id=parent_span.trace_id, span_id=child_span.span_id)
        client.end_trace(request_id=parent_span.trace_id)

        request_id_warnings = _filter_request_id_warnings(w)
        assert len(request_id_warnings) == 2

    # Valid usage without request_id -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        trace = mlflow.get_trace(parent_span.trace_id)

        request_id_warnings = _filter_request_id_warnings(w)
        assert len(request_id_warnings) == 0
        assert trace.info.trace_id == parent_span.trace_id

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        with pytest.raises(
            ValueError,
            match=r".*",
            check=lambda e: (
                "Cannot specify both" in str(e) and "request_id" in str(e) and "trace_id" in str(e)
            ),
        ):
            client.get_trace(request_id="abc", trace_id="def")

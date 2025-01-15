from typing import Sequence
from unittest import mock

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

import mlflow


class CustomExporter(SpanExporter):
    """A dummy exporter that logs spans to a mock client."""

    def __init__(self, mock_client):
        self._mock_client = mock_client

    def export(self, span: Sequence[ReadableSpan]):
        self._mock_client.log_span(span)


def test_export():
    mock_client = mock.MagicMock()

    mlflow.tracing.set_destination(CustomExporter(mock_client))

    with mlflow.start_span(name="root"):
        with mlflow.start_span(name="child") as child_span:
            child_span.set_inputs("dummy")

    assert mock_client.log_span.call_count == 2
    first_call_args, second_call_args = mock_client.log_span.call_args_list
    assert first_call_args[0][0][0].name == "child"
    assert second_call_args[0][0][0].name == "root"
    assert first_call_args[0][0][0].parent.span_id == second_call_args[0][0][0].context.span_id

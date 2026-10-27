import base64
import logging
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.span import LiveSpan
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted
from mlflow.tracing.attachments import Attachment
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION


def _make_trace_info_mock():
    info = MagicMock()
    info.trace_id = "tr-test123"
    info.tags = {TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value}
    info.metadata = {}
    return info


def _make_trace(attachments_map=None):
    span = MagicMock()
    span._attachments = attachments_map or {}

    trace = MagicMock()
    trace.info = _make_trace_info_mock()
    trace.info.trace_id = "tr-test123"
    trace.data.spans = [span]
    return trace


def _make_exporter(mock_client):
    with patch.object(MlflowV3SpanExporter, "__init__", return_value=None):
        exporter = MlflowV3SpanExporter()
    exporter._client = mock_client
    exporter._store_supports_log_spans = True
    return exporter


def test_log_trace_uploads_attachments():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    trace = _make_trace({att.id: att})

    mock_client = MagicMock()
    returned_info = _make_trace_info_mock()
    mock_client.start_trace.return_value = returned_info

    exporter = _make_exporter(mock_client)

    with (
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        exporter._log_trace(trace, prompts=[])

    mock_client._upload_trace_data.assert_called_once_with(returned_info, trace.data)
    mock_client._upload_attachments.assert_called_once()
    call_args = mock_client._upload_attachments.call_args
    assert call_args[0][0] is returned_info
    assert att.id in call_args[0][1]


def test_log_trace_uploads_attachments_in_tracking_store_mode():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    trace = _make_trace({att.id: att})
    # Override to TRACKING_STORE mode
    trace.info.tags = {TraceTagKey.SPANS_LOCATION: SpansLocation.TRACKING_STORE.value}

    mock_client = MagicMock()
    returned_info = MagicMock()
    returned_info.tags = {TraceTagKey.SPANS_LOCATION: SpansLocation.TRACKING_STORE.value}
    mock_client.start_trace.return_value = returned_info

    exporter = _make_exporter(mock_client)

    with (
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        exporter._log_trace(trace, prompts=[])

    # traces.json should NOT be uploaded in TRACKING_STORE mode
    mock_client._upload_trace_data.assert_not_called()
    # But attachments MUST still be uploaded to the artifact repo
    mock_client._upload_attachments.assert_called_once()
    call_args = mock_client._upload_attachments.call_args
    assert call_args[0][0] is returned_info
    assert att.id in call_args[0][1]


def test_trace_data_still_lands_when_attachment_upload_fails():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    trace = _make_trace({att.id: att})

    mock_client = MagicMock()
    returned_info = _make_trace_info_mock()
    mock_client.start_trace.return_value = returned_info
    mock_client._upload_attachments.side_effect = Exception("S3 timeout")

    exporter = _make_exporter(mock_client)

    with (
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        # Should not raise
        exporter._log_trace(trace, prompts=[])

    # Trace data was still uploaded despite attachment failure
    mock_client._upload_trace_data.assert_called_once_with(returned_info, trace.data)
    mock_client._upload_attachments.assert_called_once()


def _uc_table_trace_info(trace_id="trace:/cat.sch.mlflow_experiment/abc123", tags=None):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_databricks_uc_table_prefix(
            "cat", "sch", "mlflow_experiment"
        ),
        request_time=0,
        state=TraceState.OK,
        tags=tags or {},
    )


def _tracing_client():
    with patch("mlflow.tracing.client._get_store"):
        return TracingClient()


@pytest.mark.parametrize(
    "trace_location",
    [
        TraceLocation.from_databricks_uc_table_prefix("cat", "sch", "mlflow_experiment"),
        TraceLocation.from_databricks_uc_schema("cat", "sch"),
    ],
)
def test_upload_attachments_uc_table_raises_location_specific_error(trace_location):
    att = Attachment(content_type="image/png", content_bytes=b"img")
    info = TraceInfo(
        trace_id="trace:/cat.sch.mlflow_experiment/abc123",
        trace_location=trace_location,
        request_time=0,
        state=TraceState.OK,
        tags={},
    )
    client = _tracing_client()

    with pytest.raises(
        MlflowException, match="not supported for Unity Catalog table traces"
    ) as exc:
        client._upload_attachments(info, {att.id: att})

    assert type(exc.value) is MlflowException
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_upload_attachments_v4_trace_without_location_raises_location_specific_error():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    info = MagicMock()
    info.tags = {}
    info.trace_location = None
    info.trace_id = "trace:/cat.sch.mlflow_experiment/abc123"
    info.request_id = None
    client = _tracing_client()

    with pytest.raises(
        MlflowException, match="not supported for Unity Catalog table traces"
    ) as exc:
        client._upload_attachments(info, {att.id: att})

    assert type(exc.value) is MlflowException


def test_upload_attachments_missing_artifact_location_still_reports_corrupted():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    info = TraceInfo(
        trace_id="tr-test123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        state=TraceState.OK,
        tags={},
    )
    client = _tracing_client()

    with pytest.raises(MlflowTraceDataCorrupted, match="Trace data is corrupted"):
        client._upload_attachments(info, {att.id: att})


def test_upload_attachments_uses_artifact_location_when_present():
    att = Attachment(content_type="image/png", content_bytes=b"img")
    info = _uc_table_trace_info(tags={MLFLOW_ARTIFACT_LOCATION: "dbfs:/Volumes/cat/sch/vol"})
    repo = MagicMock()
    client = _tracing_client()

    with patch.object(client, "_get_artifact_repo_for_trace", return_value=repo) as mock_repo:
        client._upload_attachments(info, {att.id: att})

    mock_repo.assert_called_once_with(info)
    repo.upload_attachment.assert_called_once_with(att.id, att.content_bytes)


def test_log_trace_uc_inline_image_does_not_upload(caplog):
    from opentelemetry.sdk.trace import TracerProvider

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9aw"
        "AAAABJRU5ErkJggg=="
    )
    png = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
    otel_span = TracerProvider().get_tracer("test").start_span("test_span")
    live = LiveSpan(otel_span, trace_id="trace:/cat.sch.mlflow_experiment/abc123")
    live.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": png}},
                ],
            }
        ]
    })
    span = live.to_immutable_span()
    info = _uc_table_trace_info()
    trace = Trace(info=info, data=TraceData(spans=[span]))

    mock_client = MagicMock()
    mock_client.start_trace.return_value = info
    exporter = _make_exporter(mock_client)

    with (
        caplog.at_level(logging.WARNING, logger="mlflow.tracing.export.mlflow_v3"),
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        exporter._log_trace(trace, prompts=[])

    mock_client._upload_attachments.assert_not_called()
    mock_client.start_trace.assert_called_once_with(info)
    assert "Trace data is corrupted" not in caplog.text
    url = span.inputs["messages"][0]["content"][0]["image_url"]["url"]
    assert url == png


def test_log_trace_uc_attachment_upload_does_not_report_corrupted(caplog):
    att = Attachment(content_type="image/png", content_bytes=b"img")
    trace = _make_trace({att.id: att})
    info = _uc_table_trace_info()

    mock_client = MagicMock()
    mock_client.start_trace.return_value = info
    mock_client._upload_attachments = _tracing_client()._upload_attachments
    exporter = _make_exporter(mock_client)

    with (
        caplog.at_level(logging.WARNING, logger="mlflow.tracing.export.mlflow_v3"),
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        exporter._log_trace(trace, prompts=[])

    assert "Trace data is corrupted" not in caplog.text
    assert "not supported for Unity Catalog table traces" in caplog.text


def test_log_trace_skips_upload_when_no_attachments():
    trace = _make_trace()

    mock_client = MagicMock()
    returned_info = _make_trace_info_mock()
    mock_client.start_trace.return_value = returned_info

    exporter = _make_exporter(mock_client)

    with (
        patch("mlflow.tracing.export.mlflow_v3.try_link_prompts_to_trace"),
        patch("mlflow.tracing.export.mlflow_v3.add_size_stats_to_trace_metadata"),
    ):
        exporter._log_trace(trace, prompts=[])

    mock_client._upload_trace_data.assert_called_once()
    mock_client._upload_attachments.assert_not_called()

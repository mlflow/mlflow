from unittest.mock import MagicMock, patch

from mlflow.tracing.attachments import Attachment
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter


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

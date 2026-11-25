from unittest import mock

import mlflow
from mlflow.tracing.attachments import Attachment
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import skip_when_testing_trace_sdk

if not IS_TRACING_SDK_ONLY:
    from mlflow.server import handlers
    from mlflow.server.constants import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR


def test_attachment_roundtrip_with_local_tracking():
    image_bytes = b"\x89PNG\r\n\x1a\n fake png content"
    audio_bytes = b"RIFF fake wav content"

    with mlflow.start_span(name="integration-span") as span:
        span.set_inputs({
            "prompt": "describe this",
            "image": Attachment(content_type="image/png", content_bytes=image_bytes),
        })
        span.set_outputs({
            "audio": Attachment(content_type="audio/wav", content_bytes=audio_bytes),
            "text": "a cat",
        })
        trace_id = span.trace_id

    # Retrieve the trace and verify reference URIs are stored
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    root_span = trace.data.spans[0]

    assert root_span.inputs["prompt"] == "describe this"
    assert root_span.outputs["text"] == "a cat"

    image_ref = root_span.inputs["image"]
    audio_ref = root_span.outputs["audio"]

    image_parsed = Attachment.parse_ref(image_ref)
    audio_parsed = Attachment.parse_ref(audio_ref)

    assert image_parsed["content_type"] == "image/png"
    assert image_parsed["trace_id"] == trace_id
    assert audio_parsed["content_type"] == "audio/wav"
    assert audio_parsed["trace_id"] == trace_id

    # Verify the attachment files were written to the artifact repo
    from mlflow.tracing.client import TracingClient

    tracking_uri = mlflow.get_tracking_uri()
    client = TracingClient(tracking_uri)
    trace_info = client.get_trace_info(trace_id)
    artifact_repo = client._get_artifact_repo_for_trace(trace_info)

    stored_image = artifact_repo.download_trace_attachment(image_parsed["attachment_id"])
    stored_audio = artifact_repo.download_trace_attachment(audio_parsed["attachment_id"])

    assert stored_image == image_bytes
    assert stored_audio == audio_bytes


@skip_when_testing_trace_sdk
def test_attachment_upload_inside_server_with_proxied_artifact_root(tmp_path, monkeypatch):
    # Inside the tracking server (e.g. AI Gateway usage tracking), the tracking URI is the
    # backend store URI and the trace artifact location is `mlflow-artifacts:`.
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "true")
    monkeypatch.setenv(ARTIFACTS_DESTINATION_ENV_VAR, str(tmp_path))
    mlflow.set_experiment(
        experiment_id=mlflow.create_experiment("exp", artifact_location="mlflow-artifacts:/exp")
    )
    image_bytes = b"\x89PNG\r\n\x1a\n fake png content"

    with mock.patch("mlflow.tracing.export.mlflow_v3._logger.warning") as mock_warning:
        with mlflow.start_span(name="span") as span:
            span.set_inputs({
                "image": Attachment(content_type="image/png", content_bytes=image_bytes)
            })
        mlflow.flush_trace_async_logging()

    mock_warning.assert_not_called()
    trace = mlflow.get_trace(span.trace_id)
    assert trace.info.tags[MLFLOW_ARTIFACT_LOCATION].startswith("mlflow-artifacts:/exp/")
    attachment_id = Attachment.parse_ref(trace.data.spans[0].inputs["image"])["attachment_id"]
    artifact_repo = handlers._get_trace_artifact_repo(trace.info)
    assert artifact_repo.download_trace_attachment(attachment_id) == image_bytes

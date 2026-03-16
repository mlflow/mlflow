import mlflow
from mlflow.tracing.attachments import Attachment


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

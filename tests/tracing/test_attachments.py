import base64
import json
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import LiveSpan
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.tracing.attachments import Attachment


def test_attachment_creation_with_content():
    content = b"test content"
    content_type = "text/plain"

    attachment = Attachment(content_type=content_type, content_bytes=content)

    assert attachment.content_bytes == content
    assert attachment.content_type == content_type
    assert attachment.id is not None
    assert isinstance(attachment.id, str)
    assert attachment.filename is None


def test_attachment_from_file(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    content = b"Hello, world!"
    test_file.write_bytes(content)

    attachment = Attachment.from_file(test_file)

    assert attachment.content_bytes == content
    assert attachment.content_type == "text/plain"
    assert attachment.id is not None
    assert attachment.filename == "test.txt"  # Filename preserved


def test_attachment_from_file_with_custom_content_type(tmp_path: Path):
    test_file = tmp_path / "test.custom"
    content = b"Custom content"
    test_file.write_bytes(content)

    custom_type = "application/custom"
    attachment = Attachment.from_file(test_file, content_type=custom_type)

    assert attachment.content_bytes == content
    assert attachment.content_type == custom_type
    assert attachment.filename == "test.custom"


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("test.txt", "text/plain"),
        ("test.json", "application/json"),
        ("test.pdf", "application/pdf"),
        ("test.png", "image/png"),
        ("test.jpg", "image/jpeg"),
        ("test.jpeg", "image/jpeg"),
        ("test.gif", "image/gif"),
        ("test.webp", "image/webp"),
        ("test.mp3", "audio/mpeg"),
        ("test.wav", "audio/x-wav"),  # mimetypes returns audio/x-wav
        ("test.ogg", "audio/ogg"),
        ("test.m4a", "audio/mp4a-latm"),  # mimetypes returns audio/mp4a-latm
        ("test.aac", "audio/x-aac"),  # mimetypes returns audio/x-aac
        ("test.flac", "audio/x-flac"),  # mimetypes returns audio/x-flac
        ("test.webm", "video/webm"),  # mimetypes returns video/webm for .webm
        ("test.mp4", "video/mp4"),
        ("test.mov", "video/quicktime"),
        ("test.avi", "video/x-msvideo"),
        ("test.unknown", "application/octet-stream"),
    ],
)
def test_content_type_inference(filename, expected_type):
    assert Attachment._infer_content_type(filename) == expected_type


def test_attachment_reference_generation():
    attachment = Attachment(content_type="text/plain", content_bytes=b"test")
    trace_id = "test-trace-123"
    span_id = "span-456"

    ref = attachment.ref(trace_id, span_id)

    assert ref.startswith("mlflow-attachment:")
    assert not ref.startswith("mlflow-attachment://")  # JSON format, not URI

    # Parse the reference to check contents
    metadata = Attachment.parse_ref(ref)
    assert metadata["attachment_id"] == attachment.id
    assert metadata["trace_id"] == trace_id
    assert metadata["span_id"] == span_id
    assert metadata["content_type"] == "text/plain"
    assert metadata["size"] == 4


def test_attachment_from_ref():
    attachment_id = str(uuid.uuid4())
    content_type = "text/plain"
    trace_id = "test-trace-123"
    span_id = "span-456"

    # Create a JSON-based reference
    metadata = {
        "attachment_id": attachment_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "content_type": content_type,
        "size": 100,
    }
    json_str = json.dumps(metadata)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    ref_string = f"mlflow-attachment:{encoded}"

    attachment = Attachment.from_ref(ref_string)

    assert isinstance(attachment, Attachment)
    assert attachment.content_type == content_type
    # Content bytes would be empty until download is implemented
    assert attachment.content_bytes == b""


def test_attachment_from_ref_invalid_format():
    invalid_ref = "http://invalid-format/attachment"

    with pytest.raises(ValueError, match="Invalid attachment reference format"):
        Attachment.from_ref(invalid_ref)


def test_attachment_from_ref_missing_params():
    # Missing content_type
    metadata = {
        "attachment_id": "test-id",
        "trace_id": "trace-123",
        "span_id": "span-456",
        # Missing content_type
    }
    json_str = json.dumps(metadata)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    ref_without_type = f"mlflow-attachment:{encoded}"

    with pytest.raises(ValueError, match="Invalid attachment reference"):
        Attachment.from_ref(ref_without_type)

    # Invalid JSON
    ref_invalid_json = "mlflow-attachment:not-valid-base64"
    with pytest.raises(ValueError, match="Invalid attachment reference"):
        Attachment.from_ref(ref_invalid_json)


def test_attachment_parse_ref():
    attachment_id = "test-id"
    content_type = "text/plain"
    trace_id = "test-trace"
    span_id = "span-456"

    # Create a JSON-based reference
    metadata = {
        "attachment_id": attachment_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "content_type": content_type,
        "size": 42,
    }
    json_str = json.dumps(metadata)
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    ref_string = f"mlflow-attachment:{encoded}"

    parsed = Attachment.parse_ref(ref_string)

    assert parsed["attachment_id"] == attachment_id
    assert parsed["trace_id"] == trace_id
    assert parsed["span_id"] == span_id
    assert parsed["content_type"] == content_type
    assert parsed["size"] == 42


# AttachmentRef class removed - tests for download will be implemented
# when the actual download logic is added to Attachment.from_ref()


def test_attachment_metadata_fields():
    content = b"test content"
    content_type = "text/plain"
    filename = "test.txt"

    attachment = Attachment(content_type=content_type, content_bytes=content, filename=filename)

    # Check all metadata fields
    assert attachment.filename == filename

    # Check that filename is included in reference
    ref = attachment.ref("trace-123", "span-456")
    metadata = Attachment.parse_ref(ref)

    assert metadata["filename"] == filename


def create_mock_otel_span():
    """Create a mock OpenTelemetry span for testing."""
    mock_otel_span = Mock(spec=ReadableSpan)
    mock_otel_span.name = "test_span"
    mock_otel_span.context.span_id = 12345
    mock_otel_span.context.trace_id = 67890
    mock_otel_span._start_time = 1000000
    mock_otel_span._end_time = 2000000
    mock_otel_span.parent = None
    mock_otel_span.attributes = {}
    return mock_otel_span


def test_live_span_attachment_storage():
    mock_otel_span = create_mock_otel_span()

    with patch("mlflow.entities.span._SpanAttributesRegistry") as mock_registry:
        mock_attr_registry = Mock()
        mock_attr_registry.get.return_value = "test-trace-123"
        mock_registry.return_value = mock_attr_registry

        live_span = LiveSpan(mock_otel_span, "test-trace-123", "TEST")

        assert hasattr(live_span, "_attachments")
        assert isinstance(live_span._attachments, dict)
        assert len(live_span._attachments) == 0

        mock_registry.assert_called_once()


def test_live_span_set_inputs_with_attachments():
    mock_otel_span = create_mock_otel_span()

    with patch("mlflow.entities.span._SpanAttributesRegistry") as mock_registry:
        mock_attr_registry = Mock()
        mock_attr_registry.get.return_value = "test-trace-123"
        mock_registry.return_value = mock_attr_registry

        live_span = LiveSpan(mock_otel_span, "test-trace-123", "TEST")

        attachment = Attachment(content_type="text/plain", content_bytes=b"test")

        # Mock the ref method to work with LiveSpan's two argument call
        with patch.object(attachment, "ref") as mock_ref:
            mock_ref.return_value = "mlflow-attachment:mocked_ref"
            inputs = {"text_param": "hello", "file_attachment": attachment}
            live_span.set_inputs(inputs)

        assert len(live_span._attachments) == 1

        mock_attr_registry.set.assert_called()
        call_args = mock_attr_registry.set.call_args
        assert call_args[0][0] == "mlflow.spanInputs"

        processed_inputs = call_args[0][1]
        assert "text_param" in processed_inputs
        assert processed_inputs["text_param"] == "hello"
        assert "file_attachment" in processed_inputs

        file_ref = processed_inputs["file_attachment"]
        assert isinstance(file_ref, str)
        assert file_ref == "mlflow-attachment:mocked_ref"


def test_live_span_set_outputs_with_attachments():
    mock_otel_span = create_mock_otel_span()

    with patch("mlflow.entities.span._SpanAttributesRegistry") as mock_registry:
        mock_attr_registry = Mock()
        mock_attr_registry.get.return_value = "test-trace-123"
        mock_registry.return_value = mock_attr_registry

        live_span = LiveSpan(mock_otel_span, "test-trace-123", "TEST")

        attachment = Attachment(content_type="image/png", content_bytes=b"fake-png-data")

        # Mock the ref method to work with LiveSpan's two argument call
        with patch.object(attachment, "ref") as mock_ref:
            mock_ref.return_value = "mlflow-attachment:mocked_output_ref"
            outputs = {"result": "success", "generated_image": attachment}
            live_span.set_outputs(outputs)

        assert len(live_span._attachments) == 1

        stored_ref = list(live_span._attachments.keys())[0]
        assert stored_ref == "mlflow-attachment:mocked_output_ref"

        mock_attr_registry.set.assert_called()


def test_live_span_to_immutable_span_transfers_attachments():
    mock_otel_span = create_mock_otel_span()

    with patch("mlflow.entities.span._SpanAttributesRegistry") as mock_registry:
        mock_attr_registry = Mock()
        mock_attr_registry.get.return_value = "test-trace-123"
        mock_registry.return_value = mock_attr_registry

        live_span = LiveSpan(mock_otel_span, "test-trace-123", "TEST")

        attachment = Attachment(content_type="text/plain", content_bytes=b"test")
        ref = attachment.ref("test-trace-123", "test-span-456")
        live_span._attachments[ref] = attachment

        immutable_span = live_span.to_immutable_span()

        assert hasattr(immutable_span, "_attachments")
        assert len(immutable_span._attachments) == 1
        assert ref in immutable_span._attachments
        assert immutable_span._attachments[ref] is attachment

        mock_registry.assert_called()


def test_upload_trace_attachment():
    mock_repo = Mock(spec=ArtifactRepository)
    attachment = Attachment(content_type="text/plain", content_bytes=b"test content")

    ArtifactRepository.upload_trace_attachment(mock_repo, attachment)

    mock_repo.log_artifact.assert_called_once()
    call_args = mock_repo.log_artifact.call_args
    assert call_args[1]["artifact_path"] == "attachments"

    temp_file_path = call_args[0][0]
    assert temp_file_path.name == attachment.id


def test_download_trace_attachment():
    mock_repo = Mock(spec=ArtifactRepository)
    expected_content = b"downloaded attachment content"

    def mock_download_file(remote_path, local_path):
        assert remote_path == "attachments/test-attachment-id"
        local_path.write_bytes(expected_content)

    mock_repo._download_file.side_effect = mock_download_file

    content = ArtifactRepository.download_trace_attachment(mock_repo, "test-attachment-id")

    assert content == expected_content
    mock_repo._download_file.assert_called_once()


def test_download_trace_attachment_not_found():
    mock_repo = Mock(spec=ArtifactRepository)
    mock_repo._download_file.side_effect = FileNotFoundError("File not found")

    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match="Trace attachment not found"):
        ArtifactRepository.download_trace_attachment(mock_repo, "nonexistent-id")

    mock_repo._download_file.assert_called_once()

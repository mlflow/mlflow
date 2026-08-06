import json
import uuid

import pytest

from mlflow.exceptions import MlflowException, MlflowTraceDataNotFound
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository


def test_upload_attachment(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    attachment_id = str(uuid.uuid4())
    content = b"fake image bytes"

    repo.upload_attachment(attachment_id, content)

    artifact_path = tmp_path / "attachments" / attachment_id
    assert artifact_path.exists()
    assert artifact_path.read_bytes() == content


def test_download_trace_attachment(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))

    # Manually place an attachment file
    attachments_dir = tmp_path / "attachments"
    attachments_dir.mkdir()
    attachment_id = str(uuid.uuid4())
    (attachments_dir / attachment_id).write_bytes(b"stored bytes")

    result = repo.download_trace_attachment(attachment_id)
    assert result == b"stored bytes"


def test_download_trace_attachment_to_file(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))

    attachments_dir = tmp_path / "attachments"
    attachments_dir.mkdir()
    attachment_id = str(uuid.uuid4())
    content = b"\x89PNG to-file test"
    (attachments_dir / attachment_id).write_bytes(content)

    dst = tmp_path / "output" / attachment_id
    dst.parent.mkdir()
    result = repo.download_trace_attachment_to_file(attachment_id, dst)

    assert result == dst
    assert dst.read_bytes() == content


def test_download_trace_attachment_to_file_rejects_invalid_path(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    with pytest.raises(MlflowException, match="Invalid attachment path"):
        repo.download_trace_attachment_to_file("../../etc/passwd", tmp_path / "out")


def test_download_trace_data_to_file(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))

    trace_data = {"spans": [{"name": "test"}]}
    (tmp_path / "traces.json").write_text(json.dumps(trace_data))

    dst = tmp_path / "output" / "traces.json"
    dst.parent.mkdir()
    result = repo.download_trace_data_to_file(dst)

    assert result == dst
    assert json.loads(dst.read_text()) == trace_data


def test_download_trace_data_to_file_missing(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    dst = tmp_path / "output" / "traces.json"
    dst.parent.mkdir()

    with pytest.raises(MlflowTraceDataNotFound, match="traces.json"):
        repo.download_trace_data_to_file(dst)


def test_upload_and_download_roundtrip(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    attachment_id = str(uuid.uuid4())
    content = b"\x89PNG\r\n\x1a\n fake png header"

    repo.upload_attachment(attachment_id, content)
    result = repo.download_trace_attachment(attachment_id)
    assert result == content


def test_upload_and_download_to_file_roundtrip(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    attachment_id = str(uuid.uuid4())
    content = b"\x89PNG\r\n\x1a\n fake png header"

    repo.upload_attachment(attachment_id, content)

    dst = tmp_path / "downloaded" / attachment_id
    dst.parent.mkdir()
    repo.download_trace_attachment_to_file(attachment_id, dst)
    assert dst.read_bytes() == content


def test_upload_rejects_path_traversal(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    with pytest.raises(MlflowException, match="Invalid attachment path"):
        repo.upload_attachment("../../etc/passwd", b"data")


def test_download_rejects_path_traversal(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    with pytest.raises(MlflowException, match="Invalid attachment path"):
        repo.download_trace_attachment("../../secrets.json")


def test_rejects_non_uuid_attachment_id(tmp_path):
    repo = LocalArtifactRepository(str(tmp_path))
    with pytest.raises(MlflowException, match="Invalid attachment path"):
        repo.upload_attachment("not-a-valid-id!", b"data")

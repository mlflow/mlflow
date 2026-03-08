import tempfile
from pathlib import Path

import pytest

from mlflow.tracing.attachments import Attachment


def test_attachment_init():
    att = Attachment(content_type="image/png", content_bytes=b"fakepng")
    assert att.id is not None
    assert att.content_type == "image/png"
    assert att.content_bytes == b"fakepng"


def test_attachment_ids_are_unique():
    a = Attachment(content_type="image/png", content_bytes=b"a")
    b = Attachment(content_type="image/png", content_bytes=b"b")
    assert a.id != b.id


def test_attachment_from_file():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"fakepng")
        path = Path(f.name)
    try:
        att = Attachment.from_file(path)
        assert att.content_type == "image/png"
        assert att.content_bytes == b"fakepng"
    finally:
        path.unlink()


def test_attachment_from_file_explicit_content_type():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"data")
        path = Path(f.name)
    try:
        att = Attachment.from_file(path, content_type="custom/type")
        assert att.content_type == "custom/type"
    finally:
        path.unlink()


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".mp3", "audio/mpeg"),
        (".pdf", "application/pdf"),
        (".xyz_unknown", "application/octet-stream"),
    ],
)
def test_attachment_from_file_mime_inference(suffix, expected, tmp_path):
    file = tmp_path / f"test{suffix}"
    file.write_bytes(b"data")
    att = Attachment.from_file(file)
    assert att.content_type == expected


def test_attachment_ref():
    att = Attachment(content_type="image/png", content_bytes=b"data")
    ref = att.ref("tr-abc123")
    assert ref.startswith("mlflow-attachment://")
    assert att.id in ref
    assert "content_type=image/png" in ref
    assert "trace_id=tr-abc123" in ref


def test_parse_ref_valid():
    att = Attachment(content_type="audio/wav", content_bytes=b"data")
    ref = att.ref("tr-xyz")
    parsed = Attachment.parse_ref(ref)
    assert parsed is not None
    assert parsed["attachment_id"] == att.id
    assert parsed["content_type"] == "audio/wav"
    assert parsed["trace_id"] == "tr-xyz"


def test_parse_ref_invalid():
    assert Attachment.parse_ref("https://example.com") is None
    assert Attachment.parse_ref("not-a-uri") is None
    assert Attachment.parse_ref("mlflow-attachment://") is None

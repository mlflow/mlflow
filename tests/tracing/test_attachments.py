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


def test_attachment_from_file(tmp_path):
    file = tmp_path / "test.png"
    file.write_bytes(b"fakepng")
    att = Attachment.from_file(file)
    assert att.content_type == "image/png"
    assert att.content_bytes == b"fakepng"


def test_attachment_from_file_explicit_content_type(tmp_path):
    file = tmp_path / "test.bin"
    file.write_bytes(b"data")
    att = Attachment.from_file(file, content_type="custom/type")
    assert att.content_type == "custom/type"


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
    parsed = Attachment.parse_ref(ref)
    assert parsed["content_type"] == "image/png"
    assert parsed["trace_id"] == "tr-abc123"
    assert parsed["size"] == 4


def test_parse_ref_valid():
    att = Attachment(content_type="audio/wav", content_bytes=b"data")
    ref = att.ref("tr-xyz")
    parsed = Attachment.parse_ref(ref)
    assert parsed is not None
    assert parsed["attachment_id"] == att.id
    assert parsed["content_type"] == "audio/wav"
    assert parsed["trace_id"] == "tr-xyz"
    assert parsed["size"] == 4


def test_parse_ref_invalid():
    assert Attachment.parse_ref("https://example.com") is None
    assert Attachment.parse_ref("not-a-uri") is None
    assert Attachment.parse_ref("mlflow-attachment://") is None


def test_from_file_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        Attachment.from_file("/nonexistent/path/image.png")


def test_ref_roundtrips_special_characters_in_content_type():
    att = Attachment(content_type="application/vnd.custom+json", content_bytes=b"data")
    ref = att.ref("tr-123")
    parsed = Attachment.parse_ref(ref)
    assert parsed["content_type"] == "application/vnd.custom+json"
    assert parsed["size"] == 4


def test_parse_ref_without_size():
    uri = "mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456"
    parsed = Attachment.parse_ref(uri)
    assert parsed is not None
    assert parsed["attachment_id"] == "abc-123"
    assert parsed["content_type"] == "image/png"
    assert parsed["trace_id"] == "tr-456"
    assert parsed["size"] is None


def test_ref_size_reflects_content_length():
    content = b"x" * 1024
    att = Attachment(content_type="image/png", content_bytes=content)
    ref = att.ref("tr-001")
    parsed = Attachment.parse_ref(ref)
    assert parsed["size"] == 1024


@pytest.mark.parametrize("bad_size", ["abc", "-1", "-100", "0"])
def test_parse_ref_invalid_size(bad_size):
    uri = f"mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456&size={bad_size}"
    parsed = Attachment.parse_ref(uri)
    assert parsed is not None
    assert parsed["size"] is None

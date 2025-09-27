#!/usr/bin/env python
"""Test script for audio attachment functionality."""

import tempfile
from pathlib import Path

from mlflow.tracing.attachments import Attachment


def test_audio_content_types():
    """Test that audio file extensions are correctly mapped to MIME types."""
    test_cases = [
        ("test.mp3", "audio/mpeg"),
        ("test.wav", "audio/wav"),
        ("test.ogg", "audio/ogg"),
        ("test.m4a", "audio/mp4"),
        ("test.aac", "audio/aac"),
        ("test.flac", "audio/flac"),
        ("test.webm", "audio/webm"),
        ("test.unknown", "application/octet-stream"),
    ]

    for filename, expected_content_type in test_cases:
        content_type = Attachment._infer_content_type(filename)
        assert content_type == expected_content_type, (
            f"Failed for {filename}: expected {expected_content_type}, got {content_type}"
        )
        print(f"✓ {filename} -> {content_type}")

    print("\nAll audio content type tests passed!")


def test_audio_attachment_creation():
    """Test creating audio attachments from files."""
    # Create a temporary audio file (mock)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(b"mock audio data")
        tmp_path = Path(tmp.name)

    try:
        # Create attachment from file
        attachment = Attachment.from_file(tmp_path)

        assert attachment.content_type == "audio/mpeg"
        assert attachment.content_bytes == b"mock audio data"
        assert attachment.id is not None

        # Test reference creation
        trace_id = "test-trace-123"
        ref = attachment.ref(trace_id)
        assert "mlflow-attachment://" in ref
        assert "content_type=audio/mpeg" in ref
        assert f"trace_id=tr-{trace_id}" in ref

        print("\n✓ Audio attachment created successfully")
        print(f"  ID: {attachment.id}")
        print(f"  Content type: {attachment.content_type}")
        print(f"  Reference: {ref}")

    finally:
        # Clean up
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_audio_content_types()
    test_audio_attachment_creation()

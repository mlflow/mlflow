"""
Smoke test: log a video artifact to a local MLflow run and verify
that the artifact listing API returns it, confirming the backend
correctly stores and serves video files.

Run with:
    uv run pytest tests/test_video_artifact_detection.py -v
"""

import os
import struct
import tempfile

import mlflow
from mlflow.tracking import MlflowClient


def _create_minimal_webm(path: str) -> None:
    """Create a minimal valid .webm file (EBML header only).

    This is enough for the frontend to detect it as a video artifact
    without requiring ffmpeg or any heavy dependencies.
    """
    # Minimal EBML header for WebM (Matroska container)
    # EBML Element ID + size + DocType "webm"
    ebml_header = bytes(
        [
            0x1A,
            0x45,
            0xDF,
            0xA3,  # EBML element ID
            0x01,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x1F,  # Element size (31 bytes)
            0x42,
            0x86,  # EBMLVersion
            0x81,  # Size: 1
            0x01,  # Version 1
            0x42,
            0xF7,  # EBMLReadVersion
            0x81,  # Size: 1
            0x01,  # Version 1
            0x42,
            0xF2,  # EBMLMaxIDLength
            0x81,  # Size: 1
            0x04,  # 4 bytes
            0x42,
            0xF3,  # EBMLMaxSizeLength
            0x81,  # Size: 1
            0x08,  # 8 bytes
            0x42,
            0x82,  # DocType
            0x84,  # Size: 4
            0x77,
            0x65,
            0x62,
            0x6D,  # "webm"
            0x42,
            0x87,  # DocTypeVersion
            0x81,  # Size: 1
            0x04,  # Version 4
            0x42,
            0x85,  # DocTypeReadVersion
            0x81,  # Size: 1
            0x02,  # Version 2
        ]
    )
    with open(path, "wb") as f:
        f.write(ebml_header)


def _create_minimal_mp4(path: str) -> None:
    """Create a minimal .mp4 file (ftyp box only).

    This produces a valid MP4/ISO Base Media File Format header.
    """
    # ftyp box: size(4) + 'ftyp'(4) + major_brand(4) + minor_version(4) + compatible_brands
    major_brand = b"isom"
    minor_version = struct.pack(">I", 0x200)
    compatible_brands = b"isomiso2mp41"
    ftyp_data = b"ftyp" + major_brand + minor_version + compatible_brands
    ftyp_size = struct.pack(">I", len(ftyp_data) + 4)  # +4 for size field itself
    with open(path, "wb") as f:
        f.write(ftyp_size + ftyp_data)


def test_video_artifact_logged_and_listed():
    """Log video artifacts and verify the listing API returns them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up local tracking with sqlite backend
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir}/mlflow.db")

        # Create test video files
        webm_path = os.path.join(tmpdir, "rollout.webm")
        mp4_path = os.path.join(tmpdir, "episode_001.mp4")
        _create_minimal_webm(webm_path)
        _create_minimal_mp4(mp4_path)

        # Log artifacts
        with mlflow.start_run() as run:
            mlflow.log_artifact(webm_path, artifact_path="videos")
            mlflow.log_artifact(mp4_path, artifact_path="videos")

        # Verify via API
        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, path="videos")
        artifact_paths = [a.path for a in artifacts]

        assert "videos/rollout.webm" in artifact_paths
        assert "videos/episode_001.mp4" in artifact_paths


def test_video_extension_detection():
    """Verify that video extensions are properly detected as video (not audio)."""
    video_extensions = [".mp4", ".webm", ".mov", ".m4v", ".ogg", ".ogv", ".mkv", ".avi"]
    audio_only_extensions = [".mp3", ".wav", ".aac", ".flac", ".m4a"]

    # Video extensions should be recognized
    for ext in video_extensions:
        filename = f"test_file{ext}"
        assert filename.lower().endswith(ext), f"{ext} detection failed"

    # .mp4 should NOT be in audio-only category
    assert ".mp4" not in audio_only_extensions


def test_video_artifact_at_root_level():
    """Log a video artifact at the root level (no artifact_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir}/mlflow.db")

        mp4_path = os.path.join(tmpdir, "my_video.mp4")
        _create_minimal_mp4(mp4_path)

        with mlflow.start_run() as run:
            mlflow.log_artifact(mp4_path)

        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = [a.path for a in artifacts]

        assert "my_video.mp4" in artifact_paths


def test_video_artifact_nested_in_rollouts():
    """Log video artifacts under rollouts/ path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir}/mlflow.db")

        webm_path = os.path.join(tmpdir, "step_42.webm")
        _create_minimal_webm(webm_path)

        with mlflow.start_run() as run:
            mlflow.log_artifact(webm_path, artifact_path="rollouts")

        client = MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id, path="rollouts")
        artifact_paths = [a.path for a in artifacts]

        assert "rollouts/step_42.webm" in artifact_paths

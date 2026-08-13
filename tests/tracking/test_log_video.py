import os
import re

import numpy as np
import pytest

import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_IMAGES

# Smallest thing ffmpeg-free tests can rely on: log_video never inspects or decodes
# the container, it only checks the extension and copies the bytes.
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "rollout.mp4"
    path.write_bytes(VIDEO_BYTES)
    return str(path)


def _logged_images_dir():
    return os.path.join(local_file_uri_to_path(mlflow.get_artifact_uri()), "images")


def test_log_video_writes_step_indexed_artifact(video_file):
    with mlflow.start_run() as run:
        mlflow.log_video(video_file, key="rollout", step=7)

        files = os.listdir(_logged_images_dir())
        assert len(files) == 1
        # <key>+step+<N>+timestamp+<T>+<uuid>.mp4, the same convention log_image uses.
        assert re.fullmatch(r"rollout\+step\+7\+timestamp\+\d+\+[0-9a-f-]{36}\.mp4", files[0])

        logged = os.path.join(_logged_images_dir(), files[0])
        assert open(logged, "rb").read() == VIDEO_BYTES

    finished = mlflow.get_run(run.info.run_id)
    assert finished.data.tags[MLFLOW_LOGGED_IMAGES] == "True"


def test_log_video_without_poster_writes_no_companion(video_file):
    with mlflow.start_run():
        mlflow.log_video(video_file, key="rollout", step=0)
        files = os.listdir(_logged_images_dir())

    # The poster is optional, so MLflow needs no transcoding dependency.
    assert not any(f.endswith("+compressed.webp") for f in files)


def test_log_video_with_poster_writes_companion(video_file):
    poster = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    with mlflow.start_run():
        mlflow.log_video(video_file, key="rollout", step=3, poster=poster)
        files = sorted(os.listdir(_logged_images_dir()))

    assert len(files) == 2
    video = next(f for f in files if f.endswith(".mp4"))
    companion = next(f for f in files if f.endswith("+compressed.webp"))
    # The UI pairs the two by the shared stem, so they must agree exactly.
    assert companion == video[: -len(".mp4")] + "+compressed.webp"


def test_log_video_defaults_step_to_zero(video_file):
    with mlflow.start_run():
        mlflow.log_video(video_file, key="rollout")
        (name,) = os.listdir(_logged_images_dir())

    assert "+step+0+" in name


def test_log_video_sanitizes_slash_in_key(video_file):
    with mlflow.start_run():
        mlflow.log_video(video_file, key="eval/rollout", step=0)
        (name,) = os.listdir(_logged_images_dir())

    # '/' would otherwise create a subdirectory the UI does not scan.
    assert name.startswith("eval~rollout+step+0+")


@pytest.mark.parametrize("key", ["roll+out", "rollout!", "roll:out"])
def test_log_video_rejects_invalid_key(video_file, key):
    # '+' matters most here: it delimits the fields the UI parses back out, so a key
    # containing one would silently corrupt the step and timestamp.
    with mlflow.start_run(), pytest.raises(ValueError, match="may only contain alphanumerics"):
        mlflow.log_video(video_file, key=key, step=0)


def test_log_video_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "rollout.gif"
    path.write_bytes(VIDEO_BYTES)
    with mlflow.start_run(), pytest.raises(ValueError, match="Unsupported video format"):
        mlflow.log_video(str(path), key="rollout", step=0)


def test_log_video_rejects_missing_file(tmp_path):
    with mlflow.start_run(), pytest.raises(FileNotFoundError, match="No such video file"):
        mlflow.log_video(str(tmp_path / "absent.mp4"), key="rollout", step=0)


def test_log_video_rejects_unsupported_poster_type(video_file):
    with mlflow.start_run(), pytest.raises(TypeError, match="Unsupported poster object type"):
        mlflow.log_video(video_file, key="rollout", step=0, poster="not-an-image")


@pytest.mark.parametrize("extension", ["mp4", "webm", "mov"])
def test_log_video_accepts_supported_extensions(tmp_path, extension):
    path = tmp_path / f"rollout.{extension}"
    path.write_bytes(VIDEO_BYTES)
    with mlflow.start_run():
        mlflow.log_video(str(path), key="rollout", step=0)
        (name,) = os.listdir(_logged_images_dir())

    assert name.endswith(f".{extension}")


def test_log_video_steps_are_independently_addressable(video_file):
    with mlflow.start_run():
        for step in range(3):
            mlflow.log_video(video_file, key="rollout", step=step)
        files = os.listdir(_logged_images_dir())

    steps = sorted(int(re.search(r"\+step\+(\d+)\+", f).group(1)) for f in files)
    assert steps == [0, 1, 2]

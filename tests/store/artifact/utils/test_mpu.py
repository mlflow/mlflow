import concurrent.futures
import time
from pathlib import Path
from typing import List

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.utils.mpu import _upload_chunks_with_retry


def test_upload_chunks_with_retry(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = _upload_chunks_with_retry(
            thread_pool=executor,
            filename=tmp_file,
            upload_fn=lambda x: x,
            num_chunks=3,
        )

        assert results == [0, 1, 2]


def test_upload_chunks_with_retry_error(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.write_text("test")

    def upload_fn(index):
        raise Exception("Unexpected error")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with pytest.raises(MlflowException, match="Unexpected error"):
            _upload_chunks_with_retry(
                thread_pool=executor,
                filename=tmp_file,
                upload_fn=upload_fn,
                num_chunks=1,
            )


class UploadFn:
    def __init__(self, seconds: List[float]) -> None:
        self.seconds = (s for s in seconds)

    def __call__(self, index: int) -> int:
        time.sleep(next(self.seconds))
        return index


def test_upload_chunks_with_retry_timeout_success(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = _upload_chunks_with_retry(
            thread_pool=executor,
            filename=tmp_file,
            upload_fn=UploadFn(
                [
                    1.0,  # Timeout
                    0.0,  # Success
                ]
            ),
            num_chunks=1,
            timeout=0.5,
            max_retries=1,
        )
        assert results == [0]


def test_upload_chunks_with_retry_timeout_results_sorted(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    first_chunk_upload_fn = UploadFn([1.0, 0.0])

    def upload_fn(index):
        if index == 0:
            return first_chunk_upload_fn(index)
        return index

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = _upload_chunks_with_retry(
            thread_pool=executor,
            filename=tmp_file,
            upload_fn=upload_fn,
            num_chunks=2,
            timeout=0.5,
            max_retries=1,
        )
        assert results == [0, 1]


def test_upload_chunks_with_retry_timeout_failure(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with pytest.raises(MlflowException, match="TimeoutError"):
            _upload_chunks_with_retry(
                thread_pool=executor,
                filename=tmp_file,
                upload_fn=UploadFn(
                    [
                        1.0,  # Timeout
                        1.0,  # Timeout
                    ]
                ),
                num_chunks=1,
                timeout=0.5,
                max_retries=1,
            )


def test_upload_chunks_with_retry_timeout_and_error(tmp_path: Path):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    def upload_fn(index):
        if index == 0:
            raise Exception("Unexpected error")
        else:
            time.sleep(1.0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        with pytest.raises(MlflowException, match="Unexpected error"):
            _upload_chunks_with_retry(
                thread_pool=executor,
                filename=tmp_file,
                upload_fn=upload_fn,
                num_chunks=2,
                timeout=1.0,
                max_retries=1,
            )


def test_upload_chunks_with_retry_timeout_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tmp_file = tmp_path.joinpath("test.txt")
    tmp_file.touch()

    monkeypatch.setenv("MLFLOW_MULTIPART_CHUNK_UPLOAD_TIMEOUT", "0.5")
    monkeypatch.setenv("MLFLOW_MULTIPART_CHUNK_UPLOAD_MAX_RETRIES", "1")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = _upload_chunks_with_retry(
            thread_pool=executor,
            filename=tmp_file,
            upload_fn=UploadFn(
                [
                    1.0,  # Timeout
                    0.0,  # Success
                ]
            ),
            num_chunks=1,
        )
        assert results == [0]

import concurrent.futures
import logging
import os
from concurrent.futures import Executor
from typing import Any, Callable, List, Optional, Union

from mlflow.environment_variables import (
    MLFLOW_MULTIPART_CHUNK_UPLOAD_MAX_RETRIES,
    MLFLOW_MULTIPART_CHUNK_UPLOAD_TIMEOUT_SECONDS,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import ArtifactProgressBar

_logger = logging.getLogger(__name__)


def _upload_chunks_with_retry(
    thread_pool: Executor,
    filename: Union[str, bytes, os.PathLike],
    upload_fn: Callable[[int], Any],
    num_chunks: int,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> List[Any]:
    """
    Upload multipart upload chunks with retries on timeout.

    Args:
        thread_pool: Executor to use for concurrent chunk uploads.
        upload_fn: Function to call to upload a single chunk. Must take a single argument,
            the index of the chunk to upload.
        num_chunks: Number of chunks to upload.
        filename: The file being uploaded.
        timeout: Timeout for each chunk upload. If None, defaults to the value of
            MLFLOW_MULTIPART_CHUNK_UPLOAD_TIMEOUT_SECONDS.
        max_retries: Maximum number of retries for each chunk upload. If None, defaults to the value
            of MLFLOW_MULTIPART_CHUNK_UPLOAD_MAX_RETRIES.

    Returns:
        List of results of the upload_fn calls, in order of chunk index.
    """
    results = {}
    timeout = MLFLOW_MULTIPART_CHUNK_UPLOAD_TIMEOUT_SECONDS.get() if timeout is None else timeout
    max_retries = (
        MLFLOW_MULTIPART_CHUNK_UPLOAD_MAX_RETRIES.get() if max_retries is None else max_retries
    )
    max_attempts = max_retries + 1
    chunk_size = MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()

    with ArtifactProgressBar.chunks(
        os.path.getsize(filename), f"Uploading {filename}", chunk_size
    ) as pbar:
        futures = {thread_pool.submit(upload_fn, i): i for i in range(num_chunks)}
        for attempt in range(max_attempts):
            timed_out_chunks = []
            errors = {}
            for f, index in futures.items():
                _logger.debug("Uploading chunk %s upload", index)
                try:
                    res = f.result(timeout=timeout)
                except concurrent.futures.TimeoutError as e:
                    if attempt != max_attempts - 1:
                        timed_out_chunks.append(index)
                        continue

                    errors[index] = e
                except Exception as e:
                    _logger.debug("Chunk %s upload failed with error %s", index, e, exc_info=True)
                    errors[index] = e
                else:
                    _logger.debug("Chunk %s uploaded successfully", index)
                    results[index] = res
                    pbar.update()

            if errors:
                raise MlflowException(
                    f"Failed to upload at least one part of {filename}. Errors: {errors}"
                )

            if timed_out_chunks:
                futures = {}
                for i in timed_out_chunks:
                    _logger.debug("Retrying chunk %s upload", i)
                    futures[thread_pool.submit(upload_fn, i)] = i
            else:
                break

        return [results[k] for k in sorted(results.keys())]

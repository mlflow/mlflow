from __future__ import annotations

import posixpath
from pathlib import Path

from mlflow.artifacts import download_artifacts, list_artifacts
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import MAX_ARCHIVE_ENTRIES
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import normalize_subpath, tree_size
from mlflow.protos.databricks_pb2 import ErrorCode


def _declared_size(uri: str, *, max_bytes: int) -> int:
    """
    Sum the file sizes the artifact repository reports beneath ``uri`` before downloading.

    ``download_artifacts`` schedules every file at once, so an oversized tree would otherwise
    consume the full bandwidth and scratch disk before ``max_bytes`` is applied. The listing is
    walked with the same entry cap as archives, and the walk stops at the first file that
    pushes the total past the limit. Reported sizes are advisory, so the bytes actually written
    are measured again after the download.
    """
    total = 0
    entries = 0
    pending = [uri.rstrip("/")]
    while pending:
        current = pending.pop()
        for info in list_artifacts(artifact_uri=current):
            entries += 1
            if entries > MAX_ARCHIVE_ENTRIES:
                raise invalid_content(
                    f"MLflow artifact source '{uri}' has more than {MAX_ARCHIVE_ENTRIES} entries."
                )
            # Listed paths are relative to the parent of the listed directory; only the final
            # segment is needed to descend.
            child = f"{current}/{posixpath.basename(info.path)}"
            if info.is_dir:
                pending.append(child)
                continue
            total += info.file_size or 0
            if total > max_bytes:
                return total
    return total


def fetch_mlflow_artifacts(
    uri: str, dest: Path, *, max_bytes: int, subpath: str | None = None
) -> Path:
    """
    Download the artifact directory at ``uri`` into ``dest`` using the caller's MLflow credentials.

    ``uri`` is any artifact URI ``mlflow.artifacts.download_artifacts`` accepts, such as
    ``mlflow-artifacts:/skills/code-review/<token>`` or ``runs:/<run_id>/skill``. When
    ``subpath`` is given only that directory is downloaded, placed at ``dest/<subpath>`` so the
    caller's containment check resolves it the same way as every other source. The size limit
    is checked against the repository's listing before any file is transferred and against the
    downloaded bytes afterwards. The returned path is the base under which ``subpath``
    resolves.
    """
    prefix = normalize_subpath(subpath)
    dest.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        artifact_uri = uri
        dst = dest
    else:
        artifact_uri = f"{uri.rstrip('/')}/{prefix}"
        dst = dest.joinpath(*prefix.split("/")[:-1])
        dst.mkdir(parents=True, exist_ok=True)
    try:
        listed = _declared_size(artifact_uri, max_bytes=max_bytes)
    except MlflowException as e:
        raise source_unavailable(uri, e.message, error_code=ErrorCode.Value(e.error_code))
    if listed > max_bytes:
        raise invalid_content(
            f"MLflow artifact content is at least {listed} bytes, which exceeds the skill "
            f"content size limit of {max_bytes} bytes."
        )
    try:
        downloaded = Path(download_artifacts(artifact_uri=artifact_uri, dst_path=str(dst)))
    except MlflowException as e:
        raise source_unavailable(uri, e.message, error_code=ErrorCode.Value(e.error_code))
    except OSError as e:
        raise source_unavailable(uri, str(e))
    if not downloaded.is_dir():
        raise invalid_content(f"MLflow artifact source '{uri}' must be a directory, not a file.")
    if (size := tree_size(downloaded)) > max_bytes:
        raise invalid_content(
            f"MLflow artifact content is {size} bytes, which exceeds the skill content size "
            f"limit of {max_bytes} bytes."
        )
    return dest if prefix is not None else downloaded

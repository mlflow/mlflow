from __future__ import annotations

from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import normalize_subpath, tree_size
from mlflow.protos.databricks_pb2 import ErrorCode


def fetch_mlflow_artifacts(
    uri: str, dest: Path, *, max_bytes: int, subpath: str | None = None
) -> Path:
    """
    Download the artifact directory at ``uri`` into ``dest`` using the caller's MLflow credentials.

    ``uri`` is any artifact URI ``mlflow.artifacts.download_artifacts`` accepts, such as
    ``mlflow-artifacts:/skills/code-review/<token>`` or ``runs:/<run_id>/skill``. When
    ``subpath`` is given only that directory is downloaded, placed at ``dest/<subpath>`` so the
    caller's containment check resolves it the same way as every other source. The returned
    path is the base under which ``subpath`` resolves.
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

from __future__ import annotations

from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import tree_size


def fetch_mlflow_artifacts(uri: str, dest: Path, *, max_bytes: int) -> Path:
    """
    Download the artifact directory at ``uri`` into ``dest`` using the caller's MLflow credentials.

    ``uri`` is any artifact URI ``mlflow.artifacts.download_artifacts`` accepts, such as
    ``mlflow-artifacts:/skills/code-review/<token>`` or ``runs:/<run_id>/skill``. The returned
    path is the downloaded directory itself.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(download_artifacts(artifact_uri=uri, dst_path=str(dest)))
    except MlflowException as e:
        raise source_unavailable(uri, e.message)
    except OSError as e:
        raise source_unavailable(uri, str(e))
    if not downloaded.is_dir():
        raise invalid_content(f"MLflow artifact source '{uri}' must be a directory, not a file.")
    if (size := tree_size(downloaded)) > max_bytes:
        raise invalid_content(
            f"MLflow artifact content is {size} bytes, which exceeds the skill content size "
            f"limit of {max_bytes} bytes."
        )
    return downloaded

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

from typing import Literal

from mlflow.artifacts import download_artifacts
from mlflow.tracking.fluent import log_artifacts
from mlflow.utils.databricks_utils import DatabricksRuntimeVersion

EnvPackType = Literal["databricks_model_serving"]

_ARTIFACT_PATH = "_databricks"
_MODEL_VERSION_TAR = "model_version.tar"
_MODEL_ENVIRONMENT_TAR = "model_environment.tar"
_SUPPORTED_CLIENT_IMAGE_MAJOR_VERSIONS_FOR_MODEL_SERVING = [2, 3]

def _tar(root_path: str, tar_path: str):
    """
    Package all files under root_path into a tar at tar_path, excluding __pycache__, *.pyc, and wheels_info.json.
    """
    def exclude(tarinfo: tarfile.TarInfo):
        name = tarinfo.name
        base = os.path.basename(name)
        if "__pycache__" in name or base.endswith(".pyc") or base == "wheels_info.json":
            return None
        return tarinfo

    # Pull in symlinks
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        tar.add(root_path, arcname=".", filter=exclude)
    return tar


# TODO: Check pip requirements using uv instead.
def pack_env_for_databricks_model_serving(
    run_id: str,
    model_uri: str,
    artifact_path: str,
    *,
    enforce_pip_requirements: bool = False,
) -> str:
    """
    Generate Databricks artifacts for fast deployment. Must be called in an active run.
    """
    dbr_version = DatabricksRuntimeVersion.parse()
    if not dbr_version.is_client_image or dbr_version.major not in _SUPPORTED_CLIENT_IMAGE_MAJOR_VERSIONS_FOR_MODEL_SERVING:
        raise ValueError(f"Serverless environment of versions {_SUPPORTED_CLIENT_IMAGE_MAJOR_VERSIONS_FOR_MODEL_SERVING} is required when packing environment for Databricks Model Serving")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download model artifacts. Keep this separate from temp_dir to avoid noise in packaged artifacts.
        local_artifacts_dir = download_artifacts(artifact_uri=model_uri)
        if enforce_pip_requirements:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    f"{local_artifacts_dir}/requirements.txt",
                ]
            )

        # Package model artifacts and env into temp_dir/_databricks
        temp_artifacts_dir = os.path.join(temp_dir, _ARTIFACT_PATH)
        os.makedirs(temp_artifacts_dir, exist_ok=False)
        _tar(local_artifacts_dir, os.path.join(temp_artifacts_dir, _MODEL_VERSION_TAR))
        # VIRTUAL_ENV is set by venv and points to the active virtual environment
        _tar(os.environ["VIRTUAL_ENV"], os.path.join(temp_artifacts_dir, _MODEL_ENVIRONMENT_TAR))
        # Move the temp_databricks_dir inside local_artifacts_dir because log_artifacts is full overwrite
        shutil.move(temp_artifacts_dir, local_artifacts_dir)

        # Log the packaged temp_dir/_databricks artifacts
        log_artifacts(
            local_artifacts_dir,
            artifact_path=os.path.join(artifact_path, _ARTIFACT_PATH),
            run_id=run_id,
        )

    return local_artifacts_dir

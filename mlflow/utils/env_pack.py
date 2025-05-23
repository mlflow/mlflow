import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

from contextlib import contextmanager
from typing import Literal, Generator

from mlflow.artifacts import download_artifacts
from mlflow.utils.databricks_utils import DatabricksRuntimeVersion
from mlflow.utils.environment import _REQUIREMENTS_FILE_NAME
from mlflow.utils.logging_utils import eprint

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
@contextmanager
def pack_env_for_databricks_model_serving(
    model_uri: str,
    *,
    enforce_pip_requirements: bool = False,
) -> Generator[str, None, None]:
    """
    Generate Databricks artifacts for fast deployment.
    
    Args:
        model_uri: The URI of the model to package.
        enforce_pip_requirements: Whether to enforce pip requirements installation.
        
    Yields:
        str: The path to the local artifacts directory containing the model artifacts and environment.
        
    Example:
        >>> with pack_env_for_databricks_model_serving("models:/my-model/1") as artifacts_dir:
        ...     # Use artifacts_dir here
        ...     pass
    """
    dbr_version = DatabricksRuntimeVersion.parse()
    if not dbr_version.is_client_image or dbr_version.major not in _SUPPORTED_CLIENT_IMAGE_MAJOR_VERSIONS_FOR_MODEL_SERVING:
        raise ValueError(f"Serverless environment of versions {_SUPPORTED_CLIENT_IMAGE_MAJOR_VERSIONS_FOR_MODEL_SERVING} is required when packing environment for Databricks Model Serving. Current version: {dbr_version}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download model artifacts. Keep this separate from temp_dir to avoid noise in packaged artifacts.
        local_artifacts_dir = download_artifacts(artifact_uri=model_uri)
        if enforce_pip_requirements:
            try:
                eprint("Installing model requirements...")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        os.path.join(local_artifacts_dir, _REQUIREMENTS_FILE_NAME),
                    ],
                    capture_output=True,
                    stderr=subprocess.stdout,
                    text=True,
                )
                result.check_returncode()
            except subprocess.CalledProcessError as e:
                eprint("Error installing requirements:")
                eprint(e.stdout)
                raise

        # Package model artifacts and env into temp_dir/_databricks
        temp_artifacts_dir = os.path.join(temp_dir, _ARTIFACT_PATH)
        os.makedirs(temp_artifacts_dir, exist_ok=False)
        _tar(local_artifacts_dir, os.path.join(temp_artifacts_dir, _MODEL_VERSION_TAR))
        _tar(sys.prefix, os.path.join(temp_artifacts_dir, _MODEL_ENVIRONMENT_TAR))
        shutil.move(temp_artifacts_dir, local_artifacts_dir)

        yield local_artifacts_dir

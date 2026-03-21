import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

import yaml

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import DatabricksRuntimeVersion, get_databricks_runtime_version
from mlflow.utils.environment import _REQUIREMENTS_FILE_NAME
from mlflow.utils.logging_utils import eprint

EnvPackType = Literal["databricks_model_serving"]


@dataclass(kw_only=True)
class EnvPackConfig:
    name: EnvPackType
    install_dependencies: bool = True


_ARTIFACT_PATH = "_databricks"
_MODEL_VERSION_TAR = "model_version.tar"
_MODEL_ENVIRONMENT_TAR = "model_environment.tar"


def _validate_env_pack(env_pack):
    """Checks if env_pack is a supported value

    Supported values are:
    - the string "databricks_model_serving"
    - an ``EnvPackConfig`` with ``name == 'databricks_model_serving'`` and a boolean
      ``install_dependencies`` field.
    - None
    """
    if env_pack is None:
        return None

    if isinstance(env_pack, str):
        if env_pack == "databricks_model_serving":
            return EnvPackConfig(name="databricks_model_serving", install_dependencies=True)
        raise MlflowException.invalid_parameter_value(
            f"Invalid env_pack value: {env_pack!r}. Expected: 'databricks_model_serving'."
        )

    if isinstance(env_pack, EnvPackConfig):
        if env_pack.name != "databricks_model_serving":
            raise MlflowException.invalid_parameter_value(
                f"Invalid EnvPackConfig.name: {env_pack.name!r}. "
                "Expected 'databricks_model_serving'."
            )
        if not isinstance(env_pack.install_dependencies, bool):
            raise MlflowException.invalid_parameter_value(
                "EnvPackConfig.install_dependencies must be a bool."
            )
        return env_pack

    # Anything else is invalid
    raise MlflowException.invalid_parameter_value(
        "env_pack must be either None, the string 'databricks_model_serving', or an EnvPackConfig "
        "with a boolean 'install_dependencies' field."
    )


def _tar(root_path: Path, tar_path: Path) -> tarfile.TarFile:
    """
    Package all files under root_path into a tar at tar_path, excluding __pycache__, *.pyc, and
    wheels_info.json.
    """

    def exclude(tarinfo: tarfile.TarInfo):
        name = tarinfo.name
        base = Path(name).name
        if "__pycache__" in name or base.endswith(".pyc") or base == "wheels_info.json":
            return None
        return tarinfo

    # Pull in symlinks
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        tar.add(root_path, arcname=".", filter=exclude)
    return tar


@contextmanager
def _get_source_artifacts(
    model_uri: str, local_model_path: str | None = None
) -> Generator[Path, None, None]:
    """
    Get source artifacts and handle cleanup of downloads.
    Does not mutate local_model_path contents if provided.

    Args:
        model_uri: The URI of the model to package.
        local_model_path: Optional local path to model artifacts.

    Yields:
        Path: The path to the source artifacts directory.
    """
    source_dir = Path(local_model_path or download_artifacts(artifact_uri=model_uri))

    yield source_dir

    if not local_model_path:
        shutil.rmtree(source_dir)


# TODO: Check pip requirements using uv instead.
@contextmanager
def pack_env_for_databricks_model_serving(
    model_uri: str,
    *,
    enforce_pip_requirements: bool = False,
    local_model_path: str | None = None,
) -> Generator[str, None, None]:
    """
    Generate Databricks artifacts for fast deployment.

    Args:
        model_uri: The URI of the model to package.
        enforce_pip_requirements: Whether to enforce pip requirements installation.
        local_model_path: Optional local path to model artifacts. If provided, pack
            the local artifacts instead of downloading.

    Yields:
        str: The path to the local artifacts directory containing the model artifacts and
            environment.

    Example:
        >>> with pack_env_for_databricks_model_serving("models:/my-model/1") as artifacts_dir:
        ...     # Use artifacts_dir here
        ...     pass
    """
    dbr_version = DatabricksRuntimeVersion.parse()
    if not dbr_version.is_client_image:
        raise ValueError(
            f"Serverless environment is required when packing environment for Databricks Model "
            f"Serving. Current version: {dbr_version}"
        )

    with _get_source_artifacts(model_uri, local_model_path) as source_artifacts_dir:
        # Check runtime version consistency
        # We read the MLmodel file directly instead of using Model.to_dict() because to_dict() adds
        # the current runtime version via get_databricks_runtime_version(), which would prevent us
        # from detecting runtime version mismatches.
        mlmodel_path = source_artifacts_dir / MLMODEL_FILE_NAME
        with open(mlmodel_path) as f:
            model_dict = yaml.safe_load(f)
        if "databricks_runtime" not in model_dict:
            raise ValueError(
                "Model must have been created in a Databricks runtime environment. "
                "Missing 'databricks_runtime' field in MLmodel file."
            )

        current_runtime = DatabricksRuntimeVersion.parse()
        model_runtime = DatabricksRuntimeVersion.parse(model_dict["databricks_runtime"])
        if current_runtime.major != model_runtime.major:
            raise ValueError(
                f"Runtime version mismatch. Model was created with runtime "
                f"{model_dict['databricks_runtime']} (major version {model_runtime.major}), "
                f"but current runtime is {get_databricks_runtime_version()} "
                f"(major version {current_runtime.major})"
            )

        # Check that _databricks directory does not exist in source
        if (source_artifacts_dir / _ARTIFACT_PATH).exists():
            raise MlflowException(
                f"Source artifacts contain a '{_ARTIFACT_PATH}' directory and is not "
                "eligible for use with env_pack.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if enforce_pip_requirements:
            eprint("Installing model requirements...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(source_artifacts_dir / _REQUIREMENTS_FILE_NAME),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                eprint("Error installing requirements:")
                eprint(e.stdout)
                raise

        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy source artifacts to packaged_model_dir
            packaged_model_dir = Path(temp_dir) / "model"
            shutil.copytree(
                source_artifacts_dir, packaged_model_dir, dirs_exist_ok=False, symlinks=False
            )

            # Package model artifacts and env into packaged_model_dir/_databricks
            packaged_artifacts_dir = packaged_model_dir / _ARTIFACT_PATH
            packaged_artifacts_dir.mkdir(exist_ok=False)
            _tar(source_artifacts_dir, packaged_artifacts_dir / _MODEL_VERSION_TAR)
            _tar(Path(sys.prefix), packaged_artifacts_dir / _MODEL_ENVIRONMENT_TAR)

            yield str(packaged_model_dir)

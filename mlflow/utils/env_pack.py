import os
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

class _ChunkedTarWriter:
    """
    File-like object that writes tar data to chunks if size exceeds chunk_size.
    If data fits in a single chunk, writes to the original path without suffix.
    """

    def __init__(self, tar_path: Path):
        self.base_path = tar_path
        self.chunk_size = int(os.environ.get("TAR_CHUNK_SIZE_BYTES", 256 * 1024 * 1024))
        self.current_file = None
        self.current_chunk_index = 0
        self.current_chunk_bytes = 0
        self.total_bytes = 0
        self.is_chunked = False
        self._open_next_chunk()

    def _open_next_chunk(self):
        """Open the next chunk file for writing."""
        if self.current_file is not None:
            self.current_file.close()

        # First chunk always writes to base path (no suffix)
        # We'll rename later if we need multiple chunks
        if self.current_chunk_index == 0:
            chunk_path = self.base_path
        else:
            # We're creating a second chunk, so we need to rename the first
            if self.current_chunk_index == 1:
                self._promote_to_chunked()
            chunk_path = self._get_chunk_path_unpadded(self.current_chunk_index)

        self.current_file = open(chunk_path, "wb")
        self.current_chunk_bytes = 0

    def _promote_to_chunked(self):
        """
        Rename the first chunk to .part0 when we realize we need multiple chunks.
        This is called when opening the second chunk.
        """
        if not self.is_chunked:
            self.is_chunked = True
            # Rename the first chunk without padding (will be padded in close())
            self.base_path.rename(self._get_chunk_path_unpadded(0))

    def _get_chunk_path_unpadded(self, index: int) -> Path:
        """Get the path for a specific chunk index without padding."""
        return Path(f"{self.base_path}.part{index}")

    def _get_chunk_path_padded(self, index: int, width: int) -> Path:
        """Get the path for a specific chunk index with padding."""
        return Path(f"{self.base_path}.part{index:0{width}d}")

    def write(self, data: bytes) -> int:
        """Write data, creating new chunks as needed."""
        bytes_written = 0
        remaining = data

        while remaining:
            # Check if current chunk would exceed limit
            if self.current_chunk_bytes >= self.chunk_size and self.current_chunk_index > 0:
                # Open next chunk (but not on first chunk)
                self.current_chunk_index += 1
                self._open_next_chunk()

            # Write as much as we can to current chunk
            space_left = self.chunk_size - self.current_chunk_bytes
            to_write = remaining[:space_left] if space_left < len(remaining) else remaining

            written = self.current_file.write(to_write)
            self.current_chunk_bytes += written
            self.total_bytes += written
            bytes_written += written
            remaining = remaining[written:]

            # If we've exceeded chunk size on first chunk, prepare for chunking
            if self.current_chunk_index == 0 and self.current_chunk_bytes >= self.chunk_size:
                self.current_chunk_index += 1
                self._open_next_chunk()

        return bytes_written

    def tell(self) -> int:
        """Return the current position (total bytes written)."""
        return self.total_bytes

    def flush(self):
        """Flush the current file."""
        if self.current_file is not None:
            self.current_file.flush()

    def close(self):
        """Close the current file and finalize chunk naming."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None

        # If we ended up chunking, add minimal padding to all chunks
        if self.is_chunked:
            # Calculate minimal padding needed based on actual number of chunks
            suffix_width = len(str(self.current_chunk_index))
            # Rename all chunks from unpadded to padded
            for i in range(self.current_chunk_index + 1):
                old_path = self._get_chunk_path_unpadded(i)
                new_path = self._get_chunk_path_padded(i, suffix_width)
                old_path.rename(new_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


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


def _tar(root_path: Path, tar_path: Path) -> None:
    """
    Package all files under root_path into a tar at tar_path, excluding __pycache__, *.pyc, and
    wheels_info.json.

    Large tars will be split into multiple, lexicographically ordered chunks during creation.
    """

    def exclude(tarinfo: tarfile.TarInfo):
        name = tarinfo.name
        base = Path(name).name
        if "__pycache__" in name or base.endswith(".pyc") or base == "wheels_info.json":
            return None
        return tarinfo

    # Write tar to chunked writer which handles splitting automatically
    with _ChunkedTarWriter(tar_path) as writer:
        with tarfile.open(fileobj=writer, mode="w", dereference=True) as tar:
            tar.add(root_path, arcname=".", filter=exclude)


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

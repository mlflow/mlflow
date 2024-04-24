"""
APIs for interacting with artifacts in MLflow
"""
import json
import pathlib
import tempfile
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.tracking import _get_store
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
    _get_root_uri_and_artifact_path,
    add_databricks_profile_info_to_artifact_uri,
    get_artifact_repository,
)


def download_artifacts(
    artifact_uri: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dst_path: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> str:
    """Download an artifact file or directory to a local directory.

    Args:
        artifact_uri: URI pointing to the artifacts, such as
            ``"runs:/500cf58bee2b40a4a82861cc31a617b1/my_model.pkl"``,
            ``"models:/my_model/Production"``, or ``"s3://my_bucket/my/file.txt"``.
            Exactly one of ``artifact_uri`` or ``run_id`` must be specified.
        run_id: ID of the MLflow Run containing the artifacts. Exactly one of ``run_id`` or
            ``artifact_uri`` must be specified.
        artifact_path: (For use with ``run_id``) If specified, a path relative to the MLflow
            Run's root directory containing the artifacts to download.
        dst_path: Path of the local filesystem destination directory to which to download the
            specified artifacts. If the directory does not exist, it is created. If
            unspecified, the artifacts are downloaded to a new uniquely-named directory on
            the local filesystem, unless the artifacts already exist on the local
            filesystem, in which case their local path is returned directly.
        tracking_uri: The tracking URI to be used when downloading artifacts.

    Returns:
        The location of the artifact file or directory on the local filesystem.
    """
    if (run_id, artifact_uri).count(None) != 1:
        raise MlflowException(
            message="Exactly one of `run_id` or `artifact_uri` must be specified",
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif artifact_uri is not None and artifact_path is not None:
        raise MlflowException(
            message="`artifact_path` cannot be specified if `artifact_uri` is specified",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if dst_path is not None:
        pathlib.Path(dst_path).mkdir(exist_ok=True, parents=True)

    if artifact_uri is not None:
        return _download_artifact_from_uri(artifact_uri, output_path=dst_path)

    artifact_path = artifact_path if artifact_path is not None else ""

    store = _get_store(store_uri=tracking_uri)
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(
        add_databricks_profile_info_to_artifact_uri(artifact_uri, tracking_uri)
    )
    return artifact_repo.download_artifacts(artifact_path, dst_path=dst_path)


def list_artifacts(
    artifact_uri: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    tracking_uri: Optional[str] = None,
):
    """List artifacts at the specified URI.

    Args:
        artifact_uri: URI pointing to the artifacts, such as
            ``"runs:/500cf58bee2b40a4a82861cc31a617b1/my_model.pkl"``,
            ``"models:/my_model/Production"``, or ``"s3://my_bucket/my/file.txt"``.
            Exactly one of ``artifact_uri`` or ``run_id`` must be specified.
        run_id: ID of the MLflow Run containing the artifacts. Exactly one of ``run_id`` or
            ``artifact_uri`` must be specified.
        artifact_path: (For use with ``run_id``) If specified, a path relative to the MLflow
            Run's root directory containing the artifacts to list.
        tracking_uri: The tracking URI to be used when list artifacts.

    Returns:
        List of artifacts as FileInfo listed directly under path.
    """
    if (run_id, artifact_uri).count(None) != 1:
        raise MlflowException.invalid_parameter_value(
            message="Exactly one of `run_id` or `artifact_uri` must be specified",
        )
    elif artifact_uri is not None and artifact_path is not None:
        raise MlflowException.invalid_parameter_value(
            message="`artifact_path` cannot be specified if `artifact_uri` is specified",
        )

    if artifact_uri is not None:
        root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
        return get_artifact_repository(artifact_uri=root_uri).list_artifacts(artifact_path)

    store = _get_store(store_uri=tracking_uri)
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(
        add_databricks_profile_info_to_artifact_uri(artifact_uri, tracking_uri)
    )
    return artifact_repo.list_artifacts(artifact_path)


def load_text(artifact_uri: str) -> str:
    """Loads the artifact contents as a string.

    Args:
        artifact_uri: Artifact location.

    Returns:
        The contents of the artifact as a string.

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run() as run:
            artifact_uri = run.info.artifact_uri
            mlflow.log_text("This is a sentence", "file.txt")
            file_content = mlflow.artifacts.load_text(artifact_uri + "/file.txt")
            print(file_content)

    .. code-block:: text
        :caption: Output

        This is a sentence
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_artifact = download_artifacts(artifact_uri, dst_path=tmpdir)
        with open(local_artifact) as local_artifact_fd:
            try:
                return str(local_artifact_fd.read())
            except Exception:
                raise MlflowException("Unable to form a str object from file content", BAD_REQUEST)


def load_dict(artifact_uri: str) -> dict:
    """Loads the artifact contents as a dictionary.

    Args:
        artifact_uri: artifact location.

    Returns:
        A dictionary.

    .. code-block:: python
      :caption: Example

      import mlflow

      with mlflow.start_run() as run:
          artifact_uri = run.info.artifact_uri
          mlflow.log_dict({"mlflow-version": "0.28", "n_cores": "10"}, "config.json")
          config_json = mlflow.artifacts.load_dict(artifact_uri + "/config.json")
          print(config_json)

    .. code-block:: text
      :caption: Output

      {'mlflow-version': '0.28', 'n_cores': '10'}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_artifact = download_artifacts(artifact_uri, dst_path=tmpdir)
        with open(local_artifact) as local_artifact_fd:
            try:
                return json.load(local_artifact_fd)
            except json.JSONDecodeError:
                raise MlflowException("Unable to form a JSON object from file content", BAD_REQUEST)


def load_image(artifact_uri: str):
    """Loads artifact contents as a ``PIL.Image.Image`` object

    Args:
        artifact_uri: Artifact location.

    Returns:
        A PIL.Image object.

    .. code-block:: python
        :caption: Example

        import mlflow
        from PIL import Image

        with mlflow.start_run() as run:
            image = Image.new("RGB", (100, 100))
            artifact_uri = run.info.artifact_uri
            mlflow.log_image(image, "image.png")
            image = mlflow.artifacts.load_image(artifact_uri + "/image.png")
            print(image)

    .. code-block:: text
        :caption: Output

        <PIL.PngImagePlugin.PngImageFile image mode=RGB size=100x100 at 0x11D2FA3D0>
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "`load_image` requires Pillow. Please install it via: pip install Pillow"
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        local_artifact = download_artifacts(artifact_uri, dst_path=tmpdir)
        try:
            image_obj = Image.open(local_artifact)
            image_obj.load()
            return image_obj
        except Exception:
            raise MlflowException(
                "Unable to form a PIL Image object from file content", BAD_REQUEST
            )

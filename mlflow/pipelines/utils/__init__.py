import logging
import os
import posixpath
import pathlib
from typing import Dict, Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml

_PIPELINE_CONFIG_FILE_NAME = "pipeline.yaml"
_PIPELINE_PROFILE_DIR = "profiles"
_PIPELINE_PROFILE_ENV_VAR = "MLFLOW_PIPELINES_PROFILE"

_logger = logging.getLogger(__name__)


def get_pipeline_name(pipeline_root_path: str = None) -> str:
    """
    Obtains the name of the specified pipeline or of the pipeline corresponding to the current
    working directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory.
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory or if ``pipeline_root_path`` is ``None`` and the current
                             working directory does not correspond to a pipeline.
    :return: The name of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    _verify_is_pipeline_root_directory(pipeline_root_path=pipeline_root_path)
    return os.path.basename(pipeline_root_path)


def get_pipeline_config(pipeline_root_path: str = None, profile: str = None) -> Dict[str, Any]:
    """
    Obtains a dictionary representation of the configuration for the specified pipeline.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem. If unspecified, the pipeline root directory is
                               resolved from the current working directory, and an
    :param profile: The name of the profile under the `profiles` directory to use,
                    e.g. "dev" to use configs from "profiles/dev.yaml"
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory or if ``pipeline_root_path`` is ``None`` and the current
                             working directory does not correspond to a pipeline.
    :return: The configuration of the specified pipeline.
    """
    pipeline_root_path = pipeline_root_path or get_pipeline_root_path()
    _verify_is_pipeline_root_directory(pipeline_root_path=pipeline_root_path)
    try:
        if profile:
            # Jinja expects template names in posixpath format relative to environment root,
            # so use posixpath to construct the relative path here.
            profile_relpath = posixpath.join(_PIPELINE_PROFILE_DIR, f"{profile}.yaml")
            profile_file_path = os.path.join(
                pipeline_root_path, _PIPELINE_PROFILE_DIR, f"{profile}.yaml"
            )
            if not os.path.exists(profile_file_path):
                raise MlflowException(
                    "Did not find the YAML configuration file for the specified profile"
                    f" '{profile}' at expected path '{profile_file_path}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return render_and_merge_yaml(
                pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME, profile_relpath
            )
        else:
            return read_yaml(pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    except MlflowException:
        raise
    except Exception as e:
        raise MlflowException(
            "Failed to read pipeline configuration. Please verify that the `pipeline.yaml`"
            " configuration file and the YAML configuration file for the selected profile are"
            " syntactically correct and that the specified profile provides all required values"
            " for template substitutions defined in `pipeline.yaml`.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def get_pipeline_root_path() -> str:
    """
    Obtains the path of the pipeline corresponding to the current working directory, throwing an
    ``MlflowException`` if the current working directory does not reside within a pipeline
    directory.

    :return: The absolute path of the pipeline root directory on the local filesystem.
    """
    # In the release version of MLflow Pipelines, each pipeline will be its own git repository.
    # To improve developer velocity for now, we choose to treat a pipeline as a directory, which
    # may be a subdirectory of a git repo. The logic for resolving the repository root for
    # development purposes finds the first `pipeline.yaml` file by traversing up the directory
    # tree, while the release version will find the pipeline repository root (commented out below)
    curr_dir_path = pathlib.Path.cwd()

    while True:
        pipeline_yaml_path_to_check = curr_dir_path / _PIPELINE_CONFIG_FILE_NAME
        if pipeline_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        else:
            # If curr_dir_path == curr_dir_path.parent,
            # we have reached the root directory without finding
            # the desired pipeline.yaml file
            raise MlflowException(f"Failed to find {_PIPELINE_CONFIG_FILE_NAME}!")


def get_default_profile() -> str:
    """
    Returns the default profile name under which a pipeline is executed. The default
    profile may change depending on runtime environment.

    :return: The default profile name string.
    """
    return "databricks" if is_in_databricks_runtime() else "local"


def _verify_is_pipeline_root_directory(pipeline_root_path: str) -> str:
    """
    Verifies that the specified local filesystem path is the path of a pipeline root directory.

    :param pipeline_root_path: The absolute path of the pipeline root directory on the local
                               filesystem to validate.
    :raises MlflowException: If the specified ``pipeline_root_path`` is not a pipeline root
                             directory.
    """
    pipeline_yaml_path = os.path.join(pipeline_root_path, _PIPELINE_CONFIG_FILE_NAME)
    if not os.path.exists(pipeline_yaml_path):
        raise MlflowException(
            f"Failed to find {_PIPELINE_CONFIG_FILE_NAME} in {pipeline_yaml_path}!"
        )

import logging
import os
import pathlib
import posixpath
from typing import Any, Dict

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import read_yaml, render_and_merge_yaml

_RECIPE_CONFIG_FILE_NAME = "recipe.yaml"
_RECIPE_PROFILE_DIR = "profiles"

_logger = logging.getLogger(__name__)


def get_recipe_name(recipe_root_path: str = None) -> str:
    """
    Obtains the name of the specified recipe or of the recipe corresponding to the current
    working directory.

    :param recipe_root_path: The absolute path of the recipe root directory on the local
                               filesystem. If unspecified, the recipe root directory is
                               resolved from the current working directory.
    :raises MlflowException: If the specified ``recipe_root_path`` is not a recipe root
                             directory or if ``recipe_root_path`` is ``None`` and the current
                             working directory does not correspond to a recipe.
    :return: The name of the specified recipe.
    """
    recipe_root_path = recipe_root_path or get_recipe_root_path()
    _verify_is_recipe_root_directory(recipe_root_path=recipe_root_path)
    return os.path.basename(recipe_root_path)


def get_recipe_config(recipe_root_path: str = None, profile: str = None) -> Dict[str, Any]:
    """
    Obtains a dictionary representation of the configuration for the specified recipe.

    :param recipe_root_path: The absolute path of the recipe root directory on the local
                               filesystem. If unspecified, the recipe root directory is
                               resolved from the current working directory, and an
    :param profile: The name of the profile under the `profiles` directory to use,
                    e.g. "dev" to use configs from "profiles/dev.yaml"
    :raises MlflowException: If the specified ``recipe_root_path`` is not a recipe root
                             directory or if ``recipe_root_path`` is ``None`` and the current
                             working directory does not correspond to a recipe.
    :return: The configuration of the specified recipe.
    """
    recipe_root_path = recipe_root_path or get_recipe_root_path()
    _verify_is_recipe_root_directory(recipe_root_path=recipe_root_path)
    try:
        if profile:
            # Jinja expects template names in posixpath format relative to environment root,
            # so use posixpath to construct the relative path here.
            profile_relpath = posixpath.join(_RECIPE_PROFILE_DIR, f"{profile}.yaml")
            profile_file_path = os.path.join(
                recipe_root_path, _RECIPE_PROFILE_DIR, f"{profile}.yaml"
            )
            if not os.path.exists(profile_file_path):
                raise MlflowException(
                    "Did not find the YAML configuration file for the specified profile"
                    f" '{profile}' at expected path '{profile_file_path}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return render_and_merge_yaml(
                recipe_root_path, _RECIPE_CONFIG_FILE_NAME, profile_relpath
            )
        else:
            return read_yaml(recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    except MlflowException:
        raise
    except Exception as e:
        raise MlflowException(
            "Failed to read recipe configuration. Please verify that the `recipe.yaml`"
            " configuration file and the YAML configuration file for the selected profile are"
            " syntactically correct and that the specified profile provides all required values"
            " for template substitutions defined in `recipe.yaml`.",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def get_recipe_root_path() -> str:
    """
    Obtains the path of the recipe corresponding to the current working directory, throwing an
    ``MlflowException`` if the current working directory does not reside within a recipe
    directory.

    :return: The absolute path of the recipe root directory on the local filesystem.
    """
    # In the release version of MLflow Recipes, each recipe will be its own git repository.
    # To improve developer velocity for now, we choose to treat a recipe as a directory, which
    # may be a subdirectory of a git repo. The logic for resolving the repository root for
    # development purposes finds the first `recipe.yaml` file by traversing up the directory
    # tree, while the release version will find the recipe repository root (commented out below)
    curr_dir_path = pathlib.Path.cwd()

    while True:
        recipe_yaml_path_to_check = curr_dir_path / _RECIPE_CONFIG_FILE_NAME
        if recipe_yaml_path_to_check.exists():
            return str(curr_dir_path.resolve())
        elif curr_dir_path != curr_dir_path.parent:
            curr_dir_path = curr_dir_path.parent
        else:
            # If curr_dir_path == curr_dir_path.parent,
            # we have reached the root directory without finding
            # the desired recipe.yaml file
            raise MlflowException(f"Failed to find {_RECIPE_CONFIG_FILE_NAME}!")


def get_default_profile() -> str:
    """
    Returns the default profile name under which a recipe is executed. The default
    profile may change depending on runtime environment.

    :return: The default profile name string.
    """
    return "databricks" if is_in_databricks_runtime() else "local"


def _verify_is_recipe_root_directory(recipe_root_path: str) -> str:
    """
    Verifies that the specified local filesystem path is the path of a recipe root directory.

    :param recipe_root_path: The absolute path of the recipe root directory on the local
                               filesystem to validate.
    :raises MlflowException: If the specified ``recipe_root_path`` is not a recipe root
                             directory.
    """
    recipe_yaml_path = os.path.join(recipe_root_path, _RECIPE_CONFIG_FILE_NAME)
    if not os.path.exists(recipe_yaml_path):
        raise MlflowException(f"Failed to find {_RECIPE_CONFIG_FILE_NAME} in {recipe_yaml_path}!")

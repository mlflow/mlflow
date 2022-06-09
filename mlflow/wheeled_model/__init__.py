"""
Add documentation
"""

import os
import yaml
import mlflow
from mlflow.pyfunc.model import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _overwrite_pip_deps,
)

_ARTIFACTS_FOLDER_NAME = "artifacts"
_WHEELS_FOLDER_NAME = "wheels"

FLAVOR_NAME = "wheeled_model"


def log_model(artifact_path, model_uri, model_name):
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.wheeled_model,
        registered_model_name=model_name,
        model_uri=model_uri,
    )


def save_model(path, model_uri, mlflow_model=None):
    wheels_dir = os.path.join(path, _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME)
    pip_requirements_path = os.path.join(path, _REQUIREMENTS_FILE_NAME)
    conda_env_path = os.path.join(path, _CONDA_ENV_FILE_NAME)
    model_file_path = os.path.join(path, MLMODEL_FILE_NAME)

    if not os.path.exists(path):
        os.makedirs(path)
    _download_artifact_from_uri(model_uri, output_path=path)

    if not os.path.exists(wheels_dir):
        os.mkdir(wheels_dir)
    else:
        raise MlflowException(
            message=("Model with model_uri: {} already has wheels packaged.".format(model_uri)),
            error_code=RESOURCE_ALREADY_EXISTS,
        )

    _download_wheels(dst_path=wheels_dir, pip_requirements_path=pip_requirements_path)
    _overwrite_pip_requirements_with_wheels(
        wheels_dir=wheels_dir, pip_requirements_path=pip_requirements_path
    )
    _update_conda_file_with_wheels(
        conda_env_path=conda_env_path, pip_requirements_path=pip_requirements_path
    )

    # Updating MLModel File
    if mlflow_model:
        _update_model_file(mlflow_model=mlflow_model, original_model_file_path=model_file_path)


def _update_model_file(mlflow_model, original_model_file_path):
    with open(original_model_file_path) as f:
        model_file = yaml.safe_load(f)

    model_file["run_id"] = mlflow_model.run_id
    model_file["utc_time_created"] = mlflow_model.utc_time_created
    model_file["flavors"]["python_function"]["artifacts"]["wheels"] = True

    with open(original_model_file_path, "w") as out:
        yaml.safe_dump(model_file, stream=out, default_flow_style=False)


# Rename this
def _get_list_from_file(path):
    with open(path, "r") as file:
        return file.read().splitlines()


def _get_pip_requirements_list(path):
    return _get_list_from_file(path)


def _download_wheels(dst_path, pip_requirements_path):
    download_command = (
        f"python -m pip wheel --only-binary=:all: --wheel-dir={dst_path} -r {pip_requirements_path}"
    )
    os.system(download_command)


def _overwrite_pip_requirements_with_wheels(wheels_dir, pip_requirements_path):
    with open(pip_requirements_path, "w") as wheels_requirements:
        for wheel_file in os.listdir(wheels_dir):
            # Ignore files that are not wheels (Eg: .DS_STORE)
            if wheel_file.endswith(".whl"):
                complete_wheel_file = os.path.join(
                    _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME, wheel_file
                )
                wheels_requirements.write(complete_wheel_file + "\n")


def _update_conda_file_with_wheels(conda_env_path, pip_requirements_path):
    with open(conda_env_path) as f:
        conda_env = yaml.safe_load(f)

    new_conda_env = _overwrite_pip_deps(
        conda_env, _get_pip_requirements_list(pip_requirements_path)
    )

    with open(conda_env_path, "w") as out:
        yaml.safe_dump(new_conda_env, stream=out, default_flow_style=False)

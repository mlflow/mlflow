"""
The `wheeled_model` flavor serves as a convenience interface enabling users to
re-log registered models back to MLflow or the model registry with all the wheels
required by the model bundled in.
"""

import os
from datetime import datetime
import yaml
import logging
import mlflow
from mlflow.pyfunc.model import Model
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _overwrite_pip_deps,
)

_logger = logging.getLogger(__name__)

_ARTIFACTS_FOLDER_NAME = "artifacts"
_WHEELS_FOLDER_NAME = "wheels"

FLAVOR_NAME = "wheeled_model"


# TODO: Verify the arguments with Arjun
def log_model(artifact_path, model_uri, registered_model_name=None):
    """
    This re-logs / re-registers the model with `model_uri` to mlflow / model registry with all
    required wheels.

    :param artifact_path: The run-relative artifact path to which to log the Python model.
    :param model_uri: URI referring to the MLmodel directory. Only ``models:/`` URIs are
                      currently supported
    :param registered_model_name: If given, create a model version under ``registered_model_name``,
                                  also creating a registered model if one with the given name does
                                  not exist.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.wheeled_model,
        registered_model_name=registered_model_name,
        model_uri=model_uri,
    )


# TODO: Verify the arguments with Arjun
# TODO: I updated the save model API, I don't want users passing in their own mlflow_model
def save_model(path, model_uri, **kwargs):
    """
    Saves model registered at `model_uri` to `path` along with all the required wheels.

    :param path: The path to which to save the packaged model.
    :param model_uri: URI referring to the MLmodel directory. Only ``models:/`` URIs are
                  currently supported
    """
    mlflow_model = kwargs.pop("mlflow_model", None)
    if len(kwargs) > 0:
        raise TypeError("save_model() got unexpected keyword arguments: {}".format(kwargs))

    if not os.path.exists(path):
        os.makedirs(path)

    # Check to see that model_uri is of the form `models:/`
    _, _, _ = _parse_model_uri(model_uri)

    local_path = _download_artifact_from_uri(model_uri, output_path=path)

    wheels_dir = os.path.join(local_path, _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME)
    pip_requirements_path = os.path.join(local_path, _REQUIREMENTS_FILE_NAME)
    conda_env_path = os.path.join(local_path, _CONDA_ENV_FILE_NAME)
    model_file_path = os.path.join(local_path, MLMODEL_FILE_NAME)

    # Check if the model file has `wheels` set to True
    try:
        with open(model_file_path) as f:
            model_file = yaml.safe_load(f)
        has_wheels = model_file["flavors"]["python_function"]["artifacts"]["wheels"]
    except KeyError:
        has_wheels = False

    if has_wheels:
        _logger.warning("This model already has packaged wheels. New wheels will not be packaged.")
    else:
        # Download the wheels required by packages in requirements.txt
        _download_wheels(dst_path=wheels_dir, pip_requirements_path=pip_requirements_path)

        # Update requirements.txt with wheels
        _overwrite_pip_requirements_with_wheels(
            wheels_dir=wheels_dir, pip_requirements_path=pip_requirements_path
        )

        # Update conda.yaml file with wheels
        _update_conda_file_with_wheels(
            conda_env_path=conda_env_path, pip_requirements_path=pip_requirements_path
        )

    # Update MLModel File
    _update_model_file(mlflow_model=mlflow_model, original_model_file_path=model_file_path)

    # If mlflow_model == None, then just declare a new one.
    # This occurs when the users explicitly uses save_model()
    # If mlflow_model != None, it was created and passed in to save_model() in Model.log().

    if not mlflow_model:
        # Loading the MLmodel file into a new Model() object.
        mlflow_model = Model.load(path=model_file_path)
    else:
        # Copy all the attributes from the updated model file to the original mlflow object
        # created in Model.log()
        mlflow_model.__dict__ = Model.load(path=model_file_path).__dict__.copy()

    return mlflow_model


def _update_model_file(mlflow_model, original_model_file_path):
    """
    Updates the MLModel file with the correct run_id, and utc_time_created. Additionally,
    this also adds `wheels` to the list of artifacts.

    :param mlflow_model: :py:mod:`mlflow.models.Model` configuration to which to add the
                         **python_function** flavor.
    :param original_model_file_path: Path to the original model file
    """
    with open(original_model_file_path) as f:
        model_file = yaml.safe_load(f)

    # Update model_file with the run and utc_time_create of the newly logged model
    if mlflow_model:
        # When the user uses log_model(), Model.log() inherently creates a new a run_id and
        # mlflow_model with the time that it was created. This is used to update the run_id
        # and utc_time_created in the MLmodel file.
        model_file["run_id"] = mlflow_model.run_id
        model_file["utc_time_created"] = mlflow_model.utc_time_created
    else:
        # When the user uses save_model(), mlflow_model the value of mlflow_model == None.
        # In this case we delete the run_id from the original MLmodel file since this save
        # was not a part of any run and the older run_id is irrelevant.
        del model_file["run_id"]

        # utc_time_created in the model file updated to current time
        model_file["utc_time_created"] = datetime.utcnow()

    # TODO: Do we not want to add the path here?
    # Add wheels to artifacts in the MLModel file
    if "artifacts" not in model_file["flavors"]["python_function"]:
        model_file["flavors"]["python_function"]["artifacts"] = {}

    model_file["flavors"]["python_function"]["artifacts"]["wheels"] = True

    with open(original_model_file_path, "w") as out:
        yaml.safe_dump(model_file, stream=out, default_flow_style=False)


def _get_list_from_file(path):
    with open(path, "r") as file:
        return file.read().splitlines()


def _get_pip_requirements_list(path):
    return _get_list_from_file(path)


# TODO: Do we want to give the users option to pass in whatever extra command
#       line arguments that they want?
def _download_wheels(dst_path, pip_requirements_path):
    """
    Downloads all the wheels of the dependencies specified in the requirements.txt file

    :param dst_path: Path to the directory where the wheels are to be downloaded
    :param pip_requirements_path: Path to requirements.txt in the model directory
    """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    download_command = (
        f"python -m pip wheel --only-binary=:all: --wheel-dir={dst_path} -r {pip_requirements_path}"
    )
    os.system(download_command)


def _overwrite_pip_requirements_with_wheels(wheels_dir, pip_requirements_path):
    """
    Overwrites the requirements.txt with the wheels of the required dependencies.

    :param wheels_dir: Path to directory where wheels are stored
    :param pip_requirements_path: Path to requirements.txt in the model directory
    """
    with open(pip_requirements_path, "w") as wheels_requirements:
        for wheel_file in os.listdir(wheels_dir):
            # Ignore files that are not wheels (Eg: .DS_STORE)
            if wheel_file.endswith(".whl"):
                complete_wheel_file = os.path.join(
                    _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME, wheel_file
                )
                wheels_requirements.write(complete_wheel_file + "\n")


def _update_conda_file_with_wheels(conda_env_path, pip_requirements_path):
    """
    Updates the list pip packages in the conda.yaml file to the list of wheels in the wheels
    directory.

    {
        "name": "env",
        "channels": [...],
        "dependencies": [
            ...,
            "pip",
            {"pip": [...]},  <- Overwrite this with list of wheels
        ],
    }

    :param conda_env_path: Path to conda.yaml file in the model directory
    :param pip_requirements_path: Path to requirements.txt in the model directory
    """
    with open(conda_env_path) as f:
        conda_env = yaml.safe_load(f)

    new_conda_env = _overwrite_pip_deps(
        conda_env, _get_pip_requirements_list(pip_requirements_path)
    )

    with open(conda_env_path, "w") as out:
        yaml.safe_dump(new_conda_env, stream=out, default_flow_style=False)

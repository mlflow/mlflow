import os
import platform
import shutil
import subprocess
import sys

import yaml

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.pyfunc.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.utils.models import _parse_model_uri, get_model_name_and_version
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
    _REQUIREMENTS_FILE_NAME,
    _get_pip_deps,
    _mlflow_additional_pip_env,
    _overwrite_pip_deps,
)
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

_WHEELS_FOLDER_NAME = "wheels"
_ORIGINAL_REQ_FILE_NAME = "original_requirements.txt"
_PLATFORM = "platform"


@experimental
class WheeledModel:
    """
    Helper class to create a model with added dependency wheels from an existing registered model.
    The `wheeled` model contains all the model dependencies as wheels stored as model artifacts.
    .. note::
        This utility only operates on a model that has been registered to the Model Registry.
    """

    def __init__(self, model_uri):
        self._model_uri = model_uri
        databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(model_uri) or mlflow.get_registry_uri()
        )
        client = MlflowClient(registry_uri=databricks_profile_uri)
        self._model_name, _ = get_model_name_and_version(client, model_uri)

    @classmethod
    def log_model(cls, model_uri, registered_model_name=None):
        """
        Logs a registered model as an MLflow artifact for the current run. This only operates on
        a model which has been registered to the Model Registry. Given a registered model_uri (
        e.g. models:/<model_name>/<model_version>), this utility re-logs the model along with all
        the required model libraries back to the Model Registry. The required model libraries are
        stored along with the model as model artifacts. In addition, supporting files to the
        model (e.g. conda.yaml, requirements.txt) are modified to use the added libraries.

        By default, this utility creates a new model version under the same registered model
        specified by ``model_uri``. This behavior can be overridden by specifying the
        ``registered_model_name`` argument.

        Args:
            model_uri: A registered model uri in the Model Registry of the form
                       models:/<model_name>/<model_version/stage/latest>
            registered_model_name: The new model version (model with its libraries) is
                                   registered under the inputted registered_model_name. If None,
                                   a new version is logged to the existing model in the Model
                                   Registry.

        .. code-block:: python
            :caption: Example

            # Given a model uri, log the wheeled model
            with mlflow.start_run():
                WheeledModel.log_model(model_uri)
        """
        parsed_uri = _parse_model_uri(model_uri)
        return Model.log(
            artifact_path=None,
            flavor=WheeledModel(model_uri),
            registered_model_name=registered_model_name or parsed_uri.name,
        )

    def save_model(self, path, mlflow_model=None):
        """
        Given an existing registered model, saves the model along with it's dependencies stored as
        wheels to a path on the local file system.

        This does not modify existing model behavior or existing model flavors. It simply downloads
        the model dependencies as wheels and modifies the requirements.txt and conda.yaml file to
        point to the downloaded wheels.

        The download_command defaults to downloading only binary packages using the
        `--only-binary=:all:` option. This behavior can be overridden using an environment
        variable `MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS`, which will allows setting
        different options such as `--prefer-binary`, `--no-binary`, etc.

        Args:
            path: Local path where the model is to be saved.
            mlflow_model: The new :py:mod:`mlflow.models.Model` metadata file to store the
                updated model metadata.
        """
        from mlflow.pyfunc import ENV, FLAVOR_NAME, _extract_conda_env

        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)

        local_model_path = _download_artifact_from_uri(self._model_uri, output_path=path)

        wheels_dir = os.path.join(local_model_path, _WHEELS_FOLDER_NAME)
        pip_requirements_path = os.path.join(local_model_path, _REQUIREMENTS_FILE_NAME)
        model_metadata_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)

        model_metadata = Model.load(model_metadata_path)

        # Check if the model file has `wheels` set to True
        if model_metadata.__dict__.get(_WHEELS_FOLDER_NAME, None) is not None:
            raise MlflowException("Model libraries are already added", BAD_REQUEST)

        conda_env = _extract_conda_env(model_metadata.flavors.get(FLAVOR_NAME, {}).get(ENV, None))
        conda_env_path = os.path.join(local_model_path, conda_env)
        if conda_env is None and not os.path.isfile(pip_requirements_path):
            raise MlflowException(
                "Cannot add libraries for model with no logged dependencies.", BAD_REQUEST
            )

        if not os.path.isfile(pip_requirements_path):
            self._create_pip_requirement(conda_env_path, pip_requirements_path)

        WheeledModel._download_wheels(
            pip_requirements_path=pip_requirements_path, dst_path=wheels_dir
        )

        # Keep a copy of the original requirement.txt
        shutil.copy2(pip_requirements_path, os.path.join(local_model_path, _ORIGINAL_REQ_FILE_NAME))

        # Update requirements.txt with wheels
        pip_deps = self._overwrite_pip_requirements_with_wheels(
            pip_requirements_path=pip_requirements_path, wheels_dir=wheels_dir
        )

        # Update conda.yaml with wheels
        self._update_conda_env(pip_deps, conda_env_path)

        # Update MLModel File
        mlflow_model = self._update_mlflow_model(
            original_model_metadata=model_metadata, mlflow_model=mlflow_model
        )
        mlflow_model.save(model_metadata_path)
        return mlflow_model

    def _update_conda_env(self, new_pip_deps, conda_env_path):
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

        Args:
            new_pip_deps: List of pip dependencies as wheels
            conda_env_path: Path to conda.yaml file in the model directory
        """
        with open(conda_env_path) as f:
            conda_env = yaml.safe_load(f)

        new_conda_env = _overwrite_pip_deps(conda_env, new_pip_deps)

        with open(conda_env_path, "w") as out:
            yaml.safe_dump(new_conda_env, stream=out, default_flow_style=False)

    def _update_mlflow_model(self, original_model_metadata, mlflow_model):
        """
        Modifies the MLModel file to reflect updated information such as the run_id,
        utc_time_created. Additionally, this also adds `wheels` to the MLModel file to indicate that
        this is a `wheeled` model.

        Args:
            original_model_metadata: The model metadata stored in the original MLmodel file.
            mlflow_model: :py:mod:`mlflow.models.Model` configuration of the newly created
                          wheeled model
        """

        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        if mlflow_model is None:
            mlflow_model = Model(run_id=run_id)

        original_model_metadata.__dict__.update(
            {k: v for k, v in mlflow_model.__dict__.items() if v}
        )
        mlflow_model.__dict__.update(original_model_metadata.__dict__)
        mlflow_model.artifact_path = WheeledModel.get_wheel_artifact_path(
            mlflow_model.artifact_path
        )

        mlflow_model.wheels = {_PLATFORM: platform.platform()}
        return mlflow_model

    @classmethod
    def _download_wheels(cls, pip_requirements_path, dst_path):
        """
        Downloads all the wheels of the dependencies specified in the requirements.txt file.
        The pip wheel download_command defaults to downloading only binary packages using
        the `--only-binary=:all:` option. This behavior can be overridden using an
        environment variable `MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS`, which will allows
        setting different options such as `--prefer-binary`, `--no-binary`, etc.

        Args:
            pip_requirements_path: Path to requirements.txt in the model directory
            dst_path: Path to the directory where the wheels are to be downloaded
        """
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        pip_wheel_options = MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS.get()

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    pip_wheel_options,
                    "--wheel-dir",
                    dst_path,
                    "-r",
                    pip_requirements_path,
                    "--no-cache-dir",
                    "--progress-bar=off",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raise MlflowException(
                f"An error occurred while downloading the dependency wheels: {e.stdout}"
            )

    def _overwrite_pip_requirements_with_wheels(self, pip_requirements_path, wheels_dir):
        """
        Overwrites the requirements.txt with the wheels of the required dependencies.

        Args:
            pip_requirements_path: Path to requirements.txt in the model directory.
            wheels_dir: Path to directory where wheels are stored.
        """
        wheels = []
        with open(pip_requirements_path, "w") as wheels_requirements:
            for wheel_file in os.listdir(wheels_dir):
                if wheel_file.endswith(".whl"):
                    complete_wheel_file = os.path.join(_WHEELS_FOLDER_NAME, wheel_file)
                    wheels.append(complete_wheel_file)
                    wheels_requirements.write(complete_wheel_file + "\n")
        return wheels

    def _create_pip_requirement(self, conda_env_path, pip_requirements_path):
        """
        This method creates a requirements.txt file for the model dependencies if the file does not
        already exist. It uses the pip dependencies found in the conda.yaml env file.

        Args:
            conda_env_path: Path to conda.yaml env file which contains the required pip
                dependencies
            pip_requirements_path: Path where the new requirements.txt will be created.
        """
        with open(conda_env_path) as f:
            conda_env = yaml.safe_load(f)
        pip_deps = _get_pip_deps(conda_env)
        _mlflow_additional_pip_env(pip_deps, pip_requirements_path)

    @classmethod
    def get_wheel_artifact_path(cls, original_artifact_path):
        return original_artifact_path + "_" + _WHEELS_FOLDER_NAME

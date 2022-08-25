import os

import mlflow
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.pyfunc.model import Model, MLMODEL_FILE_NAME
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.utils.environment import (
    _REQUIREMENTS_FILE_NAME,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _validate_and_prepare_target_save_path

_WHEELS_FOLDER_NAME = "wheels"


class WheeledModel:
    """
    Helper class to create a wheeled model from an existing registered model. The wheeled model
    contains all the model dependencies as wheels stored as model artifacts.
    """

    def __init__(self, model_uri):
        self._model_uri = model_uri
        self._model_name, _, _ = _parse_model_uri(model_uri)  # Throws exception if not a model uri

    @classmethod
    def log_model(cls, artifact_path, model_uri, registered_model_name=None):
        """
        Logs a registered model as an MLflow artifact for the current run. This function will take
        an existing model and re-log the model along with the wheels of all the
        the model dependencies (stored along with the model artifacts).

        The default behavior is to log a new model version to an existing registered model, however,
        the caller can choose to log the model wheels to a different or new registered model.

        :param artifact_path: Run-relative artifact path.
        :param model_uri: registered model uri of the form
                            models:/<model_name>/<model_version/stage/latest>
        :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.
        """
        model_name, _, _ = _parse_model_uri(model_uri)
        return Model.log(
            artifact_path=artifact_path,
            flavor=WheeledModel(model_uri),
            registered_model_name=registered_model_name or model_name,
        )

    def save_model(self, path, mlflow_model=None):
        """
        Given an existing registered model, saves the model along with it's dependencies stored as
        wheels to a path on the local file system.

        This does not modify existing model behavior or
        existing model flavors. It simply downloads the model dependencies as wheels and modifies
        the requirements.txt and conda.yaml file to point to the downloaded wheels.
        :param path: Local path where the model is to be saved.
        :param mlflow_model: The new :py:mod:`mlflow.models.Model` metadata file to store the
                            updated model metadata.
        """
        from mlflow.pyfunc import FLAVOR_NAME, ENV

        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)

        local_model_path = _download_artifact_from_uri(self._model_uri, output_path=path)

        wheels_dir = os.path.join(local_model_path, _WHEELS_FOLDER_NAME)
        pip_requirements_path = os.path.join(local_model_path, _REQUIREMENTS_FILE_NAME)
        model_metadata_path = os.path.join(local_model_path, MLMODEL_FILE_NAME)

        model_metadata = Model.load(model_metadata_path)

        # Check if the model file has `wheels` set to True
        if model_metadata.__dict__.get(_WHEELS_FOLDER_NAME, None):
            raise MlflowException("Cannot add wheels to a wheeled model", BAD_REQUEST)

        conda_env = model_metadata.flavors.get(FLAVOR_NAME, {}).get(ENV, None)
        conda_env_path = os.path.join(local_model_path, conda_env)
        if conda_env is None:
            raise MlflowException(
                "Can not add wheels for model with no conda environment.", BAD_REQUEST
            )
        if not os.path.isfile(pip_requirements_path):
            raise Exception("Can not add wheels for model with no 'requirements.txt'.")

        self._download_wheels(dst_path=wheels_dir, pip_requirements_path=pip_requirements_path)

        # Update requirements.txt with wheels
        pip_wheels = self._overwrite_pip_requirements_with_wheels(
            wheels_dir=wheels_dir, pip_requirements_path=pip_requirements_path
        )

        _mlflow_conda_env(path=conda_env_path, additional_pip_deps=pip_wheels, install_mlflow=False)

        # Update MLModel File
        mlflow_model = self._update_mlflow_model(
            mlflow_model=mlflow_model, original_model_metadata=model_metadata
        )
        mlflow_model.save(model_metadata_path)
        return mlflow_model

    def _update_mlflow_model(self, mlflow_model, original_model_metadata):
        """
        Modifies the MLModel file to reflect updated information such as the run_id,
        utc_time_created. Additionally, this also adds `wheels` to the MLModel file to indicate that
        this is a `wheeled` model.
        :param  mlflow_model: :py:mod:`mlflow.models.Model` configuration of the newly created
                                wheeled model
        :param original_model_file_path: The model metadata stored in the original MLmodel file.
        """

        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        if mlflow_model is None:
            mlflow_model = Model(run_id=run_id)

        original_model_metadata.__dict__.update(
            {k: v for k, v in mlflow_model.__dict__.items() if v}
        )
        mlflow_model.__dict__.update(original_model_metadata.__dict__)

        mlflow_model.wheels = _WHEELS_FOLDER_NAME
        return mlflow_model

    def _download_wheels(self, dst_path, pip_requirements_path):
        """
        Downloads all the wheels of the dependencies specified in the requirements.txt file
        :param dst_path: Path to the directory where the wheels are to be downloaded
        :param pip_requirements_path: Path to requirements.txt in the model directory
        """
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        download_command = (
            f"python -m pip wheel --only-binary=:all: --wheel-dir={dst_path} -r"
            f"{pip_requirements_path} --no-cache-dir"
        )
        rc = os.system(download_command)
        if rc != 0:
            raise MlflowException("Error downloading dependency wheels")

    def _overwrite_pip_requirements_with_wheels(self, wheels_dir, pip_requirements_path):
        """
        Overwrites the requirements.txt with the wheels of the required dependencies.
        :param wheels_dir: Path to directory where wheels are stored
        :param pip_requirements_path: Path to requirements.txt in the model directory
        """
        wheels = []
        with open(pip_requirements_path, "w") as wheels_requirements:
            for wheel_file in os.listdir(wheels_dir):
                if wheel_file.endswith(".whl"):
                    complete_wheel_file = os.path.join(_WHEELS_FOLDER_NAME, wheel_file)
                    wheels.append(complete_wheel_file)
                    wheels_requirements.write(complete_wheel_file + "\n")
        return wheels

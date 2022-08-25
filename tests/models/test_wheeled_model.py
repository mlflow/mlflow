import pytest
from sklearn import datasets, neighbors
import yaml
import mlflow
import random
import os
import re
from mlflow.models.wheeled_model import WheeledModel, _WHEELS_FOLDER_NAME
from mlflow.pyfunc.model import Model, MLMODEL_FILE_NAME
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _is_pip_deps
from mlflow.store.artifact.utils.models import _improper_model_uri_msg
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
)


def iris_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


@pytest.fixture(scope="module")
def sklearn_knn_model():
    x, y = iris_data()
    knn_model = neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


def _get_list_from_file(path):
    with open(path, "r") as file:
        return file.read().splitlines()


def _get_pip_requirements_list(path):
    return _get_list_from_file(path)


def get_pip_requirements_from_conda_file(conda_env_path):
    with open(conda_env_path) as f:
        conda_env = yaml.safe_load(f)

    conda_pip_requirements_list = []
    dependencies = conda_env.get("dependencies")

    for dependency in dependencies:
        if _is_pip_deps(dependency):
            conda_pip_requirements_list = dependency["pip"]

    return conda_pip_requirements_list


def validate_updated_model_file(original_model_config, wheeled_model_config):
    differing_keys = {"run_id", "utc_time_created", "model_uuid"}

    # Compare wheeled model configs with original model config (MLModel files)
    for key in original_model_config:
        if key not in differing_keys:
            assert wheeled_model_config[key] == original_model_config[key]
        else:
            # `run_id` and `utc_time_created` should be different
            assert wheeled_model_config[key] != original_model_config[key]

    # Wheeled model key should only exist in wheeled_model_config
    assert wheeled_model_config.get(_WHEELS_FOLDER_NAME, None)
    assert not original_model_config.get(_WHEELS_FOLDER_NAME, None)

    # Every key in the original config should also exist in the wheeled config.
    for key in original_model_config:
        assert key in wheeled_model_config


def validate_updated_conda_dependencies(original_model_path, wheeled_model_path):
    # Check if conda.yaml files of the original model and wheeled model are the same
    # excluding the dependencies
    wheeled_model_path = os.path.join(wheeled_model_path, _CONDA_ENV_FILE_NAME)
    original_conda_env_path = os.path.join(original_model_path, _CONDA_ENV_FILE_NAME)

    with open(wheeled_model_path) as wheeled_conda_env, open(
        original_conda_env_path
    ) as original_conda_env:
        wheeled_conda_env = yaml.safe_load(wheeled_conda_env)
        original_conda_env = yaml.safe_load(original_conda_env)

        for key in wheeled_conda_env:
            if key != "dependencies":
                assert wheeled_conda_env[key] == original_conda_env[key]
            else:
                assert wheeled_conda_env[key] != original_conda_env[key]


def validate_wheeled_dependencies(wheeled_model_path):
    # Check if conda.yaml and requirements.txt are consistent
    pip_requirements_path = os.path.join(wheeled_model_path, _REQUIREMENTS_FILE_NAME)
    pip_requirements_list = _get_pip_requirements_list(pip_requirements_path)
    conda_pip_requirements_list = get_pip_requirements_from_conda_file(
        os.path.join(wheeled_model_path, _CONDA_ENV_FILE_NAME)
    )

    pip_requirements_list.sort()
    conda_pip_requirements_list.sort()
    assert pip_requirements_list == conda_pip_requirements_list

    # Check if requirements.txt and wheels directory are consistent
    wheels_dir = os.path.join(wheeled_model_path, _WHEELS_FOLDER_NAME)
    wheels_list = []
    for wheel_file in os.listdir(wheels_dir):
        if wheel_file.endswith(".whl"):
            relative_wheel_path = os.path.join(_WHEELS_FOLDER_NAME, wheel_file)
            wheels_list.append(relative_wheel_path)

    wheels_list.sort()
    assert wheels_list == pip_requirements_list


def test_model_log_load(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"
    artifact_path = "model"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )
        model_path = _download_artifact_from_uri(model_uri)
        original_model_config = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME)).__dict__

    # Re-log with wheels
    with mlflow.start_run():
        WheeledModel.log_model(
            artifact_path=artifact_path,
            model_uri=model_uri,
        )
        wheeled_model_path = _download_artifact_from_uri(wheeled_model_uri)
        wheeled_model_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_config = Model.load(
            os.path.join(wheeled_model_path, MLMODEL_FILE_NAME)
        ).__dict__

    validate_updated_model_file(original_model_config, wheeled_model_config)
    # Assert correct run_id
    assert wheeled_model_config["run_id"] == wheeled_model_run_id

    validate_updated_conda_dependencies(model_path, wheeled_model_path)

    validate_wheeled_dependencies(wheeled_model_path)


def test_model_save_load(tmp_path, sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )

        model_path = _download_artifact_from_uri(model_uri)
        original_model_config = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME)).__dict__

    # Save with wheels
    wheeled_model_path = os.path.join(tmp_path, "model")
    with mlflow.start_run():
        wheeled_model = WheeledModel(model_uri=model_uri)
        wheeled_model_data = wheeled_model.save_model(path=wheeled_model_path)
        wheeled_model_config = Model.load(os.path.join(wheeled_model_path, MLMODEL_FILE_NAME))
        wheeled_model_config_dict = wheeled_model_config.__dict__

        # Check to see if python model returned is the same as the MLModel file
        assert wheeled_model_config == wheeled_model_data

    validate_updated_model_file(original_model_config, wheeled_model_config_dict)
    validate_updated_conda_dependencies(model_path, wheeled_model_path)
    validate_wheeled_dependencies(wheeled_model_path)


def test_logging_and_saving_wheeled_model_throws(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"
    model_path = "model"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=model_path,
            registered_model_name=model_name,
        )

    # Re-log with wheels
    with mlflow.start_run():
        WheeledModel.log_model(
            artifact_path=model_path,
            model_uri=model_uri,
        )

    match = "Cannot add wheels to a wheeled model"

    # Log wheeled model
    with pytest.raises(MlflowException, match=re.escape(match)):
        with mlflow.start_run():
            WheeledModel.log_model(
                artifact_path=model_path,
                model_uri=wheeled_model_uri,
            )

    # Saved a wheeled model
    with pytest.raises(MlflowException, match=re.escape(match)):
        with mlflow.start_run():
            WheeledModel(wheeled_model_uri)


def test_log_model_with_non_model_uri():
    model_uri = "runs:/beefe0b6b5bd4acf9938244cdc006b64/model"

    # Log with wheels
    wheeled_artifact_path = "model"
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        with mlflow.start_run():
            WheeledModel.log_model(
                artifact_path=wheeled_artifact_path,
                model_uri=model_uri,
            )

    # Save with wheels
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        with mlflow.start_run():
            WheeledModel(model_uri)

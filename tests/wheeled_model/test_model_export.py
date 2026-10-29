import pytest
import random
from sklearn import datasets, neighbors
import yaml
import mlflow
import os
import re
from unittest import mock
from mlflow import wheeled_model
from mlflow.models import Model
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _is_pip_deps
from mlflow.store.artifact.utils.models import _improper_model_uri_msg
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
)

_ARTIFACTS_FOLDER_NAME = "artifacts"
_WHEELS_FOLDER_NAME = "wheels"


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


def test_model_log_load(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )

        model_path = _download_artifact_from_uri(model_uri)
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        model_config_dict = model_config.__dict__

    # Re-log with wheels
    wheeled_artifact_path = "model"
    with mlflow.start_run():
        wheeled_model.log_model(
            artifact_path=wheeled_artifact_path,
            registered_model_name=model_name,
            model_uri=model_uri,
        )

        wheeled_model_path = _download_artifact_from_uri(wheeled_model_uri)
        wheeled_model_config = Model.load(os.path.join(wheeled_model_path, "MLmodel"))
        wheeled_model_config_dict = wheeled_model_config.__dict__

    differing_keys = {"flavors", "run_id", "utc_time_created"}

    # Compare wheeled model configs with original model config (MLModel files)
    for key in wheeled_model_config_dict:
        if key not in differing_keys:
            assert wheeled_model_config_dict[key] == model_config_dict[key]
        elif key == "flavors":
            # There should be a key `wheels` under python_function artifacts set to True
            assert (
                "wheels" in wheeled_model_config_dict[key]["python_function"]["artifacts"]
                and wheeled_model_config_dict[key]["python_function"]["artifacts"]["wheels"]
            )

            # There should be no other change in `flavors`. Deleting the key `wheels` to
            # verify that the rest of the dictionary of the wheeled model is identical to
            # the dictionary of the non-wheeled model
            temp_dict = wheeled_model_config_dict[key]
            del wheeled_model_config_dict[key]["python_function"]["artifacts"]["wheels"]
            # If the `artifacts` sub-dictionary is empty after deleting the key `wheels`,
            # delete the key 'artifacts`
            if wheeled_model_config_dict[key]["python_function"]["artifacts"] == {}:
                del wheeled_model_config_dict[key]["python_function"]["artifacts"]

            assert temp_dict == model_config_dict[key]
        else:
            # `run_id` and `utc_time_created` should be different
            assert wheeled_model_config_dict[key] != model_config_dict[key]

    for key in model_config_dict:
        assert key in wheeled_model_config_dict

    # Check if conda.yaml files of the original model and wheeled model are the same
    # excluding the dependencies
    conda_env_path = os.path.join(wheeled_model_path, _CONDA_ENV_FILE_NAME)
    original_conda_env_path = os.path.join(model_path, _CONDA_ENV_FILE_NAME)

    with open(conda_env_path) as conda_env, open(original_conda_env_path) as original_conda_env:
        conda_env = yaml.safe_load(conda_env)
        original_conda_env = yaml.safe_load(original_conda_env)

        for key in conda_env:
            if key != "dependencies":
                assert conda_env[key] == original_conda_env[key]
            else:
                assert conda_env[key] != original_conda_env[key]

    # Check if conda.yaml and requirements.txt are consistent
    pip_requirements_path = os.path.join(wheeled_model_path, _REQUIREMENTS_FILE_NAME)
    pip_requirements_list = _get_pip_requirements_list(pip_requirements_path)
    conda_pip_requirements_list = get_pip_requirements_from_conda_file(conda_env_path)

    pip_requirements_list.sort()
    conda_pip_requirements_list.sort()
    assert pip_requirements_list == conda_pip_requirements_list

    # Check if requirements.txt and wheels directory are consistent
    wheels_dir = os.path.join(wheeled_model_path, _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME)
    wheels_list = []
    for wheel_file in os.listdir(wheels_dir):
        if wheel_file.endswith(".whl"):
            relative_wheel_path = os.path.join(
                _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME, wheel_file
            )
            wheels_list.append(relative_wheel_path)

    wheels_list.sort()
    assert wheels_list == pip_requirements_list


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
        model_config = Model.load(os.path.join(model_path, "MLmodel"))
        model_config_dict = model_config.__dict__

    # Save with wheels
    wheeled_model_path = os.path.join(tmp_path, "model")
    with mlflow.start_run():
        wheeled_model_data = wheeled_model.save_model(path=wheeled_model_path, model_uri=model_uri)

        wheeled_model_config = Model.load(os.path.join(wheeled_model_path, "MLmodel"))
        wheeled_model_config_dict = wheeled_model_config.__dict__

        # Check to see if python model returned is the same as the MLModel file
        assert wheeled_model_config == wheeled_model_data

    differing_keys = {"flavors", "utc_time_created"}
    deleted_keys = {"run_id", "artifact_path"}

    # Compare wheeled model configs with original model config (MLModel files)
    for key in wheeled_model_config_dict:
        if key not in differing_keys:
            assert wheeled_model_config_dict[key] == model_config_dict[key]
        elif key == "flavors":
            # There should be a key `wheels` under python_function artifacts set to True
            assert (
                "wheels" in wheeled_model_config_dict[key]["python_function"]["artifacts"]
                and wheeled_model_config_dict[key]["python_function"]["artifacts"]["wheels"]
            )

            # There should be no other change in `flavors`. Deleting the key `wheels` to
            # verify that the rest of the dictionary of the wheeled model is identical to
            # the dictionary of the non-wheeled model
            temp_dict = wheeled_model_config_dict[key]
            del wheeled_model_config_dict[key]["python_function"]["artifacts"]["wheels"]
            # If the `artifacts` sub-dictionary is empty after deleting the key `wheels`,
            # delete the key 'artifacts`
            if wheeled_model_config_dict[key]["python_function"]["artifacts"] == {}:
                del wheeled_model_config_dict[key]["python_function"]["artifacts"]

            assert temp_dict == model_config_dict[key]
        else:
            # `utc_time_created` should be different
            assert wheeled_model_config_dict[key] != model_config_dict[key]

    for key in model_config_dict:
        if key not in deleted_keys:
            assert key in wheeled_model_config_dict
        else:
            assert key not in wheeled_model_config_dict

    # Check if conda.yaml files of the original model and wheeled model are the same
    # excluding the dependencies
    conda_env_path = os.path.join(wheeled_model_path, _CONDA_ENV_FILE_NAME)
    original_conda_env_path = os.path.join(model_path, _CONDA_ENV_FILE_NAME)

    with open(conda_env_path) as conda_env, open(original_conda_env_path) as original_conda_env:
        conda_env = yaml.safe_load(conda_env)
        original_conda_env = yaml.safe_load(original_conda_env)

        for key in conda_env:
            if key != "dependencies":
                assert conda_env[key] == original_conda_env[key]
            else:
                assert conda_env[key] != original_conda_env[key]

    # Check if conda.yaml and requirements.txt are consistent
    pip_requirements_path = os.path.join(wheeled_model_path, _REQUIREMENTS_FILE_NAME)
    pip_requirements_list = _get_pip_requirements_list(pip_requirements_path)
    conda_pip_requirements_list = get_pip_requirements_from_conda_file(conda_env_path)

    pip_requirements_list.sort()
    conda_pip_requirements_list.sort()

    assert pip_requirements_list == conda_pip_requirements_list

    # Check if requirements.txt and wheels directory are consistent
    wheels_dir = os.path.join(wheeled_model_path, _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME)
    wheels_list = []
    for wheel_file in os.listdir(wheels_dir):
        if wheel_file.endswith(".whl"):
            relative_wheel_path = os.path.join(
                _ARTIFACTS_FOLDER_NAME, _WHEELS_FOLDER_NAME, wheel_file
            )
            wheels_list.append(relative_wheel_path)

    wheels_list.sort()
    assert wheels_list == pip_requirements_list


def test_logging_wheeled_model(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )

    # Re-log with wheels
    wheeled_artifact_path = "model"
    with mlflow.start_run():
        wheeled_model.log_model(
            artifact_path=wheeled_artifact_path,
            registered_model_name=model_name,
            model_uri=model_uri,
        )

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    with mock.patch("mlflow.wheeled_model._logger.warning") as warn_mock:
        warn_mock.side_effect = custom_warn

        # Log wheeled model
        double_wheeled_artifact_path = "model"
        with mlflow.start_run():
            wheeled_model.log_model(
                artifact_path=double_wheeled_artifact_path,
                registered_model_name=model_name,
                model_uri=wheeled_model_uri,
            )

    assert (
        "This model already has packaged wheels. New wheels will not be packaged." in log_messages
    )


def test_saving_wheeled_model(tmp_path, sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )

    # Re-log with wheels
    wheeled_artifact_path = "model"
    with mlflow.start_run():
        wheeled_model.log_model(
            artifact_path=wheeled_artifact_path,
            registered_model_name=model_name,
            model_uri=model_uri,
        )

    log_messages = []

    def custom_warn(message_text, *args, **kwargs):
        log_messages.append(message_text % args % kwargs)

    with mock.patch("mlflow.wheeled_model._logger.warning") as warn_mock:
        warn_mock.side_effect = custom_warn

        # Save wheeled model
        double_wheeled_model_path = os.path.join(tmp_path, "model")
        with mlflow.start_run():
            wheeled_model.save_model(path=double_wheeled_model_path, model_uri=wheeled_model_uri)

    assert (
        "This model already has packaged wheels. New wheels will not be packaged." in log_messages
    )


def test_log_model_with_non_model_uri(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )
        model_uri = model_info.model_uri

    # Re-log with wheels
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        wheeled_artifact_path = "model"
        with mlflow.start_run():
            wheeled_model.log_model(
                artifact_path=wheeled_artifact_path,
                registered_model_name=model_name,
                model_uri=model_uri,
            )


def test_save_model_with_non_model_uri(tmp_path, sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"

    # Log a model
    sklearn_artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model,
            artifact_path=sklearn_artifact_path,
            registered_model_name=model_name,
        )
        model_uri = model_info.model_uri

    # Save with wheels
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        wheeled_model_path = os.path.join(tmp_path, "model")
        with mlflow.start_run():
            wheeled_model.save_model(path=wheeled_model_path, model_uri=model_uri)


def test_save_model_with_extra_kwargs(tmp_path, sklearn_knn_model):
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

    # Add additional kwargs while saving with wheels
    extra_kwargs = {"extra_kwarg": None}
    match = "save_model() got unexpected keyword arguments: {}".format(extra_kwargs)
    with pytest.raises(TypeError, match=re.escape(match)):
        wheeled_model_path = os.path.join(tmp_path, "model")
        with mlflow.start_run():
            wheeled_model.save_model(
                path=wheeled_model_path, model_uri=model_uri, mlflow_model=Model(), extra_kwarg=None
            )

    with pytest.raises(TypeError, match=re.escape(match)):
        wheeled_model_path = os.path.join(tmp_path, "model")
        with mlflow.start_run():
            wheeled_model.save_model(path=wheeled_model_path, model_uri=model_uri, extra_kwarg=None)

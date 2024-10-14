import os
import random
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import sklearn.neighbors as knn
import yaml
from sklearn import datasets

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow.models.model import METADATA_FILES
from mlflow.models.utils import load_serving_example
from mlflow.models.wheeled_model import _ORIGINAL_REQ_FILE_NAME, _WHEELS_FOLDER_NAME, WheeledModel
from mlflow.pyfunc.model import MLMODEL_FILE_NAME, Model
from mlflow.store.artifact.utils.models import _improper_model_uri_msg
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _is_pip_deps,
    _mlflow_conda_env,
)

from tests.helper_functions import (
    _is_available_on_pypi,
    _mlflow_major_version_string,
    pyfunc_serve_and_score_model,
)

EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("scikit-learn", module="sklearn") else ["--env-manager", "local"]
)


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


def _get_list_from_file(path):
    with open(path) as file:
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
    differing_keys = {"run_id", "utc_time_created", "model_uuid", "artifact_path"}

    # Compare wheeled model configs with original model config (MLModel files)
    for key in original_model_config:
        if key not in differing_keys:
            assert wheeled_model_config[key] == original_model_config[key]
        else:
            assert wheeled_model_config[key] != original_model_config[key]

    # Wheeled model key should only exist in wheeled_model_config
    assert wheeled_model_config.get(_WHEELS_FOLDER_NAME, None)
    assert not original_model_config.get(_WHEELS_FOLDER_NAME, None)

    # Verify new artifact path
    assert wheeled_model_config["artifact_path"] == WheeledModel.get_wheel_artifact_path(
        original_model_config["artifact_path"]
    )

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


def test_model_log_load(tmp_path, sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"
    artifact_path = "model"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            artifact_path,
            registered_model_name=model_name,
        )
        model_path = _download_artifact_from_uri(model_uri, tmp_path)
        original_model_config = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME)).__dict__

    # Re-log with wheels
    with mlflow.start_run():
        WheeledModel.log_model(model_uri=model_uri)
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
    artifact_path = "model"
    model_download_path = os.path.join(tmp_path, "m")
    wheeled_model_path = os.path.join(tmp_path, "wm")

    os.mkdir(model_download_path)
    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            artifact_path,
            registered_model_name=model_name,
        )
        model_path = _download_artifact_from_uri(model_uri, model_download_path)
        original_model_config = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME)).__dict__

    # Save with wheels
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


def test_logging_and_saving_wheeled_model_throws(tmp_path, sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"
    artifact_path = "model"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            artifact_path,
            registered_model_name=model_name,
        )

    # Re-log with wheels
    with mlflow.start_run():
        WheeledModel.log_model(
            model_uri=model_uri,
        )

    match = "Model libraries are already added"

    # Log wheeled model
    with pytest.raises(MlflowException, match=re.escape(match)):
        with mlflow.start_run():
            WheeledModel.log_model(
                model_uri=wheeled_model_uri,
            )

    # Saved a wheeled model
    saved_model_path = os.path.join(tmp_path, "test")
    with pytest.raises(MlflowException, match=re.escape(match)):
        with mlflow.start_run():
            WheeledModel(wheeled_model_uri).save_model(saved_model_path)


def test_log_model_with_non_model_uri():
    model_uri = "runs:/beefe0b6b5bd4acf9938244cdc006b64/model"

    # Log with wheels
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        with mlflow.start_run():
            WheeledModel.log_model(
                model_uri=model_uri,
            )

    # Save with wheels
    with pytest.raises(MlflowException, match=_improper_model_uri_msg(model_uri)):
        with mlflow.start_run():
            WheeledModel(model_uri)


def test_create_pip_requirement(tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    conda_env_path = os.path.join(tmp_path, "conda.yaml")
    pip_reqs_path = os.path.join(tmp_path, "requirements.txt")

    wm = WheeledModel(model_uri)

    expected_pip_deps = [expected_mlflow_version, "cloudpickle==2.1.0", "psutil==5.8.0"]
    _mlflow_conda_env(
        path=conda_env_path, additional_pip_deps=expected_pip_deps, install_mlflow=False
    )
    wm._create_pip_requirement(conda_env_path, pip_reqs_path)
    with open(pip_reqs_path) as f:
        pip_reqs = [x.strip() for x in f.readlines()]
    assert expected_pip_deps.sort() == pip_reqs.sort()


def test_update_conda_env_only_updates_pip_deps(tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    conda_env_path = os.path.join(tmp_path, "conda.yaml")
    pip_deps = [expected_mlflow_version, "cloudpickle==2.1.0", "psutil==5.8.0"]
    new_pip_deps = ["wheels/mlflow", "wheels/cloudpickle", "wheels/psutil"]

    wm = WheeledModel(model_uri)
    additional_conda_deps = ["add_conda_deps"]
    additional_conda_channels = ["add_conda_channels"]

    _mlflow_conda_env(
        conda_env_path,
        additional_conda_deps,
        pip_deps,
        additional_conda_channels,
        install_mlflow=False,
    )
    with open(conda_env_path) as f:
        old_conda_yaml = yaml.safe_load(f)
    wm._update_conda_env(new_pip_deps, conda_env_path)
    with open(conda_env_path) as f:
        new_conda_yaml = yaml.safe_load(f)
    assert old_conda_yaml.get("name") == new_conda_yaml.get("name")
    assert old_conda_yaml.get("channels") == new_conda_yaml.get("channels")
    for old_item, new_item in zip(
        old_conda_yaml.get("dependencies"), new_conda_yaml.get("dependencies")
    ):
        if isinstance(old_item, str):
            assert old_item == new_item
        if isinstance(old_item, dict):
            assert old_item.get("pip") == pip_deps
        if isinstance(new_item, dict):
            assert new_item.get("pip") == new_pip_deps


def test_serving_wheeled_model(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"
    artifact_path = "model"
    (model, inference_data) = sklearn_knn_model

    # Log a model
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path,
            registered_model_name=model_name,
            input_example=pd.DataFrame(inference_data),
        )

    # Re-log with wheels
    with mlflow.start_run():
        WheeledModel.log_model(model_uri=model_uri)

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        wheeled_model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records").values.squeeze()
    np.testing.assert_array_almost_equal(scores, model.predict(inference_data))


def test_wheel_download_works(tmp_path):
    simple_dependency = "cloudpickle"
    requirements_file = os.path.join(tmp_path, "req.txt")
    wheel_dir = os.path.join(tmp_path, "wheels")
    with open(requirements_file, "w") as req_file:
        req_file.write(simple_dependency)

    WheeledModel._download_wheels(requirements_file, wheel_dir)
    wheels = os.listdir(wheel_dir)
    assert len(wheels) == 1  # Only a single wheel is downloaded
    assert wheels[0].endswith(".whl")  # Type is wheel
    assert simple_dependency in wheels[0]  # Cloudpickle wheel downloaded


def test_wheel_download_override_option_works(tmp_path):
    dependency = "pyspark"
    requirements_file = os.path.join(tmp_path, "req.txt")
    wheel_dir = os.path.join(tmp_path, "wheels")
    with open(requirements_file, "w") as req_file:
        req_file.write(dependency)

    # Default option fails to download wheel
    with pytest.raises(
        MlflowException, match="An error occurred while downloading the dependency wheels"
    ):
        WheeledModel._download_wheels(requirements_file, wheel_dir)

    # Set option override
    os.environ["MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS"] = "--prefer-binary"
    WheeledModel._download_wheels(requirements_file, wheel_dir)
    assert len(os.listdir(wheel_dir))  # Wheel dir is not empty


def test_wheel_download_dependency_conflicts(tmp_path):
    reqs_file = tmp_path / "requirements.txt"
    reqs_file.write_text("mlflow==2.15.0\nmlflow==2.16.0")
    with pytest.raises(
        MlflowException,
        # Ensure the error message contains conflict details
        match=r"Cannot install mlflow==2\.15\.0 and mlflow==2\.16\.0.+The conflict is caused by",
    ):
        WheeledModel._download_wheels(reqs_file, tmp_path / "wheels")


def test_copy_metadata(mock_is_in_databricks, sklearn_knn_model):
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            "model",
            registered_model_name="sklearn_knn_model",
        )

    with mlflow.start_run():
        model_info = WheeledModel.log_model(model_uri="models:/sklearn_knn_model/1")

    artifact_path = mlflow.artifacts.download_artifacts(model_info.model_uri)
    metadata_path = os.path.join(artifact_path, "metadata")
    if mock_is_in_databricks.return_value:
        assert set(os.listdir(metadata_path)) == set(METADATA_FILES + [_ORIGINAL_REQ_FILE_NAME])
    else:
        assert not os.path.exists(metadata_path)
    assert mock_is_in_databricks.call_count == 2

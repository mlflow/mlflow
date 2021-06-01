from collections import namedtuple
from unittest import mock
import os
import pytest
import yaml

import catboost as cb
import numpy as np
import pandas as pd
import sklearn.datasets as datasets

import mlflow.catboost
from mlflow import pyfunc
from mlflow.models.utils import _read_example
from mlflow.models import Model, infer_signature
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import _compare_conda_env_requirements

ModelWithData = namedtuple("ModelWithData", ["model", "inference_dataframe"])


def get_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target)
    return X, y


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_PARAMS = {"allow_writing_files": False, "iterations": 10}


@pytest.fixture(
    scope="module",
    params=[
        cb.CatBoost(MODEL_PARAMS),
        cb.CatBoostClassifier(**MODEL_PARAMS),
        cb.CatBoostRegressor(**MODEL_PARAMS),
    ],
    ids=["CatBoost", "CatBoostClassifier", "CatBoostRegressor"],
)
def cb_model(request):
    model = request.param
    X, y = get_iris()
    return ModelWithData(model=model.fit(X, y), inference_dataframe=X)


@pytest.fixture
def reg_model():
    model = cb.CatBoostRegressor(**MODEL_PARAMS)
    X, y = get_iris()
    return ModelWithData(model=model.fit(X, y), inference_dataframe=X)


@pytest.fixture
def model_path(tmpdir):
    return tmpdir.join("model").strpath


@pytest.fixture
def custom_env(tmpdir):
    conda_env_path = os.path.join(tmpdir.strpath, "conda_env.yml")
    _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost", "pytest"])
    return conda_env_path


@pytest.mark.large
@pytest.mark.parametrize("model_type", ["CatBoost", "CatBoostClassifier", "CatBoostRegressor"])
def test_init_model(model_type):
    model = mlflow.catboost._init_model(model_type)
    assert model.__class__.__name__ == model_type


@pytest.mark.large
def test_init_model_throws_for_invalid_model_type():
    with pytest.raises(TypeError, match="Invalid model type"):
        mlflow.catboost._init_model("unsupported")


@pytest.mark.large
def test_model_save_load(cb_model, model_path):
    model, inference_dataframe = cb_model
    mlflow.catboost.save_model(cb_model=model, path=model_path)

    loaded_model = mlflow.catboost.load_model(model_uri=model_path)
    np.testing.assert_array_almost_equal(
        model.predict(inference_dataframe), loaded_model.predict(inference_dataframe),
    )

    loaded_pyfunc = pyfunc.load_pyfunc(model_uri=model_path)
    np.testing.assert_array_almost_equal(
        loaded_model.predict(inference_dataframe), loaded_pyfunc.predict(inference_dataframe),
    )


@pytest.mark.large
def test_log_model_logs_model_type(cb_model):
    with mlflow.start_run():
        artifact_path = "model"
        mlflow.catboost.log_model(cb_model.model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    flavor_conf = Model.load(model_uri).flavors["catboost"]
    assert "model_type" in flavor_conf
    assert flavor_conf["model_type"] == cb_model.model.__class__.__name__


# Supported serialization formats:
# https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html
SUPPORTS_DESERIALIZATION = ["cbm", "coreml", "json", "onnx"]
save_formats = SUPPORTS_DESERIALIZATION + ["python", "cpp", "pmml"]


@pytest.mark.large
@pytest.mark.parametrize("save_format", save_formats)
def test_log_model_logs_save_format(reg_model, save_format):
    with mlflow.start_run():
        artifact_path = "model"
        mlflow.catboost.log_model(reg_model.model, artifact_path, format=save_format)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    flavor_conf = Model.load(model_uri).flavors["catboost"]
    assert "save_format" in flavor_conf
    assert flavor_conf["save_format"] == save_format

    if save_format in SUPPORTS_DESERIALIZATION:
        mlflow.catboost.load_model(model_uri)
    else:
        with pytest.raises(cb.CatBoostError, match="deserialization not supported or missing"):
            mlflow.catboost.load_model(model_uri)


@pytest.mark.large
@pytest.mark.parametrize("signature", [None, infer_signature(get_iris()[0])])
@pytest.mark.parametrize("input_example", [None, get_iris()[0].head(3)])
def test_signature_and_examples_are_saved_correctly(
    reg_model, model_path, signature, input_example
):
    mlflow.catboost.save_model(
        reg_model.model, model_path, signature=signature, input_example=input_example
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if input_example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        pd.testing.assert_frame_equal(_read_example(mlflow_model, model_path), input_example)


@pytest.mark.large
def test_model_load_from_remote_uri_succeeds(reg_model, model_path, mock_s3_bucket):
    model, inference_dataframe = reg_model
    mlflow.catboost.save_model(cb_model=model, path=model_path)
    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_path = "model"
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    loaded_model = mlflow.catboost.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        model.predict(inference_dataframe), loaded_model.predict(inference_dataframe),
    )


@pytest.mark.large
def test_log_model(cb_model, tmpdir):
    model, inference_dataframe = cb_model
    with mlflow.start_run():
        artifact_path = "model"
        conda_env = os.path.join(tmpdir.strpath, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["catboost"])

        mlflow.catboost.log_model(model, artifact_path, conda_env=conda_env)
        model_uri = "runs:/{}/{}".format(mlflow.active_run().info.run_id, artifact_path)

        loaded_model = mlflow.catboost.load_model(model_uri)
        np.testing.assert_array_almost_equal(
            model.predict(inference_dataframe), loaded_model.predict(inference_dataframe),
        )

        local_path = _download_artifact_from_uri(model_uri)
        model_config = Model.load(os.path.join(local_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
        assert os.path.exists(os.path.join(local_path, env_path))


def test_log_model_calls_register_model(cb_model, tmpdir):
    artifact_path = "model"
    registered_model_name = "registered_model"
    with mlflow.start_run() as run, mock.patch("mlflow.register_model") as register_model_mock:
        conda_env_path = os.path.join(tmpdir.strpath, "conda_env.yaml")
        _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost"])
        mlflow.catboost.log_model(
            cb_model.model,
            artifact_path,
            conda_env=conda_env_path,
            registered_model_name=registered_model_name,
        )
        model_uri = "runs:/{}/{}".format(run.info.run_id, artifact_path)
        register_model_mock.assert_called_once_with(
            model_uri, registered_model_name, await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(cb_model, tmpdir):
    with mlflow.start_run(), mock.patch("mlflow.register_model") as register_model_mock:
        artifact_path = "model"
        conda_env_path = os.path.join(tmpdir.strpath, "conda_env.yaml")
        _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost"])
        mlflow.catboost.log_model(cb_model.model, artifact_path, conda_env=conda_env_path)
        register_model_mock.assert_not_called()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    reg_model, model_path, custom_env
):
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=custom_env)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != custom_env
    assert read_yaml(saved_conda_env_path) == read_yaml(custom_env)


@pytest.mark.large
def test_model_save_persists_requirements_in_mlflow_model_directory(
    reg_model, model_path, custom_env
):
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(custom_env, saved_pip_req_path)


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(reg_model, model_path):
    conda_env = mlflow.catboost.get_default_conda_env()
    conda_env["dependencies"].append("pytest")
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert read_yaml(saved_conda_env_path) == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(reg_model, custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.catboost.log_model(reg_model.model, artifact_path, conda_env=custom_env)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=local_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(local_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != custom_env
    assert read_yaml(saved_conda_env_path) == read_yaml(custom_env)


@pytest.mark.large
def test_model_log_persists_requirements_in_mlflow_model_directory(reg_model, custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.catboost.log_model(reg_model.model, artifact_path, conda_env=custom_env)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    saved_pip_req_path = os.path.join(local_path, "requirements.txt")
    _compare_conda_env_requirements(custom_env, saved_pip_req_path)


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    reg_model, model_path
):
    mlflow.catboost.save_model(reg_model.model, model_path, conda_env=None)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert read_yaml(conda_env_path) == mlflow.catboost.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    reg_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.catboost.log_model(reg_model.model, artifact_path, conda_env=None)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=local_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(local_path, pyfunc_conf[pyfunc.ENV])
    assert read_yaml(conda_env_path) == mlflow.catboost.get_default_conda_env()

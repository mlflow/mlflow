import json
import os
from pathlib import Path
from typing import Any, NamedTuple
from unittest import mock

import catboost as cb
import numpy as np
import pandas as pd
import pytest
import yaml
from packaging.version import Version
from sklearn import datasets
from sklearn.pipeline import Pipeline

import mlflow.catboost
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _compare_logged_code_paths,
    _is_available_on_pypi,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    pyfunc_serve_and_score_model,
)

EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("catboost") else ["--env-manager", "local"]
)


class ModelWithData(NamedTuple):
    model: Any
    inference_dataframe: Any


def get_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target)
    return X, y


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


MODEL_PARAMS = {"allow_writing_files": False, "iterations": 10}


def iter_models():
    X, y = get_iris()
    model = cb.CatBoost(MODEL_PARAMS).fit(X, y)
    yield ModelWithData(model=model, inference_dataframe=X)

    model = cb.CatBoostClassifier(**MODEL_PARAMS).fit(X, y)
    yield ModelWithData(model=model, inference_dataframe=X)

    model = cb.CatBoostRegressor(**MODEL_PARAMS).fit(X, y)
    yield ModelWithData(model=model, inference_dataframe=X)


@pytest.fixture(
    scope="module",
    params=iter_models(),
    ids=["CatBoost", "CatBoostClassifier", "CatBoostRegressor"],
)
def cb_model(request):
    return request.param


@pytest.fixture
def reg_model():
    model = cb.CatBoostRegressor(**MODEL_PARAMS)
    X, y = get_iris()
    return ModelWithData(model=model.fit(X, y), inference_dataframe=X)


def get_reg_model_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="sepal length (cm)", type=DataType.double),
                ColSpec(name="sepal width (cm)", type=DataType.double),
            ]
        ),
        outputs=Schema([ColSpec(type=DataType.double)]),
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def custom_env(tmp_path):
    conda_env_path = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost", "pytest"])
    return conda_env_path


@pytest.mark.parametrize("model_type", ["CatBoost", "CatBoostClassifier", "CatBoostRegressor"])
def test_init_model(model_type):
    model = mlflow.catboost._init_model(model_type)
    assert model.__class__.__name__ == model_type


@pytest.mark.skipif(
    Version(cb.__version__) < Version("0.26.0"),
    reason="catboost < 0.26.0 does not support CatBoostRanker",
)
def test_log_catboost_ranker():
    """
    This is a separate test for the CatBoostRanker model.
    It is separate since the ranking task requires a group_id column which makes the code different.
    """
    # the ranking task requires setting a group_id
    # we are creating a dummy group_id here that doesn't make any sense for the Iris dataset,
    # but is ok for testing if the code is running correctly
    X, y = get_iris()
    dummy_group_id = np.arange(len(X)) % 3
    dummy_group_id.sort()

    model = cb.CatBoostRanker(**MODEL_PARAMS, subsample=1.0)
    model.fit(X, y, group_id=dummy_group_id)

    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(model, name="model")
        loaded_model = mlflow.catboost.load_model(model_info.model_uri)
        assert isinstance(loaded_model, cb.CatBoostRanker)
        np.testing.assert_array_almost_equal(model.predict(X), loaded_model.predict(X))


def test_init_model_throws_for_invalid_model_type():
    with pytest.raises(TypeError, match="Invalid model type"):
        mlflow.catboost._init_model("unsupported")


def test_model_save_load(cb_model, model_path):
    model, inference_dataframe = cb_model
    mlflow.catboost.save_model(cb_model=model, path=model_path)

    loaded_model = mlflow.catboost.load_model(model_uri=model_path)
    np.testing.assert_array_almost_equal(
        model.predict(inference_dataframe),
        loaded_model.predict(inference_dataframe),
    )

    loaded_pyfunc = pyfunc.load_model(model_uri=model_path)
    np.testing.assert_array_almost_equal(
        loaded_model.predict(inference_dataframe),
        loaded_pyfunc.predict(inference_dataframe),
    )


def test_log_model_logs_model_type(cb_model):
    with mlflow.start_run():
        artifact_path = "model"
        model_info = mlflow.catboost.log_model(cb_model.model, name=artifact_path)

    flavor_conf = Model.load(model_info.model_uri).flavors["catboost"]
    assert "model_type" in flavor_conf
    assert flavor_conf["model_type"] == cb_model.model.__class__.__name__


# Supported serialization formats:
# https://catboost.ai/docs/concepts/python-reference_catboost_save_model.html
SUPPORTS_DESERIALIZATION = ["cbm", "coreml", "json", "onnx"]
save_formats = SUPPORTS_DESERIALIZATION + ["python", "cpp", "pmml"]


@pytest.mark.allow_infer_pip_requirements_fallback
@pytest.mark.parametrize("save_format", save_formats)
def test_log_model_logs_save_format(reg_model, save_format):
    with mlflow.start_run():
        artifact_path = "model"
        model_info = mlflow.catboost.log_model(
            reg_model.model, name=artifact_path, format=save_format
        )

    flavor_conf = Model.load(model_info.model_uri).flavors["catboost"]
    assert "save_format" in flavor_conf
    assert flavor_conf["save_format"] == save_format

    if save_format in SUPPORTS_DESERIALIZATION:
        mlflow.catboost.load_model(model_info.model_uri)
    else:
        with pytest.raises(cb.CatBoostError, match="deserialization not supported or missing"):
            mlflow.catboost.load_model(model_info.model_uri)


@pytest.mark.parametrize("signature", [None, get_reg_model_signature()])
@pytest.mark.parametrize("input_example", [None, get_iris()[0].head(3)])
def test_signature_and_examples_are_saved_correctly(
    reg_model, model_path, signature, input_example
):
    mlflow.catboost.save_model(
        reg_model.model, model_path, signature=signature, input_example=input_example
    )
    mlflow_model = Model.load(model_path)
    if signature is None and input_example is None:
        assert mlflow_model.signature is None
    else:
        assert mlflow_model.signature == get_reg_model_signature()
    if input_example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        pd.testing.assert_frame_equal(_read_example(mlflow_model, model_path), input_example)


def test_model_load_from_remote_uri_succeeds(reg_model, model_path, mock_s3_bucket):
    model, inference_dataframe = reg_model
    mlflow.catboost.save_model(cb_model=model, path=model_path)
    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_path = "model"
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    loaded_model = mlflow.catboost.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        model.predict(inference_dataframe),
        loaded_model.predict(inference_dataframe),
    )


def test_log_model(cb_model, tmp_path):
    model, inference_dataframe = cb_model
    with mlflow.start_run():
        artifact_path = "model"
        conda_env = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["catboost"])

        model_info = mlflow.catboost.log_model(model, name=artifact_path, conda_env=conda_env)

        loaded_model = mlflow.catboost.load_model(model_info.model_uri)
        np.testing.assert_array_almost_equal(
            model.predict(inference_dataframe),
            loaded_model.predict(inference_dataframe),
        )

        local_path = _download_artifact_from_uri(model_info.model_uri)
        model_config = Model.load(os.path.join(local_path, "MLmodel"))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert os.path.exists(os.path.join(local_path, env_path))


def test_log_model_calls_register_model(cb_model, tmp_path):
    artifact_path = "model"
    registered_model_name = "registered_model"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.tracking._model_registry.fluent._register_model"),
    ):
        conda_env_path = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost"])
        model_info = mlflow.catboost.log_model(
            cb_model.model,
            name=artifact_path,
            conda_env=conda_env_path,
            registered_model_name=registered_model_name,
        )
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_info.model_uri,
            registered_model_name=registered_model_name,
        )


def test_log_model_no_registered_model_name(cb_model, tmp_path):
    with mlflow.start_run(), mock.patch("mlflow.register_model") as register_model_mock:
        artifact_path = "model"
        conda_env_path = os.path.join(tmp_path, "conda_env.yaml")
        _mlflow_conda_env(conda_env_path, additional_pip_deps=["catboost"])
        mlflow.catboost.log_model(cb_model.model, name=artifact_path, conda_env=conda_env_path)
        register_model_mock.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    reg_model, model_path, custom_env
):
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=custom_env)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != custom_env
    assert read_yaml(saved_conda_env_path) == read_yaml(custom_env)


def test_model_save_persists_requirements_in_mlflow_model_directory(
    reg_model, model_path, custom_env
):
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(custom_env, saved_pip_req_path)


def test_model_save_accepts_conda_env_as_dict(reg_model, model_path):
    conda_env = mlflow.catboost.get_default_conda_env()
    conda_env["dependencies"].append("pytest")
    mlflow.catboost.save_model(cb_model=reg_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert read_yaml(saved_conda_env_path) == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(reg_model, custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name=artifact_path, conda_env=custom_env
        )

    local_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=local_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(local_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != custom_env
    assert read_yaml(saved_conda_env_path) == read_yaml(custom_env)


def test_model_log_persists_requirements_in_mlflow_model_directory(reg_model, custom_env):
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(reg_model.model, name="model", conda_env=custom_env)

    local_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    saved_pip_req_path = os.path.join(local_path, "requirements.txt")
    _compare_conda_env_requirements(custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(reg_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", pip_requirements=str(req_file)
        )
        _assert_pip_requirements(model_info.model_uri, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(reg_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.catboost.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            reg_model.model, name="model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    reg_model, model_path
):
    mlflow.catboost.save_model(reg_model.model, model_path)
    _assert_pip_requirements(model_path, mlflow.catboost.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    reg_model,
):
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(reg_model.model, name="model")

    _assert_pip_requirements(model_info.model_uri, mlflow.catboost.get_default_pip_requirements())


def test_pyfunc_serve_and_score(reg_model):
    model, inference_dataframe = reg_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            model, name=artifact_path, input_example=inference_dataframe
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.DataFrame(
        data=json.loads(resp.content.decode("utf-8"))["predictions"]
    ).values.squeeze()
    np.testing.assert_array_almost_equal(scores, model.predict(inference_dataframe))


def test_pyfunc_serve_and_score_sklearn(reg_model):
    model, inference_dataframe = reg_model
    model = Pipeline([("model", reg_model.model)])

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            model, name="model", input_example=inference_dataframe.head(3)
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        inference_payload,
        pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.DataFrame(
        data=json.loads(resp.content.decode("utf-8"))["predictions"]
    ).values.squeeze()
    np.testing.assert_array_almost_equal(scores, model.predict(inference_dataframe.head(3)))


def test_log_model_with_code_paths(cb_model):
    artifact_path = "model"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.catboost._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.catboost.log_model(
            cb_model.model, name=artifact_path, code_paths=[__file__]
        )
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.catboost.FLAVOR_NAME)
        mlflow.catboost.load_model(model_uri=model_info.model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(cb_model, model_path):
    mlflow.catboost.save_model(cb_model.model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def test_model_save_load_with_metadata(cb_model, model_path):
    mlflow.catboost.save_model(
        cb_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(cb_model):
    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            cb_model.model, name="model", metadata={"metadata_key": "metadata_value"}
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(cb_model):
    artifact_path = "model"
    example = cb_model.inference_dataframe.head(3)

    with mlflow.start_run():
        model_info = mlflow.catboost.log_model(
            cb_model.model, name=artifact_path, input_example=example
        )

    loaded_model_info = Model.load(model_info.model_uri)
    assert loaded_model_info.signature.inputs == Schema(
        [
            ColSpec(name="sepal length (cm)", type=DataType.double),
            ColSpec(name="sepal width (cm)", type=DataType.double),
        ]
    )
    assert loaded_model_info.signature.outputs in [
        # when the model output is a 1D numpy array, it is cast into a `ColSpec`
        Schema([ColSpec(type=DataType.double)]),
        # when the model output is a higher dimensional numpy array, it remains a `TensorSpec`
        Schema([TensorSpec(np.dtype("int64"), (-1, 1))]),
    ]

# pep8: disable=E501

import json
import os
from typing import Any, NamedTuple
from unittest import mock

import h2o
import numpy as np
import pandas as pd
import pytest
import yaml
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn import datasets

import mlflow
import mlflow.h2o
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_conda_env_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
    pyfunc_serve_and_score_model,
)


class ModelWithData(NamedTuple):
    model: Any
    inference_data: Any


@pytest.fixture
def h2o_iris_model():
    h2o.init()
    iris = datasets.load_iris()
    data = h2o.H2OFrame(
        {
            "feature1": list(iris.data[:, 0]),
            "feature2": list(iris.data[:, 1]),
            "target": ([f"Flower {i}" for i in iris.target]),
        }
    )
    train, test = data.split_frame(ratios=[0.7])

    h2o_gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=6)
    h2o_gbm.train(["feature1", "feature2"], "target", training_frame=train)
    return ModelWithData(model=h2o_gbm, inference_data=test)


@pytest.fixture(scope="module")
def h2o_iris_model_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="feature1", type=DataType.double),
                ColSpec(name="feature2", type=DataType.double),
                ColSpec(name="target", type=DataType.string),
            ]
        ),
        outputs=Schema(
            [
                ColSpec(name="predict", type=DataType.string),
                ColSpec(name="Flower 0", type=DataType.double),
                ColSpec(name="Flower 1", type=DataType.double),
                ColSpec(name="Flower 2", type=DataType.double),
            ]
        ),
    )


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def h2o_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["h2o", "pytest"])
    return conda_env


def test_model_save_load(h2o_iris_model, model_path):
    h2o_model = h2o_iris_model.model
    mlflow.h2o.save_model(h2o_model=h2o_model, path=model_path)

    # Loading h2o model
    h2o_model_loaded = mlflow.h2o.load_model(model_path)
    assert all(
        h2o_model_loaded.predict(h2o_iris_model.inference_data).as_data_frame()
        == h2o_model.predict(h2o_iris_model.inference_data).as_data_frame()
    )

    # Loading pyfunc model
    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    assert all(
        pyfunc_loaded.predict(h2o_iris_model.inference_data.as_data_frame())
        == h2o_model.predict(h2o_iris_model.inference_data).as_data_frame()
    )


def test_signature_and_examples_are_saved_correctly(h2o_iris_model, h2o_iris_model_signature):
    model = h2o_iris_model.model
    example_ = h2o_iris_model.inference_data.as_data_frame().head(3)
    for signature in (None, h2o_iris_model_signature):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.h2o.save_model(model, path=path, signature=signature, input_example=example)
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == h2o_iris_model_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_log(h2o_iris_model):
    h2o_model = h2o_iris_model.model
    try:
        artifact_path = "gbm_model"
        model_info = mlflow.h2o.log_model(h2o_model, name=artifact_path)
        # Load model
        h2o_model_loaded = mlflow.h2o.load_model(model_uri=model_info.model_uri)
        assert all(
            h2o_model_loaded.predict(h2o_iris_model.inference_data).as_data_frame()
            == h2o_model.predict(h2o_iris_model.inference_data).as_data_frame()
        )
    finally:
        mlflow.end_run()


def test_model_load_succeeds_with_missing_data_key_when_data_exists_at_default_path(
    h2o_iris_model, model_path
):
    """
    This is a backwards compatibility test to ensure that models saved in MLflow version <= 0.7.0
    can be loaded successfully. These models are missing the `data` flavor configuration key.
    """
    h2o_model = h2o_iris_model.model
    mlflow.h2o.save_model(h2o_model=h2o_model, path=model_path)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    flavor_conf = model_conf.flavors.get(mlflow.h2o.FLAVOR_NAME, None)
    assert flavor_conf is not None
    del flavor_conf["data"]
    model_conf.save(model_conf_path)

    h2o_model_loaded = mlflow.h2o.load_model(model_path)
    assert all(
        h2o_model_loaded.predict(h2o_iris_model.inference_data).as_data_frame()
        == h2o_model.predict(h2o_iris_model.inference_data).as_data_frame()
    )


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    h2o_iris_model, model_path, h2o_custom_env
):
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path, conda_env=h2o_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != h2o_custom_env

    with open(h2o_custom_env) as f:
        h2o_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == h2o_custom_env_text


def test_model_save_persists_requirements_in_mlflow_model_directory(
    h2o_iris_model, model_path, h2o_custom_env
):
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path, conda_env=h2o_custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(h2o_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(h2o_iris_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model", pip_requirements=str(req_file)
        )
        _assert_pip_requirements(model_info.model_uri, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model,
            name="model",
            pip_requirements=[f"-r {req_file}", "b"],
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(h2o_iris_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.h2o.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(h2o_iris_model, model_path):
    conda_env = dict(mlflow.h2o.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    h2o_iris_model, h2o_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name=artifact_path, conda_env=h2o_custom_env
        )

    model_path = _download_artifact_from_uri(model_info.model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != h2o_custom_env

    with open(h2o_custom_env) as f:
        h2o_custom_env_text = f.read()
    with open(saved_conda_env_path) as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == h2o_custom_env_text


def test_model_log_persists_requirements_in_mlflow_model_directory(h2o_iris_model, h2o_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name=artifact_path, conda_env=h2o_custom_env
        )
        model_path = _download_artifact_from_uri(model_info.model_uri)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(h2o_custom_env, saved_pip_req_path)


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    h2o_iris_model, model_path
):
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.h2o.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    h2o_iris_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(h2o_iris_model.model, name=artifact_path)
    _assert_pip_requirements(model_info.model_uri, mlflow.h2o.get_default_pip_requirements())


def test_pyfunc_serve_and_score(h2o_iris_model):
    model, inference_dataframe = h2o_iris_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            model, name=artifact_path, input_example=inference_dataframe.as_data_frame()
        )

    inference_payload = load_serving_example(model_info.model_uri)
    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=inference_payload,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    decoded_json = json.loads(resp.content.decode("utf-8"))
    scores = pd.DataFrame(data=decoded_json["predictions"]).drop("predict", axis=1)
    preds = model.predict(inference_dataframe).as_data_frame().drop("predict", axis=1)
    np.testing.assert_array_almost_equal(scores, preds)


def test_log_model_with_code_paths(h2o_iris_model):
    with (
        mlflow.start_run(),
        mock.patch("mlflow.h2o._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name="model_uri", code_paths=[__file__]
        )
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.h2o.FLAVOR_NAME)
        mlflow.h2o.load_model(model_info.model_uri)
        add_mock.assert_called()


def test_model_save_load_with_metadata(h2o_iris_model, model_path):
    mlflow.h2o.save_model(
        h2o_iris_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(h2o_iris_model):
    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model,
            name="model",
            metadata={"metadata_key": "metadata_value"},
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(h2o_iris_model, h2o_iris_model_signature):
    artifact_path = "model"
    example = h2o_iris_model.inference_data.as_data_frame().head(3)

    with mlflow.start_run():
        model_info = mlflow.h2o.log_model(
            h2o_iris_model.model, name=artifact_path, input_example=example
        )

    mlflow_model = Model.load(model_info.model_uri)
    assert mlflow_model.signature == h2o_iris_model_signature

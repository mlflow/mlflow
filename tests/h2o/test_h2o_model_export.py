# pep8: disable=E501

import os
import pytest
import yaml
import pandas as pd
from collections import namedtuple
from unittest import mock

import numpy as np
from sklearn import datasets
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import mlflow.h2o
import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    pyfunc_serve_and_score_model,
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _compare_logged_code_paths,
)

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture
def h2o_iris_model():
    h2o.init()
    iris = datasets.load_iris()
    data = h2o.H2OFrame(
        {
            "feature1": list(iris.data[:, 0]),
            "feature2": list(iris.data[:, 1]),
            "target": list(map(lambda i: "Flower %d" % i, iris.target)),
        }
    )
    train, test = data.split_frame(ratios=[0.7])  # pylint: disable=unbalanced-tuple-unpacking

    h2o_gbm = H2OGradientBoostingEstimator(ntrees=10, max_depth=6)
    h2o_gbm.train(["feature1", "feature2"], "target", training_frame=train)
    return ModelWithData(model=h2o_gbm, inference_data=test)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def h2o_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
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


def test_signature_and_examples_are_saved_correctly(h2o_iris_model):
    model = h2o_iris_model.model
    signature_ = infer_signature(h2o_iris_model.inference_data.as_data_frame())
    example_ = h2o_iris_model.inference_data.as_data_frame().head(3)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.h2o.save_model(model, path=path, signature=signature, input_example=example)
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_log(h2o_iris_model):
    h2o_model = h2o_iris_model.model
    try:
        artifact_path = "gbm_model"
        model_info = mlflow.h2o.log_model(h2o_model=h2o_model, artifact_path=artifact_path)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        assert model_info.model_uri == model_uri
        # Load model
        h2o_model_loaded = mlflow.h2o.load_model(model_uri=model_uri)
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
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != h2o_custom_env

    with open(h2o_custom_env, "r") as f:
        h2o_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == h2o_custom_env_text


def test_model_save_persists_requirements_in_mlflow_model_directory(
    h2o_iris_model, model_path, h2o_custom_env
):
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path, conda_env=h2o_custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(h2o_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(h2o_iris_model, tmpdir):
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.h2o.log_model(h2o_iris_model.model, "model", pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_iris_model.model,
            "model",
            pip_requirements=[f"-r {req_file.strpath}", "b"],
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_iris_model.model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(h2o_iris_model, tmpdir):
    default_reqs = mlflow.h2o.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.h2o.log_model(h2o_iris_model.model, "model", extra_pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_iris_model.model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_iris_model.model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(h2o_iris_model, model_path):
    conda_env = dict(mlflow.h2o.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.h2o.save_model(h2o_model=h2o_iris_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    h2o_iris_model, h2o_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_model=h2o_iris_model.model, artifact_path=artifact_path, conda_env=h2o_custom_env
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != h2o_custom_env

    with open(h2o_custom_env, "r") as f:
        h2o_custom_env_text = f.read()
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_text = f.read()
    assert saved_conda_env_text == h2o_custom_env_text


def test_model_log_persists_requirements_in_mlflow_model_directory(h2o_iris_model, h2o_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.h2o.log_model(
            h2o_model=h2o_iris_model.model, artifact_path=artifact_path, conda_env=h2o_custom_env
        )
        model_path = _download_artifact_from_uri(
            "runs:/{run_id}/{artifact_path}".format(
                run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
            )
        )

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
        mlflow.h2o.log_model(h2o_model=h2o_iris_model.model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.h2o.get_default_pip_requirements())


def test_pyfunc_serve_and_score(h2o_iris_model):
    model, inference_dataframe = h2o_iris_model
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.h2o.log_model(model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_dataframe.as_data_frame(),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records").drop("predict", axis=1)
    preds = model.predict(inference_dataframe).as_data_frame().drop("predict", axis=1)
    np.testing.assert_array_almost_equal(scores, preds)


def test_log_model_with_code_paths(h2o_iris_model):
    artifact_path = "model_uri"
    with mlflow.start_run(), mock.patch(
        "mlflow.h2o._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.h2o.log_model(h2o_iris_model.model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.h2o.FLAVOR_NAME)
        mlflow.h2o.load_model(model_uri)
        add_mock.assert_called()

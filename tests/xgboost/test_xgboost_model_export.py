import json
import os
from pathlib import Path
from typing import Any, NamedTuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml
from sklearn import datasets
from sklearn.pipeline import Pipeline

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.utils
import mlflow.xgboost
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature, infer_signature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.proto_json_utils import dataframe_from_parsed_json
from mlflow.xgboost import _exclude_unrecognized_kwargs

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
    [] if _is_available_on_pypi("xgboost") else ["--env-manager", "local"]
)


class ModelWithData(NamedTuple):
    model: Any
    inference_dataframe: pd.DataFrame
    inference_dmatrix: xgb.DMatrix


@pytest.fixture(scope="module")
def xgb_model():
    iris = datasets.load_iris()
    X = pd.DataFrame(
        iris.data[:, :2],
        columns=iris.feature_names[:2],  # we only take the first two features.
    )
    y = iris.target
    dtrain = xgb.DMatrix(X, y)
    model = xgb.train({"objective": "multi:softprob", "num_class": 3}, dtrain)
    return ModelWithData(model=model, inference_dataframe=X, inference_dmatrix=dtrain)


@pytest.fixture(scope="module")
def xgb_model_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="sepal length (cm)", type=DataType.double),
                ColSpec(name="sepal width (cm)", type=DataType.double),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 3))]),
    )


@pytest.fixture(scope="module")
def xgb_sklearn_model():
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)
    regressor = xgb.XGBRegressor(n_estimators=10)
    regressor.fit(X, y)
    return ModelWithData(model=regressor, inference_dataframe=X, inference_dmatrix=None)


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def xgb_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["xgboost", "pytest"])
    return conda_env


def test_model_save_load(xgb_model, model_path):
    model = xgb_model.model

    mlflow.xgboost.save_model(xgb_model=model, path=model_path)
    reloaded_model = mlflow.xgboost.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        model.predict(xgb_model.inference_dmatrix),
        reloaded_model.predict(xgb_model.inference_dmatrix),
    )

    np.testing.assert_array_almost_equal(
        reloaded_model.predict(xgb_model.inference_dmatrix),
        reloaded_pyfunc.predict(xgb_model.inference_dataframe),
    )


def test_sklearn_model_save_load(xgb_sklearn_model, model_path):
    model = xgb_sklearn_model.model
    mlflow.xgboost.save_model(xgb_model=model, path=model_path)
    reloaded_model = mlflow.xgboost.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        model.predict(xgb_sklearn_model.inference_dataframe),
        reloaded_model.predict(xgb_sklearn_model.inference_dataframe),
    )

    np.testing.assert_array_almost_equal(
        reloaded_model.predict(xgb_sklearn_model.inference_dataframe),
        reloaded_pyfunc.predict(xgb_sklearn_model.inference_dataframe),
    )


def test_signature_and_examples_are_saved_correctly(xgb_model, xgb_model_signature):
    model = xgb_model.model
    for signature in (None, xgb_model_signature):
        for example in (None, xgb_model.inference_dataframe.head(3)):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.xgboost.save_model(
                    xgb_model=model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == xgb_model_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_load_from_remote_uri_succeeds(xgb_model, model_path, mock_s3_bucket):
    mlflow.xgboost.save_model(xgb_model=xgb_model.model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.xgboost.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        xgb_model.model.predict(xgb_model.inference_dmatrix),
        reloaded_model.predict(xgb_model.inference_dmatrix),
    )


def test_model_log(xgb_model, model_path):
    model = xgb_model.model
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "model"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["xgboost"])

                model_info = mlflow.xgboost.log_model(
                    model, name=artifact_path, conda_env=conda_env
                )
                reloaded_model = mlflow.xgboost.load_model(model_uri=model_info.model_uri)
                np.testing.assert_array_almost_equal(
                    model.predict(xgb_model.inference_dmatrix),
                    reloaded_model.predict(xgb_model.inference_dmatrix),
                )

                model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()


def test_log_model_calls_register_model(xgb_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["xgboost"])
        model_info = mlflow.xgboost.log_model(
            xgb_model.model,
            name=artifact_path,
            conda_env=conda_env,
            registered_model_name="AdsModel1",
        )
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_info.model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_no_registered_model_name(xgb_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["xgboost"])
        mlflow.xgboost.log_model(xgb_model.model, name=artifact_path, conda_env=conda_env)
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    xgb_model, model_path, xgb_custom_env
):
    mlflow.xgboost.save_model(xgb_model=xgb_model.model, path=model_path, conda_env=xgb_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != xgb_custom_env

    with open(xgb_custom_env) as f:
        xgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == xgb_custom_env_parsed


def test_model_save_persists_requirements_in_mlflow_model_directory(
    xgb_model, model_path, xgb_custom_env
):
    mlflow.xgboost.save_model(xgb_model=xgb_model.model, path=model_path, conda_env=xgb_custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(xgb_custom_env, saved_pip_req_path)


def test_save_model_with_pip_requirements(xgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    tmpdir1 = tmp_path.joinpath("1")
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    mlflow.xgboost.save_model(xgb_model.model, tmpdir1, pip_requirements=str(req_file))
    _assert_pip_requirements(tmpdir1, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    tmpdir2 = tmp_path.joinpath("2")
    mlflow.xgboost.save_model(xgb_model.model, tmpdir2, pip_requirements=[f"-r {req_file}", "b"])
    _assert_pip_requirements(tmpdir2, [expected_mlflow_version, "a", "b"], strict=True)

    # Constraints file
    tmpdir3 = tmp_path.joinpath("3")
    mlflow.xgboost.save_model(xgb_model.model, tmpdir3, pip_requirements=[f"-c {req_file}", "b"])
    _assert_pip_requirements(
        tmpdir3, [expected_mlflow_version, "b", "-c constraints.txt"], ["a"], strict=True
    )


def test_save_model_with_extra_pip_requirements(xgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.xgboost.get_default_pip_requirements()

    # Path to a requirements file
    tmpdir1 = tmp_path.joinpath("1")
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    mlflow.xgboost.save_model(xgb_model.model, tmpdir1, extra_pip_requirements=str(req_file))
    _assert_pip_requirements(tmpdir1, [expected_mlflow_version, *default_reqs, "a"])

    # List of requirements
    tmpdir2 = tmp_path.joinpath("2")
    mlflow.xgboost.save_model(
        xgb_model.model, tmpdir2, extra_pip_requirements=[f"-r {req_file}", "b"]
    )
    _assert_pip_requirements(tmpdir2, [expected_mlflow_version, *default_reqs, "a", "b"])

    # Constraints file
    tmpdir3 = tmp_path.joinpath("3")
    mlflow.xgboost.save_model(
        xgb_model.model, tmpdir3, extra_pip_requirements=[f"-c {req_file}", "b"]
    )
    _assert_pip_requirements(
        tmpdir3, [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"], ["a"]
    )


def test_log_model_with_pip_requirements(xgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", pip_requirements=str(req_file)
        )
        _assert_pip_requirements(model_info.model_uri, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(xgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.xgboost.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(xgb_model, model_path):
    conda_env = dict(mlflow.xgboost.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.xgboost.save_model(xgb_model=xgb_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    xgb_model, xgb_custom_env
):
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", conda_env=xgb_custom_env
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != xgb_custom_env

    with open(xgb_custom_env) as f:
        xgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == xgb_custom_env_parsed


def test_model_log_persists_requirements_in_mlflow_model_directory(xgb_model, xgb_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name=artifact_path, conda_env=xgb_custom_env
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(xgb_custom_env, saved_pip_req_path)


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    xgb_model, model_path
):
    mlflow.xgboost.save_model(xgb_model=xgb_model.model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.xgboost.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    xgb_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(xgb_model.model, name=artifact_path)

    _assert_pip_requirements(model_info.model_uri, mlflow.xgboost.get_default_pip_requirements())


def test_pyfunc_serve_and_score(xgb_model):
    model, inference_dataframe, inference_dmatrix = xgb_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
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
    np.testing.assert_array_almost_equal(scores, model.predict(inference_dmatrix))


def get_sklearn_models():
    model = xgb.XGBClassifier(objective="multi:softmax", n_estimators=10)
    pipe = Pipeline([("model", model)])
    return [model, pipe]


@pytest.mark.parametrize("model", get_sklearn_models())
def test_pyfunc_serve_and_score_sklearn(model):
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    model.fit(X, y)

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, name="model", input_example=X.head(3))

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
    np.testing.assert_array_equal(scores, model.predict(X.head(3)))


def test_load_pyfunc_succeeds_for_older_models_with_pyfunc_data_field(xgb_model, model_path):
    """
    This test verifies that xgboost models saved in older versions of MLflow are loaded
    successfully by ``mlflow.pyfunc.load_model``. These older models specify a pyfunc ``data``
    field referring directly to a XGBoost model file. Newer models also have the
    ``model_class`` in XGBoost flavor.
    """
    model = xgb_model.model
    mlflow.xgboost.save_model(xgb_model=model, path=model_path)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    xgboost_conf = model_conf.flavors.get(mlflow.xgboost.FLAVOR_NAME)
    assert xgboost_conf is not None
    assert "model_class" in xgboost_conf
    assert "data" in xgboost_conf
    assert pyfunc_conf is not None
    assert "model_class" not in pyfunc_conf
    assert pyfunc.DATA in pyfunc_conf

    # test old MLmodel conf
    model_conf.flavors["xgboost"] = {"xgb_version": xgb.__version__, "data": "model.xgb"}
    model_conf.save(model_conf_path)
    model_conf = Model.load(model_conf_path)
    xgboost_conf = model_conf.flavors.get(mlflow.xgboost.FLAVOR_NAME)
    assert "data" in xgboost_conf
    assert xgboost_conf["data"] == "model.xgb"

    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)
    assert isinstance(reloaded_pyfunc._model_impl.xgb_model, xgb.Booster)
    reloaded_xgb = mlflow.xgboost.load_model(model_uri=model_path)
    assert isinstance(reloaded_xgb, xgb.Booster)

    np.testing.assert_array_almost_equal(
        xgb_model.model.predict(xgb_model.inference_dmatrix),
        reloaded_pyfunc.predict(xgb_model.inference_dataframe),
    )

    np.testing.assert_array_almost_equal(
        reloaded_xgb.predict(xgb_model.inference_dmatrix),
        reloaded_pyfunc.predict(xgb_model.inference_dataframe),
    )


def test_log_model_with_code_paths(xgb_model):
    artifact_path = "model"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.xgboost._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name=artifact_path, code_paths=[__file__]
        )
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.xgboost.FLAVOR_NAME)
        mlflow.xgboost.load_model(model_uri=model_info.model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(xgb_model, model_path):
    mlflow.xgboost.save_model(xgb_model.model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


@pytest.mark.parametrize("model_format", ["xgb", "json", "ubj"])
def test_log_model_with_model_format(xgb_model, model_format):
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", model_format=model_format
        )
        loaded_model = mlflow.xgboost.load_model(model_info.model_uri)
        np.testing.assert_array_almost_equal(
            xgb_model.model.predict(xgb_model.inference_dmatrix),
            loaded_model.predict(xgb_model.inference_dmatrix),
        )


def test_model_save_load_with_metadata(xgb_model, model_path):
    mlflow.xgboost.save_model(
        xgb_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(xgb_model):
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model,
            name="model",
            metadata={"metadata_key": "metadata_value"},
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(xgb_model, xgb_model_signature):
    artifact_path = "model"
    X = xgb_model.inference_dataframe
    example = X.iloc[[0]]

    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name=artifact_path, input_example=example
        )

    mlflow_model = Model.load(model_info.model_uri)
    assert mlflow_model.signature == xgb_model_signature


def test_model_without_signature_predict(xgb_model):
    artifact_path = "model"
    X = xgb_model.inference_dataframe
    example = X.iloc[[0]]

    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(xgb_model.model, name=artifact_path)

    loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    data = pd.DataFrame(example).to_dict(orient="split")
    parsed_data = dataframe_from_parsed_json(data, pandas_orient="split")
    loaded_model.predict(parsed_data)


def test_get_raw_model(xgb_model):
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_model.model, name="model", input_example=xgb_model.inference_dataframe.head(3)
        )
    pyfunc_model = pyfunc.load_model(model_info.model_uri)
    raw_model = pyfunc_model.get_raw_model()
    assert type(raw_model) == type(xgb_model.model)
    np.testing.assert_array_almost_equal(
        raw_model.predict(xgb_model.inference_dmatrix),
        xgb_model.model.predict(xgb_model.inference_dmatrix),
    )


def test_xgbooster_predict_exclude_invalid_params(xgb_model):
    signature = infer_signature(
        xgb_model.inference_dataframe.head(3), params={"invalid_param": 1, "approx_contribs": True}
    )
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(xgb_model.model, name="model", signature=signature)
    pyfunc_model = pyfunc.load_model(model_info.model_uri)
    with mock.patch("mlflow.xgboost._logger.warning") as mock_warning:
        np.testing.assert_array_almost_equal(
            pyfunc_model.predict(
                xgb_model.inference_dataframe, params={"invalid_param": 2, "approx_contribs": True}
            ),
            xgb_model.model.predict(xgb_model.inference_dmatrix, approx_contribs=True),
        )
        mock_warning.assert_called_once_with(
            "Params {'invalid_param'} are not accepted by the xgboost model, "
            "ignoring them during predict."
        )


def test_xgbmodel_predict_exclude_invalid_params(xgb_sklearn_model):
    signature = infer_signature(
        xgb_sklearn_model.inference_dataframe.head(3),
        params={"invalid_param": 1, "output_margin": True},
    )
    with mlflow.start_run():
        model_info = mlflow.xgboost.log_model(
            xgb_sklearn_model.model, name="model", signature=signature
        )
    pyfunc_model = pyfunc.load_model(model_info.model_uri)
    with mock.patch("mlflow.xgboost._logger.warning") as mock_warning:
        np.testing.assert_array_almost_equal(
            pyfunc_model.predict(
                xgb_sklearn_model.inference_dataframe,
                params={"invalid_param": 2, "output_margin": True},
            ),
            xgb_sklearn_model.model.predict(
                xgb_sklearn_model.inference_dataframe, output_margin=True
            ),
        )
        mock_warning.assert_called_once_with(
            "Params {'invalid_param'} are not accepted by the xgboost model, "
            "ignoring them during predict."
        )


def test_exclude_unrecognized_kwargs():
    def custom_func(*args, **kwargs):
        return [1, 2, 3]

    def custom_func2(data, **kwargs):
        return [2, 3, 4]

    def custom_func3(x, y):
        return x + y

    params = {"data": 1, "x": 1, "y": 2, "z": 3}
    assert _exclude_unrecognized_kwargs(custom_func, params) == params
    assert _exclude_unrecognized_kwargs(custom_func2, params) == params
    assert _exclude_unrecognized_kwargs(custom_func3, params) == {"x": 1, "y": 2}

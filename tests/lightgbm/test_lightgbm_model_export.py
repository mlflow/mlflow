import json
import os
from pathlib import Path
from typing import NamedTuple
from unittest import mock

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn import datasets
from sklearn.pipeline import Pipeline

import mlflow.lightgbm
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.utils
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
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
    [] if _is_available_on_pypi("lightgbm") else ["--env-manager", "local"]
)


class ModelWithData(NamedTuple):
    model: lgb.Booster
    inference_dataframe: pd.DataFrame


@pytest.fixture(scope="module")
def lgb_model():
    iris = datasets.load_iris()
    X = pd.DataFrame(
        iris.data[:, :2],
        columns=iris.feature_names[:2],  # we only take the first two features.
    )
    y = iris.target

    dtrain = lgb.Dataset(X, y)
    model = lgb.train({"objective": "multiclass", "num_class": 3}, dtrain)
    return ModelWithData(model=model, inference_dataframe=X)


@pytest.fixture(scope="module")
def lgb_model_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="sepal length (cm)", type=DataType.double),
                ColSpec(name="sepal width (cm)", type=DataType.double),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1, 3))]),
    )


@pytest.fixture(scope="module")
def lgb_sklearn_model():
    iris = datasets.load_iris()
    X = pd.DataFrame(
        iris.data[:, :2],
        columns=iris.feature_names[:2],  # we only take the first two features.
    )
    y = iris.target
    model = lgb.LGBMClassifier(n_estimators=10)
    model.fit(X, y)
    return ModelWithData(model=model, inference_dataframe=X)


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def lgb_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["lightgbm", "pytest"])
    return conda_env


def test_model_save_load(lgb_model, model_path):
    model = lgb_model.model

    mlflow.lightgbm.save_model(lgb_model=model, path=model_path)
    reloaded_model = mlflow.lightgbm.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        model.predict(lgb_model.inference_dataframe),
        reloaded_model.predict(lgb_model.inference_dataframe),
    )

    np.testing.assert_array_almost_equal(
        reloaded_model.predict(lgb_model.inference_dataframe),
        reloaded_pyfunc.predict(lgb_model.inference_dataframe),
    )


def test_sklearn_model_save_load(lgb_sklearn_model, model_path):
    model = lgb_sklearn_model.model
    mlflow.lightgbm.save_model(lgb_model=model, path=model_path)
    reloaded_model = mlflow.lightgbm.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        model.predict(lgb_sklearn_model.inference_dataframe),
        reloaded_model.predict(lgb_sklearn_model.inference_dataframe),
    )

    np.testing.assert_array_almost_equal(
        reloaded_model.predict(lgb_sklearn_model.inference_dataframe),
        reloaded_pyfunc.predict(lgb_sklearn_model.inference_dataframe),
    )


def test_signature_and_examples_are_saved_correctly(lgb_model, lgb_model_signature):
    model = lgb_model.model
    X = lgb_model.inference_dataframe
    example_ = X.head(3)
    for signature in (None, lgb_model_signature):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.lightgbm.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == lgb_model_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_load_from_remote_uri_succeeds(lgb_model, model_path, mock_s3_bucket):
    mlflow.lightgbm.save_model(lgb_model=lgb_model.model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.lightgbm.load_model(model_uri=model_uri)
    np.testing.assert_array_almost_equal(
        lgb_model.model.predict(lgb_model.inference_dataframe),
        reloaded_model.predict(lgb_model.inference_dataframe),
    )


def test_model_log(lgb_model, model_path):
    model = lgb_model.model
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "model"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["xgboost"])

                model_info = mlflow.lightgbm.log_model(
                    model, name=artifact_path, conda_env=conda_env
                )
                reloaded_model = mlflow.lightgbm.load_model(model_uri=model_info.model_uri)
                np.testing.assert_array_almost_equal(
                    model.predict(lgb_model.inference_dataframe),
                    reloaded_model.predict(lgb_model.inference_dataframe),
                )

                model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()


def test_log_model_calls_register_model(lgb_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["lightgbm"])
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model,
            name=artifact_path,
            conda_env=conda_env,
            registered_model_name="AdsModel1",
        )
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_info.model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_no_registered_model_name(lgb_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["lightgbm"])
        mlflow.lightgbm.log_model(lgb_model.model, name=artifact_path, conda_env=conda_env)
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    lgb_model, model_path, lgb_custom_env
):
    mlflow.lightgbm.save_model(lgb_model=lgb_model.model, path=model_path, conda_env=lgb_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != lgb_custom_env

    with open(lgb_custom_env) as f:
        lgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == lgb_custom_env_parsed


def test_model_save_persists_requirements_in_mlflow_model_directory(
    lgb_model, model_path, lgb_custom_env
):
    mlflow.lightgbm.save_model(lgb_model=lgb_model.model, path=model_path, conda_env=lgb_custom_env)

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(lgb_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(lgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", pip_requirements=str(req_file)
        )
        _assert_pip_requirements(model_info.model_uri, [expected_mlflow_version, "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(lgb_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.lightgbm.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri, [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name="model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(lgb_model, model_path):
    conda_env = dict(mlflow.lightgbm.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.lightgbm.save_model(lgb_model=lgb_model.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    lgb_model, lgb_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name=artifact_path, conda_env=lgb_custom_env
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != lgb_custom_env

    with open(lgb_custom_env) as f:
        lgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == lgb_custom_env_parsed


def test_model_log_persists_requirements_in_mlflow_model_directory(lgb_model, lgb_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name=artifact_path, conda_env=lgb_custom_env
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_info.model_uri)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(lgb_custom_env, saved_pip_req_path)


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    lgb_model, model_path
):
    mlflow.lightgbm.save_model(lgb_model=lgb_model.model, path=model_path)
    _assert_pip_requirements(model_path, mlflow.lightgbm.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    lgb_model,
):
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(lgb_model.model, name="model")

    _assert_pip_requirements(model_info.model_uri, mlflow.lightgbm.get_default_pip_requirements())


def test_pyfunc_serve_and_score(lgb_model):
    model, inference_dataframe = lgb_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
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


def get_sklearn_models():
    model = lgb.LGBMClassifier(n_estimators=10)
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


def test_load_pyfunc_succeeds_for_older_models_with_pyfunc_data_field(lgb_model, model_path):
    """
    This test verifies that LightGBM models saved in older versions of MLflow are loaded
    successfully by ``mlflow.pyfunc.load_model``. These older models specify a pyfunc ``data``
    field referring directly to a LightGBM model file. Newer models also have the
    ``model_class`` in LightGBM flavor.
    """
    model = lgb_model.model
    mlflow.lightgbm.save_model(lgb_model=model, path=model_path)

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    lgb_conf = model_conf.flavors.get(mlflow.lightgbm.FLAVOR_NAME)
    assert lgb_conf is not None
    assert "model_class" in lgb_conf
    assert "data" in lgb_conf
    assert pyfunc_conf is not None
    assert "model_class" not in pyfunc_conf
    assert pyfunc.DATA in pyfunc_conf

    # test old MLmodel conf
    model_conf.flavors["lightgbm"] = {"lgb_version": lgb.__version__, "data": "model.lgb"}
    model_conf.save(model_conf_path)
    model_conf = Model.load(model_conf_path)
    lgb_conf = model_conf.flavors.get(mlflow.lightgbm.FLAVOR_NAME)
    assert "data" in lgb_conf
    assert lgb_conf["data"] == "model.lgb"

    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)
    assert isinstance(reloaded_pyfunc._model_impl.lgb_model, lgb.Booster)
    reloaded_lgb = mlflow.lightgbm.load_model(model_uri=model_path)
    assert isinstance(reloaded_lgb, lgb.Booster)

    np.testing.assert_array_almost_equal(
        lgb_model.model.predict(lgb_model.inference_dataframe),
        reloaded_pyfunc.predict(lgb_model.inference_dataframe),
    )

    np.testing.assert_array_almost_equal(
        reloaded_lgb.predict(lgb_model.inference_dataframe),
        reloaded_pyfunc.predict(lgb_model.inference_dataframe),
    )


def test_log_model_with_code_paths(lgb_model):
    artifact_path = "model"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.lightgbm._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name=artifact_path, code_paths=[__file__]
        )
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.lightgbm.FLAVOR_NAME)
        mlflow.lightgbm.load_model(model_info.model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(lgb_model, model_path):
    mlflow.lightgbm.save_model(lgb_model.model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def test_model_save_load_with_metadata(lgb_model, model_path):
    mlflow.lightgbm.save_model(
        lgb_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(lgb_model):
    artifact_path = "model"

    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model,
            name=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(lgb_model):
    artifact_path = "model"
    X = lgb_model.inference_dataframe
    example = X.head(3)

    with mlflow.start_run():
        model_info = mlflow.lightgbm.log_model(
            lgb_model.model, name=artifact_path, input_example=example
        )

    mlflow_model = Model.load(model_info.model_uri)
    assert mlflow_model.signature == ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="sepal length (cm)", type=DataType.double),
                ColSpec(name="sepal width (cm)", type=DataType.double),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1, 3))]),
    )

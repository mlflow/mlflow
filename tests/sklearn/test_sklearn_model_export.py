import json
import os
import pickle
import tempfile
from collections import namedtuple
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.linear_model as glm
import sklearn.naive_bayes as nb
import sklearn.neighbors as knn
import yaml
from packaging.version import Version
from sklearn import datasets
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
import mlflow.sklearn
import mlflow.utils
from mlflow import pyfunc
from mlflow.entities.model_registry.model_version import ModelVersion, ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.utils import _read_example, load_serving_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
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
    _is_available_on_pypi,
    _mlflow_major_version_string,
    assert_register_model_called_with_local_model_path,
    pyfunc_serve_and_score_model,
)
from tests.store._unity_catalog.conftest import (
    configure_client_for_uc,  # noqa: F401
    mock_databricks_uc_host_creds,  # noqa: F401
)

EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("scikit-learn", module="sklearn") else ["--env-manager", "local"]
)

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def iris_df():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_df = pd.DataFrame(X, columns=iris.feature_names)
    X_df = X_df.iloc[:, :2]  # we only take the first two features.
    y_series = pd.Series(y)
    return X_df, y_series


@pytest.fixture(scope="module")
def iris_signature():
    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(name="sepal length (cm)", type=DataType.double),
                ColSpec(name="sepal width (cm)", type=DataType.double),
            ]
        ),
        outputs=Schema([ColSpec(type=DataType.long)]),
    )


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_df):
    X, y = iris_df
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture(scope="module")
def sklearn_logreg_model(iris_df):
    X, y = iris_df
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(X, y)
    return ModelWithData(model=linear_lr, inference_data=X)


@pytest.fixture(scope="module")
def sklearn_gaussian_model(iris_df):
    X, y = iris_df
    gaussian_nb = nb.GaussianNB()
    gaussian_nb.fit(X, y)
    return ModelWithData(model=gaussian_nb, inference_data=X)


@pytest.fixture(scope="module")
def sklearn_custom_transformer_model(sklearn_knn_model, iris_df):
    def transform(vec):
        return vec + 1

    transformer = SKFunctionTransformer(transform, validate=True)
    pipeline = SKPipeline([("custom_transformer", transformer), ("knn", sklearn_knn_model.model)])
    X, _ = iris_df
    return ModelWithData(pipeline, inference_data=X)


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


@pytest.fixture
def sklearn_custom_env(tmp_path):
    conda_env = os.path.join(tmp_path, "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn", "pytest"])
    return conda_env


def test_model_save_load(sklearn_knn_model, model_path):
    knn_model = sklearn_knn_model.model

    mlflow.sklearn.save_model(sk_model=knn_model, path=model_path)
    reloaded_knn_model = mlflow.sklearn.load_model(model_uri=model_path)
    reloaded_knn_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_equal(
        knn_model.predict(sklearn_knn_model.inference_data),
        reloaded_knn_model.predict(sklearn_knn_model.inference_data),
    )

    np.testing.assert_array_equal(
        reloaded_knn_model.predict(sklearn_knn_model.inference_data),
        reloaded_knn_pyfunc.predict(sklearn_knn_model.inference_data),
    )


def test_model_save_behavior_with_preexisting_folders(sklearn_knn_model, tmp_path):
    sklearn_model_path = tmp_path / "sklearn_model_empty_exists"
    sklearn_model_path.mkdir()
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)

    sklearn_model_path = tmp_path / "sklearn_model_filled_exists"
    sklearn_model_path.mkdir()
    (sklearn_model_path / "foo.txt").write_text("dummy content")
    with pytest.raises(MlflowException, match="already exists and is not empty"):
        mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=sklearn_model_path)


def test_signature_and_examples_are_saved_correctly(sklearn_knn_model, iris_signature):
    data = sklearn_knn_model.inference_data
    model = sklearn_knn_model.model
    example_ = data[:3]
    for signature in (None, iris_signature):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.sklearn.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                if signature is None and example is None:
                    assert mlflow_model.signature is None
                else:
                    assert mlflow_model.signature == iris_signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_array_equal(_read_example(mlflow_model, path), example)


def test_model_load_from_remote_uri_succeeds(sklearn_knn_model, model_path, mock_s3_bucket):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_knn_model = mlflow.sklearn.load_model(model_uri=model_uri)
    np.testing.assert_array_equal(
        sklearn_knn_model.model.predict(sklearn_knn_model.inference_data),
        reloaded_knn_model.predict(sklearn_knn_model.inference_data),
    )


def test_model_log(sklearn_logreg_model, model_path):
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "linear"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])

                model_info = mlflow.sklearn.log_model(
                    sk_model=sklearn_logreg_model.model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                )
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                assert model_info.model_uri == model_uri

                reloaded_logsklearn_knn_model = mlflow.sklearn.load_model(model_uri=model_uri)
                np.testing.assert_array_equal(
                    sklearn_logreg_model.model.predict(sklearn_logreg_model.inference_data),
                    reloaded_logsklearn_knn_model.predict(sklearn_logreg_model.inference_data),
                )

                model_path = _download_artifact_from_uri(artifact_uri=model_uri)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()


def test_log_model_calls_register_model(sklearn_logreg_model):
    artifact_path = "linear"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])
        mlflow.sklearn.log_model(
            sk_model=sklearn_logreg_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="AdsModel1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert_register_model_called_with_local_model_path(
            register_model_mock=mlflow.tracking._model_registry.fluent._register_model,
            model_uri=model_uri,
            registered_model_name="AdsModel1",
        )


def test_log_model_call_register_model_to_uc(configure_client_for_uc, sklearn_logreg_model):
    artifact_path = "linear"
    mock_model_version = ModelVersion(
        name="AdsModel1",
        version=1,
        creation_timestamp=123,
        status=ModelVersionStatus.to_string(ModelVersionStatus.READY),
    )
    with mock.patch.object(UcModelRegistryStore, "create_registered_model"), mock.patch.object(
        UcModelRegistryStore, "create_model_version", return_value=mock_model_version, autospec=True
    ) as mock_create_mv, TempDir(chdr=True, remove_on_exit=True) as tmp:
        with mlflow.start_run():
            conda_env = os.path.join(tmp.path(), "conda_env.yaml")
            _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])
            mlflow.sklearn.log_model(
                sk_model=sklearn_logreg_model.model,
                artifact_path=artifact_path,
                conda_env=conda_env,
                registered_model_name="AdsModel1",
            )
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id
            [(args, kwargs)] = mock_create_mv.call_args_list
            expected_source = os.path.join(active_run.info.artifact_uri, artifact_path)
            assert args[1:] == ("AdsModel1", expected_source, run_id, [], None, None)
            assert kwargs["local_model_path"].startswith(tempfile.gettempdir())


def test_log_model_no_registered_model_name(sklearn_logreg_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.tracking._model_registry.fluent._register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])
        mlflow.sklearn.log_model(
            sk_model=sklearn_logreg_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
        )
        mlflow.tracking._model_registry.fluent._register_model.assert_not_called()


def test_custom_transformer_can_be_saved_and_loaded_with_cloudpickle_format(
    sklearn_custom_transformer_model, tmp_path
):
    custom_transformer_model = sklearn_custom_transformer_model.model

    # Because the model contains a customer transformer that is not defined at the top level of the
    # current test module, we expect pickle to fail when attempting to serialize it. In contrast,
    # we expect cloudpickle to successfully locate the transformer definition and serialize the
    # model successfully.
    pickle_format_model_path = os.path.join(tmp_path, "pickle_model")
    with pytest.raises(AttributeError, match="Can't pickle local object"):
        mlflow.sklearn.save_model(
            sk_model=custom_transformer_model,
            path=pickle_format_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    cloudpickle_format_model_path = os.path.join(tmp_path, "cloud_pickle_model")
    mlflow.sklearn.save_model(
        sk_model=custom_transformer_model,
        path=cloudpickle_format_model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    reloaded_custom_transformer_model = mlflow.sklearn.load_model(
        model_uri=cloudpickle_format_model_path
    )

    np.testing.assert_array_equal(
        custom_transformer_model.predict(sklearn_custom_transformer_model.inference_data),
        reloaded_custom_transformer_model.predict(sklearn_custom_transformer_model.inference_data),
    )


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, model_path, sklearn_custom_env
):
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model, path=model_path, conda_env=sklearn_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != sklearn_custom_env

    with open(sklearn_custom_env) as f:
        sklearn_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == sklearn_custom_env_parsed


def test_model_save_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, model_path, sklearn_custom_env
):
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model, path=model_path, conda_env=sklearn_custom_env
    )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(sklearn_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(sklearn_knn_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.sklearn.log_model(sklearn_knn_model.model, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a"], strict=True
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(sklearn_knn_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(sklearn_knn_model, model_path):
    conda_env = dict(mlflow.sklearn.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    sklearn_knn_model, sklearn_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            conda_env=sklearn_custom_env,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != sklearn_custom_env

    with open(sklearn_custom_env) as f:
        sklearn_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == sklearn_custom_env_parsed


def test_model_log_persists_requirements_in_mlflow_model_directory(
    sklearn_knn_model, sklearn_custom_env
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            conda_env=sklearn_custom_env,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(sklearn_custom_env, saved_pip_req_path)


def test_model_save_throws_exception_if_serialization_format_is_unrecognized(
    sklearn_knn_model, model_path
):
    with pytest.raises(MlflowException, match="Unrecognized serialization format") as exc:
        mlflow.sklearn.save_model(
            sk_model=sklearn_knn_model.model,
            path=model_path,
            serialization_format="not a valid format",
        )
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    # The unsupported serialization format should have been detected prior to the execution of
    # any directory creation or state-mutating persistence logic that would prevent a second
    # serialization call with the same model path from succeeding
    assert not os.path.exists(model_path)
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)
    _assert_pip_requirements(
        model_path, mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)
    )


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    sklearn_knn_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model.model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    _assert_pip_requirements(
        model_uri, mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)
    )


def test_model_save_uses_cloudpickle_serialization_format_by_default(sklearn_knn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)

    sklearn_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    assert "serialization_format" in sklearn_conf
    assert sklearn_conf["serialization_format"] == mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


def test_model_log_uses_cloudpickle_serialization_format_by_default(sklearn_knn_model):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=sklearn_knn_model.model, artifact_path=artifact_path)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    sklearn_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    assert "serialization_format" in sklearn_conf
    assert sklearn_conf["serialization_format"] == mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE


def test_model_save_with_cloudpickle_format_adds_cloudpickle_to_conda_environment(
    sklearn_knn_model, model_path
):
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model,
        path=model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    sklearn_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    assert "serialization_format" in sklearn_conf
    assert sklearn_conf["serialization_format"] == mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
    assert os.path.exists(saved_conda_env_path)
    with open(saved_conda_env_path) as f:
        saved_conda_env_parsed = yaml.safe_load(f)

    pip_deps = [
        dependency
        for dependency in saved_conda_env_parsed["dependencies"]
        if type(dependency) == dict and "pip" in dependency
    ]
    assert len(pip_deps) == 1
    assert any("cloudpickle" in pip_dep for pip_dep in pip_deps[0]["pip"])


def test_model_save_without_cloudpickle_format_does_not_add_cloudpickle_to_conda_environment(
    sklearn_knn_model, model_path
):
    non_cloudpickle_serialization_formats = list(mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS)
    non_cloudpickle_serialization_formats.remove(mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    for serialization_format in non_cloudpickle_serialization_formats:
        mlflow.sklearn.save_model(
            sk_model=sklearn_knn_model.model,
            path=model_path,
            serialization_format=serialization_format,
        )

        sklearn_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
        )
        assert "serialization_format" in sklearn_conf
        assert sklearn_conf["serialization_format"] == serialization_format

        pyfunc_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
        )
        saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV]["conda"])
        assert os.path.exists(saved_conda_env_path)
        with open(saved_conda_env_path) as f:
            saved_conda_env_parsed = yaml.safe_load(f)
        assert all(
            "cloudpickle" not in dependency for dependency in saved_conda_env_parsed["dependencies"]
        )


def test_load_pyfunc_succeeds_for_older_models_with_pyfunc_data_field(
    sklearn_knn_model, model_path
):
    """
    This test verifies that scikit-learn models saved in older versions of MLflow are loaded
    successfully by ``mlflow.pyfunc.load_model``. These older models specify a pyfunc ``data``
    field referring directly to a serialized scikit-learn model file. In contrast, newer models
    omit the ``data`` field.
    """
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model,
        path=model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    pyfunc_conf = model_conf.flavors.get(pyfunc.FLAVOR_NAME)
    sklearn_conf = model_conf.flavors.get(mlflow.sklearn.FLAVOR_NAME)
    assert sklearn_conf is not None
    assert pyfunc_conf is not None
    pyfunc_conf[pyfunc.DATA] = sklearn_conf["pickled_model"]

    reloaded_knn_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_equal(
        sklearn_knn_model.model.predict(sklearn_knn_model.inference_data),
        reloaded_knn_pyfunc.predict(sklearn_knn_model.inference_data),
    )


def test_add_pyfunc_flavor_only_when_model_defines_predict(model_path):
    from sklearn.cluster import AgglomerativeClustering

    sk_model = AgglomerativeClustering()
    assert not hasattr(sk_model, "predict")

    mlflow.sklearn.save_model(
        sk_model=sk_model,
        path=model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )

    model_conf_path = os.path.join(model_path, "MLmodel")
    model_conf = Model.load(model_conf_path)
    assert pyfunc.FLAVOR_NAME not in model_conf.flavors


def test_pyfunc_serve_and_score(sklearn_knn_model):
    model, inference_dataframe = sklearn_knn_model
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            model, artifact_path, input_example=inference_dataframe
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


@pytest.mark.skipif(
    Version(sklearn.__version__) != Version("1.2.2"),
    reason="'sklearn.metrics._dist_metrics' doesn't have attribute 'EuclideanDistance'",
)
def test_sklearn_compatible_with_mlflow_2_4_0(sklearn_knn_model, tmp_path):
    model, inference_dataframe = sklearn_knn_model
    model_predict = model.predict(inference_dataframe)

    # save test model
    tmp_path.joinpath("MLmodel").write_text(
        f"""
artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.8.16
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: {sklearn.__version__}
mlflow_version: 2.4.0
model_uuid: c9833d74b1ff4013a1c9eff05d39eeef
run_id: 8146a2ae86104f5b853351e600fc9d7b
utc_time_created: '2023-07-04 07:19:43.561797'
"""
    )
    tmp_path.joinpath("python_env.yaml").write_text(
        """
python: 3.8.16
build_dependencies:
   - pip==23.1.2
   - setuptools==56.0.0
   - wheel==0.40.0
dependencies:
   - -r requirements.txt
"""
    )
    tmp_path.joinpath("requirements.txt").write_text(
        f"""
mlflow==2.4.0
cloudpickle
numpy
psutil
scikit-learn=={sklearn.__version__}
scipy
"""
    )
    with open(tmp_path / "model.pkl", "wb") as out:
        pickle.dump(model, out, protocol=pickle.DEFAULT_PROTOCOL)

    assert Version(mlflow.__version__) > Version("2.4.0")
    model_uri = str(tmp_path)
    pyfunc_loaded = mlflow.pyfunc.load_model(model_uri)

    # predict is compatible
    local_predict = pyfunc_loaded.predict(inference_dataframe)
    np.testing.assert_array_almost_equal(local_predict, model_predict)

    # model serving is compatible
    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=pd.DataFrame(inference_dataframe),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.DataFrame(
        data=json.loads(resp.content.decode("utf-8"))["predictions"]
    ).values.squeeze()
    np.testing.assert_array_almost_equal(scores, model_predict)

    # Issues a warning if params are specified prior to MLflow support in 2.5.0
    with mock.patch("mlflow.models.utils._logger.warning") as mock_warning:
        pyfunc_loaded.predict(inference_dataframe, params={"top_k": 2})
    mock_warning.assert_called_with(
        "`params` can only be specified at inference time if the model signature defines a params "
        "schema. This model does not define a params schema. Ignoring provided params: "
        "['top_k']"
    )


def test_log_model_with_code_paths(sklearn_knn_model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.sklearn._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.sklearn.log_model(sklearn_knn_model.model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.sklearn.FLAVOR_NAME)
        mlflow.sklearn.load_model(model_uri=model_uri)
        add_mock.assert_called()


@pytest.mark.parametrize(
    "predict_fn", ["predict", "predict_proba", "predict_log_proba", "predict_joint_log_proba"]
)
def test_log_model_with_custom_pyfunc_predict_fn(sklearn_gaussian_model, predict_fn):
    if Version(sklearn.__version__) < Version("1.2.0") and predict_fn == "predict_joint_log_proba":
        pytest.skip("predict_joint_log_proba is not available in scikit-learn < 1.2.0")

    model, inference_dataframe = sklearn_gaussian_model
    expected_scores = getattr(model, predict_fn)(inference_dataframe)
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path, pyfunc_predict_fn=predict_fn)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    loaded_model = pyfunc.load_model(model_uri)
    actual_scores = loaded_model.predict(inference_dataframe)
    np.testing.assert_array_almost_equal(expected_scores, actual_scores)


def test_virtualenv_subfield_points_to_correct_path(sklearn_logreg_model, model_path):
    mlflow.sklearn.save_model(sklearn_logreg_model.model, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def test_model_save_load_with_metadata(sklearn_knn_model, model_path):
    mlflow.sklearn.save_model(
        sklearn_knn_model.model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(sklearn_knn_model):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model,
            artifact_path=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_signature_inference(sklearn_knn_model, iris_signature):
    artifact_path = "model"
    X = sklearn_knn_model.inference_data
    example = X.iloc[[0]]

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, artifact_path=artifact_path, input_example=example
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    mlflow_model = Model.load(model_uri)
    assert mlflow_model.signature == iris_signature


def test_model_size_bytes(sklearn_logreg_model, tmp_path):
    mlflow.sklearn.save_model(sklearn_logreg_model.model, path=tmp_path)

    # expected size only counts for files saved before the MLmodel file is saved
    model_file = tmp_path.joinpath("model.pkl")
    with model_file.open("rb") as fp:
        expected_size = len(fp.read())

    mlmodel = yaml.safe_load(tmp_path.joinpath("MLmodel").read_bytes())
    assert mlmodel["model_size_bytes"] == expected_size


def test_model_registration_metadata_handling(sklearn_knn_model, tmp_path):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name="test",
        )
        model_uri = "models:/test/1"

    artifact_repository = get_artifact_repository(model_uri)

    dst_full = tmp_path.joinpath("full")
    dst_full.mkdir()

    artifact_repository.download_artifacts("MLmodel", dst_full)
    # This validates that the models artifact repo will not attempt to create a
    # "registered model metadata" file if the source of an artifact download is a file.
    assert os.listdir(dst_full) == ["MLmodel"]


def test_pipeline_predict_proba(sklearn_knn_model, model_path):
    knn_model = sklearn_knn_model.model
    pipeline = make_pipeline(knn_model)

    mlflow.sklearn.save_model(sk_model=pipeline, path=model_path, pyfunc_predict_fn="predict_proba")
    reloaded_knn_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_equal(
        knn_model.predict_proba(sklearn_knn_model.inference_data),
        reloaded_knn_pyfunc.predict(sklearn_knn_model.inference_data),
    )


def test_get_raw_model(sklearn_knn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", input_example=sklearn_knn_model.inference_data
        )
    pyfunc_model = pyfunc.load_model(model_info.model_uri)
    raw_model = pyfunc_model.get_raw_model()
    assert type(raw_model) == type(sklearn_knn_model.model)
    np.testing.assert_array_equal(
        raw_model.predict(sklearn_knn_model.inference_data),
        sklearn_knn_model.model.predict(sklearn_knn_model.inference_data),
    )

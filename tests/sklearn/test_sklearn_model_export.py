from unittest import mock
import os
import pytest
import yaml
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn import datasets
import sklearn.linear_model as glm
import sklearn.neighbors as knn
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.sklearn
import mlflow.utils
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _read_example
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from mlflow.models import Model, infer_signature
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import (
    pyfunc_serve_and_score_model,
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    _is_available_on_pypi,
    _compare_logged_code_paths,
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


@pytest.fixture(scope="module")
def sklearn_logreg_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(X, y)
    return ModelWithData(model=linear_lr, inference_data=X)


@pytest.fixture(scope="module")
def sklearn_custom_transformer_model(sklearn_knn_model):
    def transform(vec):
        return vec + 1

    transformer = SKFunctionTransformer(transform, validate=True)
    pipeline = SKPipeline([("custom_transformer", transformer), ("knn", sklearn_knn_model.model)])
    return ModelWithData(pipeline, inference_data=datasets.load_iris().data[:, :2])


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def sklearn_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
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


def test_signature_and_examples_are_saved_correctly(sklearn_knn_model):
    data = sklearn_knn_model.inference_data
    model = sklearn_knn_model.model
    signature_ = infer_signature(data)
    example_ = data[
        :3,
    ]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.sklearn.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    np.testing.assert_array_equal(_read_example(mlflow_model, path), example)


def test_model_load_from_remote_uri_succeeds(sklearn_knn_model, model_path, mock_s3_bucket):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
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
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
                )
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
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()


def test_log_model_calls_register_model(sklearn_logreg_model):
    artifact_path = "linear"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])
        mlflow.sklearn.log_model(
            sk_model=sklearn_logreg_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="AdsModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "AdsModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(sklearn_logreg_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["scikit-learn"])
        mlflow.sklearn.log_model(
            sk_model=sklearn_logreg_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
        )
        mlflow.register_model.assert_not_called()


def test_custom_transformer_can_be_saved_and_loaded_with_cloudpickle_format(
    sklearn_custom_transformer_model, tmpdir
):
    custom_transformer_model = sklearn_custom_transformer_model.model

    # Because the model contains a customer transformer that is not defined at the top level of the
    # current test module, we expect pickle to fail when attempting to serialize it. In contrast,
    # we expect cloudpickle to successfully locate the transformer definition and serialize the
    # model successfully.
    pickle_format_model_path = os.path.join(str(tmpdir), "pickle_model")
    with pytest.raises(AttributeError, match="Can't pickle local object"):
        mlflow.sklearn.save_model(
            sk_model=custom_transformer_model,
            path=pickle_format_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    cloudpickle_format_model_path = os.path.join(str(tmpdir), "cloud_pickle_model")
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
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != sklearn_custom_env

    with open(sklearn_custom_env, "r") as f:
        sklearn_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
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


def test_log_model_with_pip_requirements(sklearn_knn_model, tmpdir):
    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(sklearn_knn_model, tmpdir):
    default_reqs = mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sklearn_knn_model.model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", *default_reqs, "b", "-c constraints.txt"],
            ["a"],
        )


def test_model_save_accepts_conda_env_as_dict(sklearn_knn_model, model_path):
    conda_env = dict(mlflow.sklearn.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
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
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != sklearn_custom_env

    with open(sklearn_custom_env, "r") as f:
        sklearn_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
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
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

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
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

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
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    with open(saved_conda_env_path, "r") as f:
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
        saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
        assert os.path.exists(saved_conda_env_path)
        with open(saved_conda_env_path, "r") as f:
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
        mlflow.sklearn.log_model(model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=pd.DataFrame(inference_dataframe),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records").values.squeeze()
    np.testing.assert_array_almost_equal(scores, model.predict(inference_dataframe))


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


def test_log_predict_proba(sklearn_logreg_model):
    model, inference_dataframe = sklearn_logreg_model
    expected_scores = model.predict_proba(inference_dataframe)
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path, pyfunc_predict_fn="predict_proba")
        model_uri = mlflow.get_artifact_uri(artifact_path)

    loaded_model = pyfunc.load_model(model_uri)
    actual_scores = loaded_model.predict(inference_dataframe)
    np.testing.assert_array_almost_equal(expected_scores, actual_scores)

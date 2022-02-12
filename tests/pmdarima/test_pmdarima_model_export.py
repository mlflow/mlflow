import os
import pytest
from unittest import mock

import pmdarima
import numpy as np
import pandas as pd
import yaml

import mlflow.pmdarima
from mlflow import pyfunc
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.prophet.test_prophet_model_export import DataGeneration

from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    pyfunc_serve_and_score_model,
)

pytestmark = pytest.mark.large


@pytest.fixture(scope="function")
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def pmdarima_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
    return conda_env


@pytest.fixture(scope="module")
def test_data():

    data_conf = {
        "shift": False,
        "start": "2016-01-01",
        "size": 365 * 3,
        "seasonal_period": 7,
        "seasonal_freq": 0.1,
        "date_field": "date",
        "target_field": "orders",
    }
    raw = DataGeneration(**data_conf).create_series_df()
    return raw.set_index("date")


@pytest.fixture(scope="module")
def auto_arima_model(test_data):

    return pmdarima.auto_arima(
        test_data["orders"], max_d=1, suppress_warnings=True, error_action="raise"
    )


@pytest.fixture(scope="module")
def auto_arima_object_model(test_data):

    model = pmdarima.arima.ARIMA(order=(2, 1, 3), maxiter=25)
    return model.fit(test_data["orders"])


def test_pmdarima_auto_arima_save_and_load(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)

    loaded_model = mlflow.pmdarima.load_model(model_uri=model_path)

    np.testing.assert_array_equal(auto_arima_model.predict(10), loaded_model.predict(10))


def test_pmdarima_arima_object_save_and_load(auto_arima_object_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_object_model, path=model_path)

    loaded_model = mlflow.pmdarima.load_model(model_uri=model_path)

    np.testing.assert_array_equal(auto_arima_object_model.predict(30), loaded_model.predict(30))


def test_pmdarima_autoarima_pyfunc_save_and_load(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri=model_path)

    predict_conf = pd.DataFrame({"n_periods": 60, "return_conf_int": True, "alpha": 0.1}, index=[0])

    model_predict = auto_arima_model.predict(n_periods=60, return_conf_int=True, alpha=0.1)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    for idx, arr in enumerate(model_predict):
        np.testing.assert_array_equal(arr, pyfunc_predict[idx])


def test_pmdarima_signature_and_examples_saved_correctly(auto_arima_model, test_data):

    # NB: with return_conf_int=True, the return type of pmdarima models is a tuple.
    prediction = auto_arima_model.predict(n_periods=20, return_conf_int=True, alpha=0.05)
    signature_ = infer_signature(test_data, prediction[0])
    example_ = test_data[0:5].copy(deep=False)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.pmdarima.save_model(
                    auto_arima_model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    r_example = _read_example(mlflow_model, path).copy(deep=False)
                    np.testing.assert_array_equal(r_example, example)


def test_pmdarima_load_from_remote_uri_succeeds(
    auto_arima_object_model, model_path, mock_s3_bucket
):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_object_model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_pmdarima_model = mlflow.pmdarima.load_model(model_uri=model_uri)

    np.testing.assert_array_equal(
        auto_arima_object_model.predict(30), reloaded_pmdarima_model.predict(30)
    )


def test_pmdarima_log_model(auto_arima_model):

    old_uri = mlflow.get_tracking_uri()
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()
                artifact_path = "pmdarima"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
                model_info = mlflow.pmdarima.log_model(
                    pmdarima_model=auto_arima_model,
                    artifact_path=artifact_path,
                    conda_env=conda_env,
                )
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                assert model_info.model_uri == model_uri
                reloaded_model = mlflow.pmdarima.load_model(model_uri=model_uri)
                np.testing.assert_array_equal(
                    auto_arima_model.predict(20), reloaded_model.predict(20)
                )

                model_path = _download_artifact_from_uri(artifact_uri=model_uri)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
                assert os.path.exists(os.path.join(model_path, env_path))
            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(old_uri)


def test_pmdarima_log_model_calls_register_model(auto_arima_object_model):
    artifact_path = "pmdarima"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
        mlflow.pmdarima.log_model(
            pmdarima_model=auto_arima_object_model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="PmdarimaModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri, "PmdarimaModel", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_pmdarima_log_model_no_registered_model_name(auto_arima_model):
    artifact_path = "pmdarima"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
        mlflow.pmdarima.log_model(
            pmdarima_model=auto_arima_model, artifact_path=artifact_path, conda_env=conda_env
        )
        mlflow.register_model.assert_not_called()


def test_pmdarima_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    auto_arima_object_model, model_path, pmdarima_custom_env
):
    mlflow.pmdarima.save_model(
        pmdarima_model=auto_arima_object_model, path=model_path, conda_env=pmdarima_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != pmdarima_custom_env

    with open(pmdarima_custom_env, "r") as f:
        pmdarima_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == pmdarima_custom_env_parsed


def test_pmdarima_model_save_persists_requirements_in_mlflow_model_directory(
    auto_arima_model, model_path, pmdarima_custom_env
):
    mlflow.pmdarima.save_model(
        pmdarima_model=auto_arima_model, path=model_path, conda_env=pmdarima_custom_env
    )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(pmdarima_custom_env, saved_pip_req_path)


def test_pmdarima_log_model_with_pip_requirements(auto_arima_object_model, tmpdir):
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_object_model, "model", pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_object_model, "model", pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_object_model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_pmdarima_log_model_with_extra_pip_requirements(auto_arima_model, tmpdir):
    default_reqs = mlflow.pmdarima.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_model, "model", extra_pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            model_uri=mlflow.get_artifact_uri("model"),
            requirements=["mlflow", *default_reqs, "b", "-c constraints.txt"],
            constraints=["a"],
            strict=False,
        )


def test_pmdarima_model_save_without_conda_env_uses_default_env_with_expected_dependencies(
    auto_arima_model, model_path
):
    mlflow.pmdarima.save_model(auto_arima_model, model_path)
    _assert_pip_requirements(model_path, mlflow.pmdarima.get_default_pip_requirements())


def test_pmdarima_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    auto_arima_object_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pmdarima.log_model(auto_arima_object_model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.pmdarima.get_default_pip_requirements())


def test_pmdarima_pyfunc_serve_and_score(auto_arima_model):

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.pmdarima.log_model(auto_arima_model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    local_predict = auto_arima_model.predict(30)

    inference_data = pd.DataFrame({"n_periods": 30}, index=[0])

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records")
    np.testing.assert_array_almost_equal(scores.to_numpy(), local_predict)

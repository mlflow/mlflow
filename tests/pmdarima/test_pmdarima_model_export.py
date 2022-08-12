import os
import pathlib
import pytest
from unittest import mock

import pmdarima
import numpy as np
import pandas as pd
import yaml

import mlflow.pmdarima
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from tests.prophet.test_prophet_model_export import DataGeneration

from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    pyfunc_serve_and_score_model,
    _is_available_on_pypi,
    _compare_logged_code_paths,
)


EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("pmdarima") else ["--env-manager", "local"]
)


@pytest.fixture(scope="function")
def model_path(tmp_path):
    return tmp_path.joinpath("model")


@pytest.fixture
def pmdarima_custom_env(tmp_path):
    conda_env = tmp_path.joinpath("conda_env.yml")
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

    model_predict = auto_arima_model.predict(n_periods=60, return_conf_int=True, alpha=0.1)

    predict_conf = pd.DataFrame({"n_periods": 60, "return_conf_int": True, "alpha": 0.1}, index=[0])
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    np.testing.assert_array_equal(model_predict[0], pyfunc_predict["yhat"])
    yhat_low, yhat_high = list(zip(*model_predict[1]))
    np.testing.assert_array_equal(yhat_low, pyfunc_predict["yhat_lower"])
    np.testing.assert_array_equal(yhat_high, pyfunc_predict["yhat_upper"])


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_pmdarima_signature_and_examples_saved_correctly(
    auto_arima_model, test_data, model_path, use_signature, use_example
):

    # NB: Signature inference will only work on the first element of the tuple return
    prediction = auto_arima_model.predict(n_periods=20, return_conf_int=True, alpha=0.05)
    signature = infer_signature(test_data, prediction[0]) if use_signature else None
    example = test_data[0:5].copy(deep=False) if use_example else None
    mlflow.pmdarima.save_model(
        auto_arima_model, path=model_path, signature=signature, input_example=example
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_pmdarima_signature_and_example_for_confidence_interval_mode(
    auto_arima_model, model_path, test_data, use_signature, use_example
):
    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri=model_path_primary)
    predict_conf = pd.DataFrame([{"n_periods": 10, "return_conf_int": True, "alpha": 0.2}])
    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(test_data[["orders"]], forecast) if use_signature else None
    example = test_data[0:10].copy(deep=False) if use_example else None
    mlflow.pmdarima.save_model(
        auto_arima_model, path=model_path_secondary, signature=signature, input_example=example
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path_secondary).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


def test_pmdarima_load_from_remote_uri_succeeds(
    auto_arima_object_model, model_path, mock_s3_bucket
):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_object_model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    # NB: cloudpathlib would need to be used here to handle object store uri
    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_pmdarima_model = mlflow.pmdarima.load_model(model_uri=model_uri)

    np.testing.assert_array_equal(
        auto_arima_object_model.predict(30), reloaded_pmdarima_model.predict(30)
    )


@pytest.mark.parametrize("should_start_run", [True, False])
def test_pmdarima_log_model(auto_arima_model, tmp_path, should_start_run):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "pmdarima"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
        model_info = mlflow.pmdarima.log_model(
            pmdarima_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.pmdarima.load_model(model_uri=model_uri)
        np.testing.assert_array_equal(auto_arima_model.predict(20), reloaded_model.predict(20))
        model_path = pathlib.Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
        assert model_path.joinpath(env_path).exists()
    finally:
        mlflow.end_run()


def test_pmdarima_log_model_calls_register_model(auto_arima_object_model, tmp_path):
    artifact_path = "pmdarima"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
        mlflow.pmdarima.log_model(
            pmdarima_model=auto_arima_object_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="PmdarimaModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri, "PmdarimaModel", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_pmdarima_log_model_no_registered_model_name(auto_arima_model, tmp_path):
    artifact_path = "pmdarima"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pmdarima"])
        mlflow.pmdarima.log_model(
            pmdarima_model=auto_arima_model, artifact_path=artifact_path, conda_env=str(conda_env)
        )
        mlflow.register_model.assert_not_called()


def test_pmdarima_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    auto_arima_object_model, model_path, pmdarima_custom_env
):
    mlflow.pmdarima.save_model(
        pmdarima_model=auto_arima_object_model, path=model_path, conda_env=str(pmdarima_custom_env)
    )
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = model_path.joinpath(pyfunc_conf[pyfunc.ENV])
    assert saved_conda_env_path.exists()
    assert not pmdarima_custom_env.samefile(saved_conda_env_path)

    pmdarima_custom_env_parsed = yaml.safe_load(pmdarima_custom_env.read_bytes())
    saved_conda_env_parsed = yaml.safe_load(saved_conda_env_path.read_bytes())
    assert saved_conda_env_parsed == pmdarima_custom_env_parsed


def test_pmdarima_model_save_persists_requirements_in_mlflow_model_directory(
    auto_arima_model, model_path, pmdarima_custom_env
):
    mlflow.pmdarima.save_model(
        pmdarima_model=auto_arima_model, path=model_path, conda_env=str(pmdarima_custom_env)
    )
    saved_pip_req_path = model_path.joinpath("requirements.txt")
    _compare_conda_env_requirements(pmdarima_custom_env, str(saved_pip_req_path))


def test_pmdarima_log_model_with_pip_requirements(auto_arima_object_model, tmp_path):
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.pmdarima.log_model(auto_arima_object_model, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_object_model, "model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_object_model, "model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_pmdarima_log_model_with_extra_pip_requirements(auto_arima_model, tmp_path):
    default_reqs = mlflow.pmdarima.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.pmdarima.log_model(auto_arima_model, "model", extra_pip_requirements=str(req_file))
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_model, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            auto_arima_model, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
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
        mlflow.pmdarima.log_model(
            auto_arima_model,
            artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)
    local_predict = auto_arima_model.predict(30)

    inference_data = pd.DataFrame({"n_periods": 30}, index=[0])

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.read_json(resp.content.decode("utf-8"), orient="records").to_numpy().flatten()
    np.testing.assert_array_almost_equal(scores, local_predict)


def test_pmdarima_pyfunc_raises_invalid_df_input(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(pd.DataFrame([{"n_periods": 60}, {"n_periods": 100}]))

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))

    with pytest.raises(MlflowException, match="The provided `n_periods` value "):
        loaded_pyfunc.predict(pd.DataFrame([{"n_periods": "60"}]))


def test_pmdarima_pyfunc_return_correct_structure(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri=model_path)

    predict_conf_no_ci = pd.DataFrame([{"n_periods": 10, "return_conf_int": False}])
    forecast_no_ci = loaded_pyfunc.predict(predict_conf_no_ci)

    assert isinstance(forecast_no_ci, pd.DataFrame)
    assert len(forecast_no_ci) == 10
    assert len(forecast_no_ci.columns.values) == 1

    predict_conf_with_ci = pd.DataFrame([{"n_periods": 10, "return_conf_int": True}])
    forecast_with_ci = loaded_pyfunc.predict(predict_conf_with_ci)

    assert isinstance(forecast_with_ci, pd.DataFrame)
    assert len(forecast_with_ci) == 10
    assert len(forecast_with_ci.columns.values) == 3


def test_log_model_with_code_paths(auto_arima_model):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.pmdarima._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.pmdarima.log_model(auto_arima_model, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.pmdarima.FLAVOR_NAME)
        mlflow.pmdarima.load_model(model_uri)
        add_mock.assert_called()

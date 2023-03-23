import os
import pytest
from unittest import mock
from copy import deepcopy
from pathlib import Path

import json
import yaml
import numpy as np
import pandas as pd

from diviner import GroupedProphet, GroupedPmdarima
from diviner.utils.example_utils import example_data_generator

from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature, Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _read_example
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

import mlflow.diviner
from mlflow import pyfunc

from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    pyfunc_serve_and_score_model,
    _is_available_on_pypi,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
)


DS_FORMAT = "%Y-%m-%dT%H:%M:%S"
EXTRA_PYFUNC_SERVING_TEST_ARGS = (
    [] if _is_available_on_pypi("diviner") else ["--env-manager", "local"]
)


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def diviner_data():
    return example_data_generator.generate_example_data(
        column_count=3, series_count=4, series_size=365 * 4, start_dt="2019-01-01", days_period=1
    )


@pytest.fixture(scope="module")
def grouped_prophet(diviner_data):
    return GroupedProphet(n_changepoints=20, uncertainty_samples=0).fit(
        df=diviner_data.df, group_key_columns=diviner_data.key_columns, y_col="y", datetime_col="ds"
    )


@pytest.fixture(scope="module")
def grouped_pmdarima(diviner_data):
    from pmdarima.arima.auto import AutoARIMA

    base_model = AutoARIMA(out_of_sample_size=60, maxiter=30)
    return GroupedPmdarima(model_template=base_model).fit(
        df=diviner_data.df,
        group_key_columns=diviner_data.key_columns,
        y_col="y",
        datetime_col="ds",
        silence_warnings=True,
    )


@pytest.fixture
def diviner_custom_env(tmp_path):
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["diviner"])
    return conda_env


def test_diviner_native_save_and_load(grouped_prophet, model_path):
    mlflow.diviner.save_model(diviner_model=grouped_prophet, path=model_path)

    loaded = mlflow.diviner.load_model(model_path)

    local_forecast = grouped_prophet.forecast(10, "D")
    loaded_forecast = loaded.forecast(10, "D")

    pd.testing.assert_frame_equal(local_forecast, loaded_forecast)


def test_diviner_pyfunc_save_load(grouped_pmdarima, model_path):
    mlflow.diviner.save_model(diviner_model=grouped_pmdarima, path=model_path)
    loaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    model_predict = grouped_pmdarima.predict(n_periods=10, return_conf_int=True, alpha=0.075)

    predict_conf = pd.DataFrame(
        {"n_periods": 10, "return_conf_int": True, "alpha": 0.075}, index=[0]
    )
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    pd.testing.assert_frame_equal(model_predict, pyfunc_predict)


def test_diviner_pyfunc_invalid_config_raises(grouped_prophet, model_path):
    mlflow.diviner.save_model(diviner_model=grouped_prophet, path=model_path)
    loaded_pyfunc_model = pyfunc.load_pyfunc(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match="The provided prediction configuration Pandas "
        "DataFrame does not contain either the `n_periods` "
        "or `horizon` columns.",
    ):
        loaded_pyfunc_model.predict(pd.DataFrame({"bogus": "config"}, index=[0]))

    with pytest.raises(
        MlflowException,
        match="The `n_periods` column contains invalid data. Supplied type must be an integer.",
    ):
        loaded_pyfunc_model.predict(pd.DataFrame({"n_periods": "20D"}, index=[0]))

    with pytest.raises(
        MlflowException,
        match="Diviner's GroupedProphet model requires a `frequency` value to be submitted",
    ):
        loaded_pyfunc_model.predict(pd.DataFrame({"horizon": 30}, index=[0]))

    bad_conf = pd.DataFrame({"n_periods": 30, "horizon": 20, "frequency": "D"}, index=[0])
    with pytest.raises(
        MlflowException,
        match="The provided prediction configuration contains both "
        "`n_periods` and `horizon` with different values.",
    ):
        loaded_pyfunc_model.predict(bad_conf)


def test_diviner_pyfunc_group_predict_prophet(grouped_prophet, model_path, diviner_data):
    groups = []
    for i in [0, -1]:
        key_entries = []
        for value in diviner_data.df[diviner_data.key_columns].iloc[[i]].to_dict().values():
            key_entries.append(list(value.values())[0])
        groups.append(tuple(key_entries))

    mlflow.diviner.save_model(diviner_model=grouped_prophet, path=model_path)
    loaded_pyfunc_model = pyfunc.load_pyfunc(model_uri=model_path)

    local_group_pred = grouped_prophet.predict_groups(groups=groups, horizon=10, frequency="D")
    pyfunc_conf = pd.DataFrame({"groups": [groups], "horizon": 10, "frequency": "D"}, index=[0])
    pyfunc_group_predict = loaded_pyfunc_model.predict(pyfunc_conf)

    pd.testing.assert_frame_equal(local_group_pred, pyfunc_group_predict)


def test_diviner_pyfunc_group_predict_pmdarima(grouped_pmdarima, model_path, diviner_data):
    groups = []
    for i in [0, -1]:
        key_entries = []
        for value in diviner_data.df[diviner_data.key_columns].iloc[[i]].to_dict().values():
            key_entries.append(list(value.values())[0])
        groups.append(tuple(key_entries))

    mlflow.diviner.save_model(diviner_model=grouped_pmdarima, path=model_path)
    loaded_pyfunc_model = pyfunc.load_pyfunc(model_uri=model_path)

    local_group_pred = grouped_pmdarima.predict_groups(
        groups=groups,
        n_periods=10,
        predict_col="prediction",
        alpha=0.1,
        return_conf_int=True,
        on_error="warn",
    )
    pyfunc_conf = pd.DataFrame(
        {
            "groups": [groups],
            "n_periods": 10,
            "predict_col": "prediction",
            "alpha": 0.1,
            "return_conf_int": True,
            "on_error": "warn",
        },
        index=[0],
    )
    pyfunc_group_predict = loaded_pyfunc_model.predict(pyfunc_conf)

    pd.testing.assert_frame_equal(local_group_pred, pyfunc_group_predict)


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_diviner_signature_and_examples_saved_correctly(
    grouped_prophet, diviner_data, model_path, use_signature, use_example
):
    prediction = grouped_prophet.forecast(horizon=20, frequency="D")
    signature = infer_signature(diviner_data.df, prediction) if use_signature else None
    example = diviner_data.df[0:5].copy(deep=False) if use_example else None
    mlflow.diviner.save_model(
        grouped_prophet, path=model_path, signature=signature, input_example=example
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path).copy(deep=False)
        # NB: datetime values are implicitly cast, so this needs to be reverted.
        r_example["ds"] = pd.to_datetime(r_example["ds"], format=DS_FORMAT)
        np.testing.assert_array_equal(r_example, example)


def test_diviner_load_from_remote_uri_succeeds(grouped_pmdarima, model_path, mock_s3_bucket):
    mlflow.diviner.save_model(diviner_model=grouped_pmdarima, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    # NB: cloudpathlib would need to be used here to handle object store uri
    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_model = mlflow.diviner.load_model(model_uri=model_uri)

    pd.testing.assert_frame_equal(grouped_pmdarima.predict(10), reloaded_model.predict(10))


@pytest.mark.parametrize("should_start_run", [True, False])
def test_diviner_log_model(grouped_prophet, tmp_path, should_start_run):
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "diviner"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["diviner"])
        model_info = mlflow.diviner.log_model(
            diviner_model=grouped_prophet,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow.diviner.load_model(model_uri=model_uri)
        pd.testing.assert_frame_equal(
            grouped_prophet.forecast(horizon=10, frequency="D"),
            reloaded_model.forecast(horizon=10, frequency="D"),
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
        assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
        env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]["conda"]
        assert model_path.joinpath(env_path).exists()
    finally:
        mlflow.end_run()


def test_diviner_log_model_calls_register_model(grouped_pmdarima, tmp_path):
    artifact_path = "diviner"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["diviner"])
        mlflow.diviner.log_model(
            diviner_model=grouped_pmdarima,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="DivinerModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri, "DivinerModel", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_diviner_log_model_no_registered_model_name(grouped_prophet, tmp_path):
    artifact_path = "diviner"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["diviner"])
        mlflow.diviner.log_model(
            diviner_model=grouped_prophet, artifact_path=artifact_path, conda_env=str(conda_env)
        )
        mlflow.register_model.assert_not_called()


def test_diviner_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    grouped_prophet, model_path, diviner_custom_env
):
    mlflow.diviner.save_model(
        diviner_model=grouped_prophet, path=model_path, conda_env=str(diviner_custom_env)
    )
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = model_path.joinpath(pyfunc_conf[pyfunc.ENV]["conda"])

    assert saved_conda_env_path.exists()
    assert not diviner_custom_env.samefile(saved_conda_env_path)

    diviner_custom_env_parsed = yaml.safe_load(diviner_custom_env.read_bytes())
    saved_conda_env_parsed = yaml.safe_load(saved_conda_env_path.read_bytes())
    assert saved_conda_env_parsed == diviner_custom_env_parsed


def test_diviner_model_save_persists_requirements_in_mlflow_model_directory(
    grouped_pmdarima, model_path, diviner_custom_env
):
    mlflow.diviner.save_model(
        diviner_model=grouped_pmdarima, path=model_path, conda_env=str(diviner_custom_env)
    )
    saved_pip_req_path = model_path.joinpath("requirements.txt")
    _compare_conda_env_requirements(diviner_custom_env, str(saved_pip_req_path))


def test_diviner_log_model_with_pip_requirements(grouped_prophet, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.diviner.log_model(grouped_prophet, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a"], strict=True
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.diviner.log_model(grouped_prophet, "model", pip_requirements=[f"-r {req_file}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.diviner.log_model(grouped_prophet, "model", pip_requirements=[f"-c {req_file}", "b"])
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            [expected_mlflow_version, "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_diviner_log_model_with_extra_pip_requirements(grouped_pmdarima, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    default_reqs = mlflow.diviner.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        mlflow.diviner.log_model(grouped_pmdarima, "model", extra_pip_requirements=str(req_file))
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a"]
        )

    # List of requirements
    with mlflow.start_run():
        mlflow.diviner.log_model(
            grouped_pmdarima, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), [expected_mlflow_version, *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.diviner.log_model(
            grouped_pmdarima, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_uri=mlflow.get_artifact_uri("model"),
            requirements=[expected_mlflow_version, *default_reqs, "b", "-c constraints.txt"],
            constraints=["a"],
            strict=False,
        )


def test_diviner_model_save_without_conda_env_uses_default_env_with_expected_dependencies(
    grouped_prophet, model_path
):
    mlflow.diviner.save_model(grouped_prophet, model_path)
    _assert_pip_requirements(model_path, mlflow.diviner.get_default_pip_requirements())


def test_diviner_model_log_without_conda_env_uses_default_env_with_expected_dependencies(
    grouped_pmdarima,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.diviner.log_model(grouped_pmdarima, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.diviner.get_default_pip_requirements())


def test_pmdarima_pyfunc_serve_and_score(grouped_prophet):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.diviner.log_model(
            grouped_prophet,
            artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    local_predict = grouped_prophet.forecast(horizon=10, frequency="W")

    inference_data = pd.DataFrame({"horizon": 10, "frequency": "W"}, index=[0])

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = pd.DataFrame(data=json.loads(resp.content.decode("utf-8"))["predictions"])
    scores["ds"] = pd.to_datetime(scores["ds"], format=DS_FORMAT)
    scores["multiplicative_terms"] = scores["multiplicative_terms"].astype("float64")
    pd.testing.assert_frame_equal(local_predict, scores)


def test_pmdarima_pyfunc_serve_and_score_groups(grouped_prophet, diviner_data):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.diviner.log_model(
            grouped_prophet,
            artifact_path,
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    groups = []
    for i in [0, -1]:
        key_entries = []
        for value in diviner_data.df[diviner_data.key_columns].iloc[[i]].to_dict().values():
            key_entries.append(list(value.values())[0])
        groups.append(tuple(key_entries))

    local_predict = grouped_prophet.predict_groups(groups=groups, horizon=10, frequency="W")

    inference_data = pd.DataFrame({"groups": [groups], "horizon": 10, "frequency": "W"}, index=[0])

    from mlflow.deployments import PredictionsResponse

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
        extra_args=EXTRA_PYFUNC_SERVING_TEST_ARGS,
    )
    scores = PredictionsResponse.from_json(resp.content.decode("utf-8")).get_predictions()
    scores["ds"] = pd.to_datetime(scores["ds"], format=DS_FORMAT)
    scores["multiplicative_terms"] = scores["multiplicative_terms"].astype("float64")
    pd.testing.assert_frame_equal(local_predict, scores)


def test_log_model_with_code_paths(grouped_pmdarima):
    artifact_path = "model"
    with mlflow.start_run(), mock.patch(
        "mlflow.diviner._add_code_from_conf_to_system_path"
    ) as add_mock:
        mlflow.diviner.log_model(grouped_pmdarima, artifact_path, code_paths=[__file__])
        model_uri = mlflow.get_artifact_uri(artifact_path)
        _compare_logged_code_paths(__file__, model_uri, mlflow.diviner.FLAVOR_NAME)
        mlflow.diviner.load_model(model_uri)
        add_mock.assert_called()


def test_virtualenv_subfield_points_to_correct_path(grouped_pmdarima, model_path):
    mlflow.diviner.save_model(grouped_pmdarima, path=model_path)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    python_env_path = Path(model_path, pyfunc_conf[pyfunc.ENV]["virtualenv"])
    assert python_env_path.exists()
    assert python_env_path.is_file()


def test_model_save_load_with_metadata(grouped_pmdarima, model_path):
    mlflow.pmdarima.save_model(
        grouped_pmdarima, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(grouped_pmdarima):
    artifact_path = "model"

    with mlflow.start_run():
        mlflow.pmdarima.log_model(
            grouped_pmdarima,
            artifact_path=artifact_path,
            metadata={"metadata_key": "metadata_value"},
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_diviner_model_fit_with_spark_cannot_be_loaded_as_pyfunc(grouped_prophet, model_path):
    mlflow.diviner.save_model(grouped_prophet, model_path)

    diviner_model_info_path = model_path.joinpath(MLMODEL_FILE_NAME)
    diviner_model_info = yaml.safe_load(diviner_model_info_path.read_text())

    # We can't actually test this by saving in Spark due to method unavailability in OSS Diviner.
    diviner_model_info["flavors"]["diviner"]["fit_with_spark"] = True

    diviner_model_info_path.write_text(yaml.safe_dump(diviner_model_info))

    with pytest.raises(MlflowException, match="The model being loaded was fit in Spark. Diviner"):
        pyfunc.load_model(model_uri=model_path)


@pytest.mark.parametrize("path", ["dbfs:/model", "file/storage", "Users/model/save"])
def test_diviner_model_fit_with_spark_raises_with_invalid_paths(grouped_prophet, path):
    mod_model = deepcopy(grouped_prophet)
    setattr(mod_model, "_fit_with_spark", True)
    with pytest.raises(MlflowException, match="The save path provided must be a relative"):
        mlflow.diviner._save_diviner_model(mod_model, Path(path))

import os
import pytest
import yaml
import numpy as np
import pandas as pd
from collections import namedtuple
from datetime import datetime, timedelta, date
from unittest import mock

from prophet import Prophet

import mlflow
import mlflow.prophet
import mlflow.utils
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.utils import _read_example
from mlflow.models import infer_signature, Model
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import (
    _compare_conda_env_requirements,
    _assert_pip_requirements,
    pyfunc_serve_and_score_model,
)


class DataGeneration:
    def __init__(self, **kwargs):
        self.shift = kwargs["shift"]
        self.start = datetime.strptime(kwargs["start"], "%Y-%M-%d")
        self.size = kwargs["size"]
        self.date_field = kwargs["date_field"]
        self.target_field = kwargs["target_field"]
        self.seasonal_period = kwargs["seasonal_period"]
        self.seasonal_freq = kwargs["seasonal_freq"]
        np.random.seed(42)

    def _period_gen(self):
        period = np.sin(np.arange(0, self.seasonal_period, self.seasonal_freq)) * 50 + 50
        return np.tile(
            period, int(np.ceil(self.size / (self.seasonal_period / self.seasonal_freq)))
        )[: self.size]

    def _generate_raw(self):
        base = np.random.lognormal(mean=2.0, sigma=0.92, size=self.size)
        seasonal = [
            np.polyval([-5.0, -1.0], x) for x in np.linspace(start=0, stop=2, num=self.size)
        ]
        series = (
            np.linspace(start=45.0, stop=90.0, num=self.size) + base + seasonal + self._period_gen()
        )
        return series

    def _generate_linear_data(self):
        DataStruct = namedtuple("DataStruct", "dates, series")
        series = self._generate_raw()
        date_ranges = np.arange(
            self.start, self.start + timedelta(days=self.size), timedelta(days=1)
        ).astype(date)
        return DataStruct(date_ranges, series)

    def _generate_shift_data(self):
        DataStruct = namedtuple("DataStruct", "dates, series")
        raw = self._generate_raw()[: int(self.size * 0.6)]
        temperature = np.concatenate((raw, raw / 2.0)).ravel()[: self.size]
        date_ranges = np.arange(
            self.start, self.start + timedelta(days=self.size), timedelta(days=1)
        ).astype(date)
        return DataStruct(date_ranges, temperature)

    def _gen_series(self):
        if self.shift:
            return self._generate_shift_data()
        else:
            return self._generate_linear_data()

    def create_series_df(self):
        gen_data = self._gen_series()
        temporal_df = pd.DataFrame.from_records(gen_data).T
        temporal_df.columns = [self.date_field, self.target_field]
        return temporal_df


TEST_CONFIG = {
    "shift": False,
    "start": "2011-07-25",
    "size": 365 * 4,
    "seasonal_period": 7,
    "seasonal_freq": 0.1,
    "date_field": "ds",
    "target_field": "y",
}
FORECAST_HORIZON = 60
SEED = 98765
HORIZON_FIELD_NAME = "horizon"
TARGET_FIELD_NAME = "yhat"
DS_FORMAT = "%Y-%m-%dT%H:%M:%S"
INFER_FORMAT = "%Y-%m-%d %H:%M:%S"

ModelWithSource = namedtuple("ModelWithSource", ["model", "data"])

pytestmark = pytest.mark.large


@pytest.fixture(scope="session")
def prophet_model():
    np.random.seed(SEED)
    data = DataGeneration(**TEST_CONFIG).create_series_df()
    model = Prophet().fit(data)
    return ModelWithSource(model, data)


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def prophet_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["pystan", "prophet", "pytest"])
    return conda_env


def future_horizon_df(model, horizon):
    return model.make_future_dataframe(periods=horizon)


@pytest.fixture
def generate_forecast(model, horizon):
    return model.predict(model.make_future_dataframe(periods=horizon))[TARGET_FIELD_NAME]


def test_model_native_save_load(prophet_model, model_path):
    model = prophet_model.model
    mlflow.prophet.save_model(pr_model=model, path=model_path)
    loaded_model = mlflow.prophet.load_model(model_uri=model_path)

    np.testing.assert_array_equal(
        generate_forecast(model, FORECAST_HORIZON),
        loaded_model.predict(future_horizon_df(loaded_model, FORECAST_HORIZON))[TARGET_FIELD_NAME],
    )


def test_model_pyfunc_save_load(prophet_model, model_path):
    model = prophet_model.model
    mlflow.prophet.save_model(pr_model=model, path=model_path)
    loaded_pyfunc = pyfunc.load_pyfunc(model_uri=model_path)

    horizon_df = future_horizon_df(model, FORECAST_HORIZON)

    np.testing.assert_array_equal(
        generate_forecast(model, FORECAST_HORIZON),
        loaded_pyfunc.predict(horizon_df)[TARGET_FIELD_NAME],
    )


def test_signature_and_examples_saved_correctly(prophet_model):
    data = prophet_model.data
    model = prophet_model.model
    horizon_df = future_horizon_df(model, FORECAST_HORIZON)
    signature_ = infer_signature(data, model.predict(horizon_df))
    example_ = data[0:5].copy(deep=False)
    example_["y"] = pd.to_numeric(example_["y"])  # cast to appropriate precision
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.prophet.save_model(
                    model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    r_example = _read_example(mlflow_model, path).copy(deep=False)
                    r_example["ds"] = pd.to_datetime(r_example["ds"], format=DS_FORMAT)
                    np.testing.assert_array_equal(r_example, example)


def test_model_load_from_remote_uri_succeeds(prophet_model, model_path, mock_s3_bucket):
    mlflow.prophet.save_model(pr_model=prophet_model.model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_prophet_model = mlflow.prophet.load_model(model_uri=model_uri)
    np.testing.assert_array_equal(
        generate_forecast(prophet_model.model, FORECAST_HORIZON),
        generate_forecast(reloaded_prophet_model, FORECAST_HORIZON),
    )


def test_model_log(prophet_model):
    old_uri = mlflow.get_tracking_uri()
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()
                artifact_path = "prophet"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["pystan", "prophet"])

                mlflow.prophet.log_model(
                    pr_model=prophet_model.model, artifact_path=artifact_path, conda_env=conda_env
                )
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                reloaded_prophet_model = mlflow.prophet.load_model(model_uri=model_uri)

                np.testing.assert_array_equal(
                    generate_forecast(prophet_model.model, FORECAST_HORIZON),
                    generate_forecast(reloaded_prophet_model, FORECAST_HORIZON),
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


def test_log_model_calls_register_model(prophet_model):
    artifact_path = "prophet"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pystan", "prophet"])
        mlflow.prophet.log_model(
            pr_model=prophet_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="ProphetModel1",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri, "ProphetModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(prophet_model):
    artifact_path = "prophet"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["pystan", "prophet"])
        mlflow.prophet.log_model(
            pr_model=prophet_model.model, artifact_path=artifact_path, conda_env=conda_env,
        )
        mlflow.register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    prophet_model, model_path, prophet_custom_env
):
    mlflow.prophet.save_model(
        pr_model=prophet_model.model, path=model_path, conda_env=prophet_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != prophet_custom_env

    with open(prophet_custom_env, "r") as f:
        prophet_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == prophet_custom_env_parsed


def test_model_save_persists_requirements_in_mlflow_model_directory(
    prophet_model, model_path, prophet_custom_env
):
    mlflow.prophet.save_model(
        pr_model=prophet_model.model, path=model_path, conda_env=prophet_custom_env
    )

    saved_pip_req_path = os.path.join(model_path, "requirements.txt")
    _compare_conda_env_requirements(prophet_custom_env, saved_pip_req_path)


def test_log_model_with_pip_requirements(prophet_model, tmpdir):
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.prophet.log_model(prophet_model.model, "model", pip_requirements=req_file.strpath)
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", "a"], strict=True)

    # List of requirements
    with mlflow.start_run():
        mlflow.prophet.log_model(
            prophet_model.model, "model", pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", "a", "b"], strict=True
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.prophet.log_model(
            prophet_model.model, "model", pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"),
            ["mlflow", "b", "-c constraints.txt"],
            ["a"],
            strict=True,
        )


def test_log_model_with_extra_pip_requirements(prophet_model, tmpdir):
    default_reqs = mlflow.prophet.get_default_pip_requirements()

    # Path to a requirements file
    req_file = tmpdir.join("requirements.txt")
    req_file.write("a")
    with mlflow.start_run():
        mlflow.prophet.log_model(
            prophet_model.model, "model", extra_pip_requirements=req_file.strpath
        )
        _assert_pip_requirements(mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a"])

    # List of requirements
    with mlflow.start_run():
        mlflow.prophet.log_model(
            prophet_model.model, "model", extra_pip_requirements=[f"-r {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            mlflow.get_artifact_uri("model"), ["mlflow", *default_reqs, "a", "b"]
        )

    # Constraints file
    with mlflow.start_run():
        mlflow.prophet.log_model(
            prophet_model.model, "model", extra_pip_requirements=[f"-c {req_file.strpath}", "b"]
        )
        _assert_pip_requirements(
            model_uri=mlflow.get_artifact_uri("model"),
            requirements=["mlflow", *default_reqs, "b", "-c constraints.txt"],
            constraints=["a"],
            strict=False,
        )


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    prophet_model, model_path
):
    mlflow.prophet.save_model(prophet_model.model, model_path)
    _assert_pip_requirements(model_path, mlflow.prophet.get_default_pip_requirements())


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    prophet_model,
):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.prophet.log_model(prophet_model.model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    _assert_pip_requirements(model_uri, mlflow.prophet.get_default_pip_requirements())


def test_pyfunc_serve_and_score(prophet_model):

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.prophet.log_model(prophet_model.model, artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
    local_predict = prophet_model.model.predict(
        prophet_model.model.make_future_dataframe(FORECAST_HORIZON)
    )

    # cast to string representation of datetime series, otherwise will default cast to Unix time
    # which Prophet does not support for encoding
    inference_data = (
        prophet_model.model.make_future_dataframe(FORECAST_HORIZON)["ds"]
        .dt.strftime(INFER_FORMAT)
        .to_frame(name="ds")
    )

    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=inference_data,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    )

    scores = pd.read_json(resp.content, orient="records")

    # predictions are deterministic, but yhat_lower, yhat_upper are non-deterministic based on
    # stan build underlying environment. Seed value only works for reproducibility of yhat.
    # see: https://github.com/facebook/prophet/issues/1124
    pd.testing.assert_series_equal(
        left=local_predict["yhat"], right=scores["yhat"], check_dtype=True
    )

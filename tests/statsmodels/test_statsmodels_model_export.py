import sys
import mock
import os
import pickle
import pytest
import yaml
import json

import numpy as np
import pandas as pd
import pandas.testing

import mlflow.statsmodels
import mlflow.utils
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _read_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.models import Model, infer_signature
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import score_model_in_sagemaker_docker_container

import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import namedtuple
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA


ModelWithResults = namedtuple("ModelWithResults", ["model", "inference_dataframe"])


@pytest.fixture(scope="session")
def ols_model():
    # Ordinary Least Squares (OLS)
    np.random.seed(9876789)
    nsamples = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x ** 2))
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=nsamples)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e

    ols = sm.OLS(y, X)
    model = ols.fit()

    return ModelWithResults(model=model, inference_dataframe=X)


@pytest.fixture(scope="session")
def arma_model():
    # Autoregressive Moving Average (ARMA)
    np.random.seed(12345)
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    arparams = np.r_[1, -arparams]
    maparams = np.r_[1, maparams]
    nobs = 250
    y = arma_generate_sample(arparams, maparams, nobs)
    dates = pd.date_range('1980-1-1', freq="M", periods=nobs)
    y = pd.Series(y, index=dates)

    arima = ARIMA(y, order=(2, 0, 2), trend='n')
    model = arima.fit()
    inference_dataframe = pd.DataFrame([['1999-06-30', '2001-05-31']], columns=["start", "end"])

    return ModelWithResults(model=model, inference_dataframe=inference_dataframe)


def _model_autolog(name):
    mlflow.statsmodels.autolog()
    models_dict = {
        "ols_model": ols_model,
        "arma_model": arma_model
    }
    f = models_dict.get(name)
    return f()


def _get_dates_from_df(df):
    start_date = df["start"][0]
    end_date = df["end"][0]
    return start_date, end_date


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture
def statsmodels_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["statsmodels", "pytest"])
    return conda_env


def _test_save_load(statsmodels_model, model_path, *predict_args):

    mlflow.statsmodels.save_model(statsmodels_model=statsmodels_model.model, path=model_path)
    reloaded_model = mlflow.statsmodels.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    np.testing.assert_array_almost_equal(
        statsmodels_model.model.predict(*predict_args),
        reloaded_model.predict(*predict_args),
    )

    np.testing.assert_array_almost_equal(
        reloaded_model.predict(*predict_args),
        reloaded_pyfunc.predict(statsmodels_model.inference_dataframe),
    )


def test_arma_save_load(arma_model, model_path):
    start_date, end_date = _get_dates_from_df(arma_model.inference_dataframe)
    _test_save_load(arma_model, model_path, start_date, end_date)


def test_ols_save_load(ols_model, model_path):
    _test_save_load(ols_model, model_path, ols_model.inference_dataframe)


def test_signature_and_examples_are_saved_correctly(ols_model):
    model = ols_model.model
    X = ols_model.inference_dataframe
    signature_ = infer_signature(X)
    example_ = X[0:3, :]
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.statsmodels.save_model(model, path=path,
                                              signature=signature,
                                              input_example=example)
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    assert all((_read_example(mlflow_model, path) == example).all())


def test_model_load_from_remote_uri_succeeds(arma_model, model_path, mock_s3_bucket):
    mlflow.statsmodels.save_model(statsmodels_model=arma_model.model, path=model_path)

    artifact_root = "s3://{bucket_name}".format(bucket_name=mock_s3_bucket)
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = artifact_root + "/" + artifact_path
    reloaded_model = mlflow.statsmodels.load_model(model_uri=model_uri)
    start_date, end_date = _get_dates_from_df(arma_model.inference_dataframe)
    np.testing.assert_array_almost_equal(
        arma_model.model.predict(start=start_date, end=end_date),
        reloaded_model.predict(start=start_date, end=end_date)
    )


def _test_model_log(statsmodels_model, model_path, *predict_args):
    old_uri = mlflow.get_tracking_uri()
    model = statsmodels_model.model
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "model"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_conda_deps=["statsmodels=0.11.1"])

                mlflow.statsmodels.log_model(
                    statsmodels_model=model,
                    artifact_path=artifact_path,
                    conda_env=conda_env)
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id,
                    artifact_path=artifact_path)

                reloaded_model = mlflow.statsmodels.load_model(model_uri=model_uri)
                np.testing.assert_array_almost_equal(
                    model.predict(*predict_args),
                    reloaded_model.predict(*predict_args)
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


def test_ols_model_log(ols_model, model_path):
    _test_model_log(ols_model, model_path, ols_model.inference_dataframe)


def test_arma_model_log(arma_model, model_path):
    start_date, end_date = _get_dates_from_df(arma_model.inference_dataframe)
    _test_model_log(arma_model, model_path, start_date, end_date)


def test_ols_autolog():
    _model_autolog("ols_model")


def test_arma_autolog():
    _model_autolog("arma_model")


"""
def test_log_model_calls_register_model(statsmodels_results):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["statsmodels"])
        mlflow.statsmodels.log_model(statsmodels_results=statsmodels_results.model,
                                     artifact_path=artifact_path, conda_env=conda_env,
                                     registered_model_name="AdsModel1")
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=mlflow.active_run().info.run_id,
                                                            artifact_path=artifact_path)
        mlflow.register_model.assert_called_once_with(model_uri, "AdsModel1")


def test_log_model_no_registered_model_name(statsmodels_results):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["statsmodels"])
        mlflow.statsmodels.log_model(statsmodels_results=statsmodels_results.model,
                                     artifact_path=artifact_path, conda_env=conda_env)
        mlflow.register_model.assert_not_called()


@pytest.mark.large
def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
        statsmodels_results, model_path, lgb_custom_env):
    mlflow.statsmodels.save_model(
        statsmodels_results=statsmodels_results.model, path=model_path, conda_env=lgb_custom_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != lgb_custom_env

    with open(lgb_custom_env, "r") as f:
        lgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == lgb_custom_env_parsed


@pytest.mark.large
def test_model_save_accepts_conda_env_as_dict(statsmodels_results, model_path):
    conda_env = dict(mlflow.statsmodels.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.statsmodels.save_model(
        statsmodels_results=statsmodels_results.model, path=model_path, conda_env=conda_env)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


@pytest.mark.large
def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
        statsmodels_results, lgb_custom_env):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.statsmodels.log_model(statsmodels_results=statsmodels_results.model,
                                  artifact_path=artifact_path,
                                  conda_env=lgb_custom_env)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path=artifact_path)

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != lgb_custom_env

    with open(lgb_custom_env, "r") as f:
        lgb_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == lgb_custom_env_parsed


@pytest.mark.large
def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        statsmodels_results, model_path):
    mlflow.statsmodels.save_model(statsmodels_results=statsmodels_results.model, path=model_path,
                                  conda_env=None)

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.statsmodels.get_default_conda_env()


@pytest.mark.large
def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
        statsmodels_results):
    artifact_path = "model"
    with mlflow.start_run():
        mlflow.statsmodels.log_model(statsmodels_results=statsmodels_results.model,
                                     artifact_path=artifact_path, conda_env=None)
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path=artifact_path)

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.statsmodels.get_default_conda_env()


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(statsmodels_results, model_path):
    mlflow.statsmodels.save_model(statsmodels_results=statsmodels_results.model, path=model_path,
                                  conda_env=None)
    reloaded_pyfunc = pyfunc.load_pyfunc(model_uri=model_path)

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=statsmodels_results.inference_dataframe,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        flavor=mlflow.pyfunc.FLAVOR_NAME)
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    pandas.testing.assert_frame_equal(
        deployed_model_preds,
        pd.DataFrame(reloaded_pyfunc.predict(statsmodels_results.inference_dataframe)),
        check_dtype=False,
        check_less_precise=6)
"""

import pytest
import numpy as np
import pandas as pd
import mock
import os
import yaml
import json
import pandas.testing

import mlflow.statsmodels
import mlflow.utils
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import pyfunc
from mlflow.models.utils import _read_example
from mlflow.models import Model, infer_signature
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import

from tests.statsmodels.model_fixtures import (
    ols_model,
    arma_model,
    glsar_model,
    gee_model,
    glm_model,
    gls_model,
    recursivels_model,
    rolling_ols_model,
    rolling_wls_model,
    wls_model,
)


# The code in this file has been adapted from the test cases of the lightgbm flavor.


def _get_dates_from_df(df):
    start_date = df["start"][0]
    end_date = df["end"][0]
    return start_date, end_date


@pytest.fixture
def model_path(tmpdir, subdir="model"):
    return os.path.join(str(tmpdir), subdir)


@pytest.fixture
def statsmodels_custom_env(tmpdir):
    conda_env = os.path.join(str(tmpdir), "conda_env.yml")
    _mlflow_conda_env(
        conda_env, additional_conda_deps=["statsmodels"], additional_pip_deps=["pytest"]
    )
    return conda_env


def _test_models_list(tmpdir, func_to_apply):
    from statsmodels.tsa.base.tsa_model import TimeSeriesModel

    fixtures = [
        ols_model,
        arma_model,
        glsar_model,
        gee_model,
        glm_model,
        gls_model,
        recursivels_model,
        rolling_ols_model,
        rolling_wls_model,
        wls_model,
    ]

    for algorithm in fixtures:
        name = algorithm.__name__
        path = model_path(tmpdir, name)
        model = algorithm()
        if isinstance(model.alg, TimeSeriesModel):
            start_date, end_date = _get_dates_from_df(model.inference_dataframe)
            func_to_apply(model, path, start_date, end_date)
        else:
            func_to_apply(model, path, model.inference_dataframe)


def _test_model_save_load(statsmodels_model, model_path, *predict_args):
    mlflow.statsmodels.save_model(statsmodels_model=statsmodels_model.model, path=model_path)
    reloaded_model = mlflow.statsmodels.load_model(model_uri=model_path)
    reloaded_pyfunc = pyfunc.load_model(model_uri=model_path)

    if hasattr(statsmodels_model.model, "predict"):
        np.testing.assert_array_almost_equal(
            statsmodels_model.model.predict(*predict_args), reloaded_model.predict(*predict_args),
        )

        np.testing.assert_array_almost_equal(
            reloaded_model.predict(*predict_args),
            reloaded_pyfunc.predict(statsmodels_model.inference_dataframe),
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
                _mlflow_conda_env(conda_env, additional_conda_deps=["statsmodels"])

                mlflow.statsmodels.log_model(
                    statsmodels_model=model, artifact_path=artifact_path, conda_env=conda_env
                )
                model_uri = "runs:/{run_id}/{artifact_path}".format(
                    run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
                )

                reloaded_model = mlflow.statsmodels.load_model(model_uri=model_uri)
                if hasattr(model, "predict"):
                    np.testing.assert_array_almost_equal(
                        model.predict(*predict_args), reloaded_model.predict(*predict_args)
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


@pytest.mark.large
def test_models_save_load(tmpdir):
    _test_models_list(tmpdir, _test_model_save_load)


@pytest.mark.large
def test_models_log(tmpdir):
    _test_models_list(tmpdir, _test_model_log)


def test_signature_and_examples_are_saved_correctly(ols_model):
    model = ols_model.model
    X = ols_model.inference_dataframe
    signature_ = infer_signature(X)
    example_ = X[0:3, :]

    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.statsmodels.save_model(
                    model, path=path, signature=signature, input_example=example
                )
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
        reloaded_model.predict(start=start_date, end=end_date),
    )


def test_log_model_calls_register_model(ols_model):
    # Adapted from lightgbm tests
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_conda_deps=["statsmodels"])
        mlflow.statsmodels.log_model(
            statsmodels_model=ols_model.model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            registered_model_name="OLSModel1",
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )
        mlflow.register_model.assert_called_once_with(
            model_uri, "OLSModel1", await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS
        )


def test_log_model_no_registered_model_name(ols_model):
    artifact_path = "model"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch, TempDir(chdr=True, remove_on_exit=True) as tmp:
        conda_env = os.path.join(tmp.path(), "conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_conda_deps=["statsmodels"])
        mlflow.statsmodels.log_model(
            statsmodels_model=ols_model.model, artifact_path=artifact_path, conda_env=conda_env
        )
        mlflow.register_model.assert_not_called()


def test_model_save_persists_specified_conda_env_in_mlflow_model_directory(
    ols_model, model_path, statsmodels_custom_env
):

    mlflow.statsmodels.save_model(
        statsmodels_model=ols_model.model, path=model_path, conda_env=statsmodels_custom_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != statsmodels_custom_env

    with open(statsmodels_custom_env, "r") as f:
        statsmodels_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == statsmodels_custom_env_parsed


def test_model_save_accepts_conda_env_as_dict(ols_model, model_path):
    conda_env = dict(mlflow.statsmodels.get_default_conda_env())
    conda_env["dependencies"].append("pytest")
    mlflow.statsmodels.save_model(
        statsmodels_model=ols_model.model, path=model_path, conda_env=conda_env
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)

    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == conda_env


def test_model_log_persists_specified_conda_env_in_mlflow_model_directory(
    ols_model, statsmodels_custom_env
):

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.statsmodels.log_model(
            statsmodels_model=ols_model.model,
            artifact_path=artifact_path,
            conda_env=statsmodels_custom_env,
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    saved_conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    assert os.path.exists(saved_conda_env_path)
    assert saved_conda_env_path != statsmodels_custom_env

    with open(statsmodels_custom_env, "r") as f:
        statsmodels_custom_env_parsed = yaml.safe_load(f)
    with open(saved_conda_env_path, "r") as f:
        saved_conda_env_parsed = yaml.safe_load(f)
    assert saved_conda_env_parsed == statsmodels_custom_env_parsed


def test_model_save_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    ols_model, model_path
):

    mlflow.statsmodels.save_model(
        statsmodels_model=ols_model.model, path=model_path, conda_env=None
    )

    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.statsmodels.get_default_conda_env()


def test_model_log_without_specified_conda_env_uses_default_env_with_expected_dependencies(
    ols_model,
):

    artifact_path = "model"
    with mlflow.start_run():
        mlflow.statsmodels.log_model(
            statsmodels_model=ols_model.model, artifact_path=artifact_path, conda_env=None
        )
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id, artifact_path=artifact_path
        )

    model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    pyfunc_conf = _get_flavor_configuration(model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME)
    conda_env_path = os.path.join(model_path, pyfunc_conf[pyfunc.ENV])
    with open(conda_env_path, "r") as f:
        conda_env = yaml.safe_load(f)

    assert conda_env == mlflow.statsmodels.get_default_conda_env()


@pytest.mark.release
def test_sagemaker_docker_model_scoring_with_default_conda_env(ols_model, model_path):
    mlflow.statsmodels.save_model(
        statsmodels_model=ols_model.model, path=model_path, conda_env=None
    )

    reloaded_pyfunc = pyfunc.load_pyfunc(model_uri=model_path)

    scoring_response = score_model_in_sagemaker_docker_container(
        model_uri=model_path,
        data=ols_model.inference_dataframe,
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        flavor=mlflow.pyfunc.FLAVOR_NAME,
    )
    deployed_model_preds = pd.DataFrame(json.loads(scoring_response.content))

    pandas.testing.assert_frame_equal(
        deployed_model_preds,
        pd.DataFrame(reloaded_pyfunc.predict(ols_model.inference_dataframe)),
        check_dtype=False,
        check_less_precise=6,
    )

import pytest
import numpy as np
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
import mlflow
import mlflow.statsmodels
from tests.statsmodels.model_fixtures import (
    arma_model,
    ols_model,
    failing_logit_model,
    glsar_model,
    gee_model,
    glm_model,
    gls_model,
    recursivels_model,
    rolling_ols_model,
    rolling_wls_model,
    wls_model,
)

from tests.statsmodels.test_statsmodels_model_export import _get_dates_from_df

# The code in this file has been adapted from the test cases of the lightgbm flavor.


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


def test_statsmodels_autolog_ends_auto_created_run():
    mlflow.statsmodels.autolog()
    arma_model()
    assert mlflow.active_run() is None


def test_statsmodels_autolog_persists_manually_created_run():
    mlflow.statsmodels.autolog()
    with mlflow.start_run() as run:
        ols_model()
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


def test_statsmodels_autolog_logs_default_params():
    mlflow.statsmodels.autolog()
    ols_model()
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        "cov_kwds": "None",
        "cov_type": "nonrobust",
        "method": "pinv",
        "use_t": "None",
    }

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    mlflow.end_run()


def test_statsmodels_autolog_logs_specified_params():
    mlflow.statsmodels.autolog()
    ols_model(method="qr")

    expected_params = {"method": "qr"}

    run = get_latest_run()
    params = run.data.params

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    mlflow.end_run()


def test_statsmodels_autolog_works_after_exception():
    mlflow.statsmodels.autolog()
    # We first fit a model known to raise an exception
    pytest.raises(Exception, failing_logit_model)
    # and then fit another one that should go well
    model_with_results = ols_model()

    run = get_latest_run()
    run_id = run.info.run_id
    loaded_model = mlflow.statsmodels.load_model("runs:/{}/model".format(run_id))

    model_predictions = model_with_results.model.predict(model_with_results.inference_dataframe)
    loaded_model_predictions = loaded_model.predict(model_with_results.inference_dataframe)
    np.testing.assert_array_almost_equal(model_predictions, loaded_model_predictions)


@pytest.mark.large
@pytest.mark.parametrize(
    "log_models", [True, False],
)
def test_statsmodels_autolog_respects_log_models_flag(log_models):
    mlflow.statsmodels.autolog(log_models=log_models)
    ols_model()
    run = get_latest_run()
    client = mlflow.tracking.MlflowClient()
    artifact_paths = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]
    assert ("model" in artifact_paths) == log_models


@pytest.mark.large
def test_statsmodels_autolog_loads_model_from_artifact():
    mlflow.statsmodels.autolog()
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
        model_with_results = algorithm()
        run = get_latest_run()
        run_id = run.info.run_id
        loaded_model = mlflow.statsmodels.load_model("runs:/{}/model".format(run_id))

        if hasattr(model_with_results.model, "predict"):

            if isinstance(model_with_results.alg, TimeSeriesModel):
                start_date, end_date = _get_dates_from_df(model_with_results.inference_dataframe)
                model_predictions = model_with_results.model.predict(start_date, end_date)
                loaded_model_predictions = loaded_model.predict(start_date, end_date)
            else:
                model_predictions = model_with_results.model.predict(
                    model_with_results.inference_dataframe
                )
                loaded_model_predictions = loaded_model.predict(
                    model_with_results.inference_dataframe
                )

            np.testing.assert_array_almost_equal(model_predictions, loaded_model_predictions)

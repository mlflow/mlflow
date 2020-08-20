import numpy as np

import mlflow
import mlflow.statsmodels
from tests.statsmodels.test_statsmodels_model_export import ols_model, arma_model

"""
    Test cases concerning autologging a statsmodels model params and metrics.
    All the tests employ either an OLS model or an ARIMA model.
    The code has been adapted from the autolog test cases of the lightgbm flavor.
"""


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


def test_ols_autolog():
    mlflow.statsmodels.autolog()
    ols_model(method="qr")
    mlflow.end_run()


def test_arma_autolog():
    mlflow.statsmodels.autolog()
    arma_model()
    mlflow.end_run()


def test_statsmodels_autolog_ends_auto_created_run():
    mlflow.statsmodels.autolog()
    arma_model()
    assert mlflow.active_run() is None
    mlflow.end_run()


def test_statsmodels_autolog_persists_manually_created_run():
    mlflow.statsmodels.autolog()
    with mlflow.start_run() as run:
        ols_model()
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id
    mlflow.end_run()


def test_statsmodels_autolog_logs_default_params():
    mlflow.statsmodels.autolog()
    ols_model()
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        'cov_kwds': 'None',
        'cov_type': 'nonrobust',
        'method': 'pinv',
        'use_t': 'None'
    }

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    mlflow.end_run()


def test_statsmodels_autolog_logs_specified_params():
    mlflow.statsmodels.autolog()
    ols_model(method="qr")

    expected_params = {
        'method': 'qr'
    }

    run = get_latest_run()
    params = run.data.params

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    mlflow.end_run()


def test_statsmodels_autolog_loads_model_from_artifact():
    mlflow.statsmodels.autolog()
    model_with_results = ols_model()
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.statsmodels.load_model('runs:/{}/model'.format(run_id))
    np.testing.assert_array_almost_equal(
        model_with_results.model.predict(model_with_results.inference_dataframe),
        loaded_model.predict(model_with_results.inference_dataframe)
    )

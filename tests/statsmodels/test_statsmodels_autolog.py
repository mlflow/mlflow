import pytest
from unittest import mock
import numpy as np
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
import mlflow
from mlflow import MlflowClient
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
    client = MlflowClient()
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


def test_statsmodels_autolog_logs_summary_artifact():
    mlflow.statsmodels.autolog()
    with mlflow.start_run():
        model = ols_model().model
        summary_path = mlflow.get_artifact_uri("model_summary.txt").replace("file://", "")
        with open(summary_path, "r") as f:
            saved_summary = f.read()

    # don't compare the whole summary text because it includes a "Time" field which may change.
    assert model.summary().as_text().split("\n")[:4] == saved_summary.split("\n")[:4]


def test_statsmodels_autolog_emit_warning_when_model_is_large():
    mlflow.statsmodels.autolog()

    with mock.patch(
        "mlflow.statsmodels._model_size_threshold_for_emitting_warning", float("inf")
    ), mock.patch("mlflow.statsmodels._logger.warning") as mock_warning:
        ols_model()
        assert all(
            not call_args[0][0].startswith("The fitted model is larger than")
            for call_args in mock_warning.call_args_list
        )

    with mock.patch("mlflow.statsmodels._model_size_threshold_for_emitting_warning", 1), mock.patch(
        "mlflow.statsmodels._logger.warning"
    ) as mock_warning:
        ols_model()
        assert any(
            call_args[0][0].startswith("The fitted model is larger than")
            for call_args in mock_warning.call_args_list
        )


def test_statsmodels_autolog_logs_basic_metrics():
    mlflow.statsmodels.autolog()
    ols_model()
    run = get_latest_run()
    metrics = run.data.metrics
    assert set(metrics.keys()) == set(mlflow.statsmodels._autolog_metric_allowlist)


def test_statsmodels_autolog_failed_metrics_warning():
    mlflow.statsmodels.autolog()

    @property
    def metric_raise_error(_):
        raise RuntimeError()

    class MockSummary:
        def as_text(self):
            return "mock summary."

    with mock.patch(
        "statsmodels.regression.linear_model.OLSResults.f_pvalue", metric_raise_error
    ), mock.patch(
        "statsmodels.regression.linear_model.OLSResults.fvalue", metric_raise_error
    ), mock.patch(
        # Prevent `OLSResults.summary` from calling `fvalue` and `f_pvalue` that raise an exception
        "statsmodels.regression.linear_model.OLSResults.summary",
        return_value=MockSummary(),
    ), mock.patch(
        "mlflow.statsmodels._logger.warning"
    ) as mock_warning:
        ols_model()
        mock_warning.assert_called_once_with("Failed to autolog metrics: f_pvalue, fvalue.")


def test_statsmodels_autolog_works_after_exception():
    mlflow.statsmodels.autolog()
    # We first fit a model known to raise an exception
    with pytest.raises(Exception, match=r".+"):
        failing_logit_model()
    # and then fit another one that should go well
    model_with_results = ols_model()

    run = get_latest_run()
    run_id = run.info.run_id
    loaded_model = mlflow.statsmodels.load_model("runs:/{}/model".format(run_id))

    model_predictions = model_with_results.model.predict(model_with_results.inference_dataframe)
    loaded_model_predictions = loaded_model.predict(model_with_results.inference_dataframe)
    np.testing.assert_array_almost_equal(model_predictions, loaded_model_predictions)


@pytest.mark.parametrize("log_models", [True, False])
def test_statsmodels_autolog_respects_log_models_flag(log_models):
    mlflow.statsmodels.autolog(log_models=log_models)
    ols_model()
    run = get_latest_run()
    client = MlflowClient()
    artifact_paths = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]
    assert ("model" in artifact_paths) == log_models


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


def test_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.statsmodels.autolog(registered_model_name=registered_model_name)
    with mlflow.start_run():
        ols_model()

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name

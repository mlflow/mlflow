import pytest
import numpy as np
from tests.conftest import tracking_uri_mock  # pylint: disable=unused-import

import pandas as pd
import sklearn.datasets as datasets
from fastai.tabular import tabular_learner, TabularList
from fastai.metrics import accuracy
import mlflow
import mlflow.fastai
from fastai.callbacks import EarlyStoppingCallback
from mlflow.utils.autologging_utils import BatchMetricsLogger
from unittest.mock import patch

np.random.seed(1337)

NUM_EPOCHS = 3
MIN_DELTA = 99999999  # Forces earlystopping


@pytest.fixture(params=[True, False])
def manual_run(request):
    if request.param:
        mlflow.start_run()
    yield
    mlflow.end_run()


def iris_dataframe():
    iris = datasets.load_iris()
    return pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])


@pytest.fixture(scope="session")
def iris_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target, name="label")
    return (
        TabularList.from_df(pd.concat([X, y], axis=1), cont_names=list(X.columns))
        .split_by_rand_pct(valid_pct=0.1, seed=42)
        .label_from_df(cols="label")
        .databunch()
    )


def fastai_model(data, **kwargs):
    return tabular_learner(data, metrics=accuracy, layers=[5, 3, 2], **kwargs)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_ends_auto_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()
    model = fastai_model(iris_data)
    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(1)
    else:
        model.fit(1)
    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_persists_manually_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()

    with mlflow.start_run() as run:
        model = fastai_model(iris_data)

        if fit_variant == "fit_one_cycle":
            model.fit_one_cycle(NUM_EPOCHS)
        else:
            model.fit(NUM_EPOCHS)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def fastai_random_data_run(iris_data, fit_variant, manual_run):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_model(iris_data)

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    else:
        model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_logs_expected_data(fastai_random_data_run, fit_variant):
    # pylint: disable=unused-argument
    model, run = fastai_random_data_run
    data = run.data

    # Testing metrics are logged
    assert "train_loss" in data.metrics
    assert "valid_loss" in data.metrics

    for o in model.metrics:
        assert o.__name__ in data.metrics

    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    assert np.array_equal([m.value for m in metric_history], model.recorder.val_losses)

    # Testing explicitly passed parameters are logged correctly
    assert "epochs" in data.params
    assert data.params["epochs"] == str(NUM_EPOCHS)

    # Testing implicitly passed parameters are logged correctly
    assert "wd" in data.params

    # Testing unwanted parameters are not logged
    assert "callbacks" not in data.params
    assert "learn" not in data.params

    # Testing optimizer parameters are logged
    assert "opt_func" in data.params
    assert data.params["opt_func"] == "Adam"
    assert "model_summary" in data.tags

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_fastai_autolog_log_models_configuration(log_models):
    mlflow.fastai.autolog(log_models=log_models)
    model = fastai_model(iris_data())
    model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert ("model" in artifacts) == log_models


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_logs_default_params(fastai_random_data_run, fit_variant):
    _, run = fastai_random_data_run
    if fit_variant == "fit":
        assert "lr" in run.data.params
        assert run.data.params["lr"] == "slice(None, 0.003, None)"
    else:
        assert "pct_start" in run.data.params
        assert run.data.params["pct_start"] == "0.3"


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_model_can_load_from_artifact(fastai_random_data_run):
    run_id = fastai_random_data_run[1].info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    model = mlflow.fastai.load_model("runs:/" + run_id + "/model")
    model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
    model_wrapper.predict(iris_dataframe())


@pytest.fixture
def fastai_random_data_run_with_callback(iris_data, fit_variant, manual_run, callback, patience):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_model(iris_data)

    callbacks = []
    if callback == "early":
        callback = EarlyStoppingCallback(learn=model, patience=patience, min_delta=MIN_DELTA)
        callbacks.append(callback)

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS, callbacks=callbacks)
    else:
        model.fit(NUM_EPOCHS, callbacks=callbacks)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [0, 1, 5])
def test_fastai_autolog_early_stop_logs(fastai_random_data_run_with_callback, patience):
    model, run = fastai_random_data_run_with_callback
    params = run.data.params
    assert "early_stop_patience" in params
    assert params["early_stop_patience"] == str(patience)
    assert "early_stop_monitor" in params
    assert params["early_stop_monitor"] == "valid_loss"
    assert "early_stop_mode" in params
    assert params["early_stop_mode"] == "auto"
    assert "early_stop_min_delta" in params
    assert params["early_stop_min_delta"] == "-{}".format(MIN_DELTA)

    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    num_of_epochs = len(model.recorder.val_losses)

    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [11])
def test_fastai_autolog_early_stop_no_stop_does_not_log(
    fastai_random_data_run_with_callback, patience
):
    model, run, = fastai_random_data_run_with_callback
    params = run.data.params
    assert "early_stop_patience" in params
    assert params["early_stop_patience"] == str(patience)
    assert "early_stop_monitor" in params
    assert params["early_stop_monitor"] == "valid_loss"
    assert "early_stop_mode" in params
    assert "early_stop_min_delta" in params
    assert params["early_stop_min_delta"] == "-{}".format(99999999)

    num_of_epochs = len(model.recorder.val_losses)
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == NUM_EPOCHS
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
def test_fastai_autolog_non_early_stop_callback_does_not_log(fastai_random_data_run_with_callback):
    model, run, = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "early_stop_patience" not in params
    assert "early_stop_monitor" not in params
    assert "early_stop_mode" not in params
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "early_stop_min_delta" not in params
    num_of_epochs = len(model.recorder.val_losses)
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == NUM_EPOCHS
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
def test_fastai_autolog_batch_metrics_logger_logs_expected_metrics(fit_variant, callback, patience):
    patched_metrics_data = []

    # Mock patching BatchMetricsLogger.record_metrics()
    # to ensure that expected metrics are being logged.
    original = BatchMetricsLogger.record_metrics

    with patch(
        "mlflow.utils.autologging_utils.BatchMetricsLogger.record_metrics", autospec=True
    ) as record_metrics_mock:

        def record_metrics_side_effect(self, metrics, step=None):
            patched_metrics_data.extend(metrics.items())
            original(self, metrics, step)

        record_metrics_mock.side_effect = record_metrics_side_effect
        _, run = fastai_random_data_run_with_callback(
            iris_data(), fit_variant, manual_run, callback, patience
        )

    patched_metrics_data = dict(patched_metrics_data)
    original_metrics = run.data.metrics
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data
        assert original_metrics[metric_name] == patched_metrics_data[metric_name]

    assert "train_loss" in original_metrics
    assert "train_loss" in patched_metrics_data

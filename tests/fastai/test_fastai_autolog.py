import pytest
import numpy as np
from tests.conftest import tracking_uri_mock  # pylint: disable=unused-import

import pandas as pd
import sklearn.datasets as datasets
from fastai.tabular.all import tabular_learner, TabularDataLoaders
from fastai.vision.all import ImageDataLoaders, cnn_learner
from fastai.vision import models
from fastai.data.external import untar_data, URLs
from fastai.metrics import accuracy
import mlflow
import mlflow.fastai
from fastai.callback.all import EarlyStoppingCallback, SaveModelCallback
from mlflow.utils.autologging_utils import BatchMetricsLogger
from unittest.mock import patch

import torch
from functools import partial
from fastai.optimizer import OptimWrapper

import matplotlib as mpl

mpl.use("Agg")

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
    return TabularDataLoaders.from_df(
        df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
    )


def fastai_tabular_model(data, **kwargs):
    return tabular_learner(data, metrics=accuracy, layers=[5, 3, 2], **kwargs)


def mnist_path():
    mnist = untar_data(URLs.MNIST_TINY)
    return mnist


@pytest.fixture(scope="session")
def mnist_data():
    mnist = untar_data(URLs.MNIST_TINY)
    return ImageDataLoaders.from_folder(mnist, num_workers=0)


def fastai_visual_model(data, **kwargs):
    return cnn_learner(data, models.resnet18, normalize=False)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_ends_auto_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()
    model = fastai_tabular_model(iris_data)
    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(1)
    elif fit_variant == "fine_tune":
        model.fine_tune(1, freeze_epochs=1)
    else:
        model.fit(1)
    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_persists_manually_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()

    with mlflow.start_run() as run:
        model = fastai_tabular_model(iris_data)

        if fit_variant == "fit_one_cycle":
            model.fit_one_cycle(NUM_EPOCHS)
        elif fit_variant == "fine_tune":
            model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
        else:
            model.fit(NUM_EPOCHS)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def fastai_random_tabular_data_run(iris_data, fit_variant, manual_run):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_tabular_model(iris_data)

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.fixture
def fastai_random_visual_data_run(mnist_data, fit_variant, manual_run):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_visual_model(mnist_data)

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle", "fine_tune"])
def test_fastai_autolog_logs_expected_data(fastai_random_visual_data_run, fit_variant):
    # pylint: disable=unused-argument
    model, run = fastai_random_visual_data_run
    data = run.data

    # Testing metrics are logged
    assert "train_loss" in data.metrics
    assert "valid_loss" in data.metrics

    for o in model.metrics:
        assert o.name in data.metrics

    # Testing explicitly passed parameters are logged correctly
    if fit_variant != "fine_tune":
        assert "n_epoch" in data.params
        assert data.params["n_epoch"] == str(NUM_EPOCHS)
    else:
        assert "epochs" in data.params
        assert data.params["epochs"] == str(NUM_EPOCHS - 1)

    # Testing unwanted parameters are not logged
    assert "cbs" not in data.params
    assert "callbacks" not in data.params
    assert "learn" not in data.params

    # Testing optimizer parameters are logged
    assert "opt_func" in data.params
    assert data.params["opt_func"] == "Adam"
    assert "wd" in data.params
    assert "sqr_mom" in data.params
    if fit_variant == "fit_one_cycle":
        for param in ["lr", "mom"]:
            for stat in ["min", "max", "init", "final"]:
                assert param + "_" + stat in data.params
    elif fit_variant == "fine_tune":
        freeze_prefix = "freeze_"
        assert freeze_prefix + "wd" in data.params
        assert freeze_prefix + "sqr_mom" in data.params
        assert freeze_prefix + "epochs" in data.params
        assert data.params[freeze_prefix + "epochs"] == str(1)
        for prefix in [freeze_prefix, ""]:
            for param in ["lr", "mom"]:
                for stat in ["min", "max", "init", "final"]:
                    assert prefix + param + "_" + stat in data.params
    else:
        assert "lr" in data.params
        assert "mom" in data.params

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "module_summary.txt" in artifacts


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle", "fine_tune"])
def test_fastai_autolog_opt_func_expected_data(mnist_data, fit_variant, manual_run):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = cnn_learner(
        mnist_data,
        models.resnet18,
        normalize=False,
        opt_func=partial(OptimWrapper, opt=torch.optim.Adam),
    )

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    data = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id).data

    assert "opt_func" in data.params
    assert data.params["opt_func"] == "Adam"

    if fit_variant == "fine_tune":
        freeze_prefix = "freeze_"
        assert freeze_prefix + "opt_func" in data.params
        assert data.params[freeze_prefix + "opt_func"] == "Adam"


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_fastai_autolog_log_models_configuration(log_models):
    mlflow.fastai.autolog(log_models=log_models)
    model = fastai_tabular_model(iris_data())
    model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert ("model" in artifacts) == log_models


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle", "fine_tune"])
def test_fastai_autolog_logs_default_params(fastai_random_visual_data_run, fit_variant):
    _, _ = fastai_random_visual_data_run
    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    if fit_variant == "fit_one_cycle":
        for param in ["lr", "mom"]:
            assert any([a.startswith(param + ".") for a in artifacts])
    elif fit_variant == "fine_tune":
        freeze_prefix = "freeze_"
        for prefix in [freeze_prefix, ""]:
            for param in ["lr", "mom"]:
                assert any([a.startswith(freeze_prefix + param + ".") for a in artifacts])


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_model_can_load_from_artifact(fastai_random_tabular_data_run):
    run_id = fastai_random_tabular_data_run[1].info.run_id
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

    model = fastai_tabular_model(iris_data)

    if callback == "early":
        cb = EarlyStoppingCallback(patience=patience, min_delta=MIN_DELTA)
        model.add_cb(cb)
    elif callback == "save_and_early_stop":
        early_cb = EarlyStoppingCallback(patience=patience, min_delta=MIN_DELTA)
        save_cb = SaveModelCallback(min_delta=MIN_DELTA)
        model.add_cbs([save_cb, early_cb])

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["save_and_early_stop"])
@pytest.mark.parametrize("patience", [0, 1, 5])
def test_fastai_autolog_save_and_early_stop_logs(fastai_random_data_run_with_callback, patience):
    model, run = fastai_random_data_run_with_callback

    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    num_of_epochs = len(model.recorder.values)

    assert len(metric_history) == num_of_epochs

    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=run.info.run_id, artifact_path="model"
    )

    model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
    reloaded_model = mlflow.fastai.load_model(model_uri=model_uri)
    reloaded_model_wrapper = mlflow.fastai._FastaiModelWrapper(reloaded_model)

    model_result = model_wrapper.predict(iris_dataframe())
    reloaded_result = reloaded_model_wrapper.predict(iris_dataframe())

    np.testing.assert_array_almost_equal(model_result, reloaded_result)


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
    assert "early_stop_comp" in params
    assert params["early_stop_comp"] == "less"
    assert "early_stop_min_delta" in params
    assert params["early_stop_min_delta"] == "-{}".format(MIN_DELTA)

    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    num_of_epochs = len(model.recorder.values)

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
    assert "early_stop_comp" in params
    assert "early_stop_min_delta" in params
    assert params["early_stop_min_delta"] == "-{}".format(MIN_DELTA)

    num_of_epochs = len(model.recorder.values)
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
    assert "early_stop_comp" not in params
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "early_stop_min_delta" not in params
    num_of_epochs = len(model.recorder.values)
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

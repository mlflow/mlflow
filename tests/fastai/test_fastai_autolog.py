import pickle
from functools import partial
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import datasets
from torch import nn, optim
from fastai.learner import Learner
from fastai.optimizer import OptimWrapper
from fastai.tabular.all import TabularDataLoaders
from fastai.callback.all import EarlyStoppingCallback, SaveModelCallback

import mlflow
import mlflow.fastai
from mlflow import MlflowClient
from mlflow.fastai.callback import __MlflowFastaiCallback
from mlflow.utils.autologging_utils import BatchMetricsLogger
from tests.conftest import tracking_uri_mock  # pylint: disable=unused-import

mpl.use("Agg")

np.random.seed(1337)

NUM_EPOCHS = 3
MIN_DELTA = 99999999  # Forces earlystopping


def iris_dataframe():
    iris = datasets.load_iris()
    return pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])


@pytest.fixture(scope="module")
def iris_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target, name="label", dtype=np.float32)
    return TabularDataLoaders.from_df(
        df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 1)

    def forward(self, _, x_cont):
        x = self.linear1(x_cont)
        return self.linear2(x)


def splitter(model):
    """
    Splits model parameters into multiple groups to allow fine-tuning
    """
    params = list(model.parameters())
    return [
        # weights and biases of the first linear layer
        params[:2],
        # weights and biases of the second linear layer
        params[2:],
    ]


def fastai_tabular_model(data, **kwargs):
    # Create a fine-tunable learner
    return Learner(data, Model(), loss_func=nn.MSELoss(), splitter=splitter, **kwargs)


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
def fastai_random_tabular_data_run(iris_data, fit_variant):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_tabular_model(iris_data)

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle", "fine_tune"])
def test_fastai_autolog_logs_expected_data(fastai_random_tabular_data_run, fit_variant):
    # pylint: disable=unused-argument
    model, run = fastai_random_tabular_data_run
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
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "module_summary.txt" in artifacts


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle", "fine_tune"])
def test_fastai_autolog_opt_func_expected_data(iris_data, fit_variant):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()
    model = fastai_tabular_model(iris_data, opt_func=partial(OptimWrapper, opt=optim.Adam))

    if fit_variant == "fit_one_cycle":
        model.fit_one_cycle(NUM_EPOCHS)
    elif fit_variant == "fine_tune":
        model.fine_tune(NUM_EPOCHS - 1, freeze_epochs=1)
    else:
        model.fit(NUM_EPOCHS)

    client = MlflowClient()
    data = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id).data

    assert "opt_func" in data.params
    assert data.params["opt_func"] == "Adam"

    if fit_variant == "fine_tune":
        freeze_prefix = "freeze_"
        assert freeze_prefix + "opt_func" in data.params
        assert data.params[freeze_prefix + "opt_func"] == "Adam"


@pytest.mark.parametrize("log_models", [True, False])
def test_fastai_autolog_log_models_configuration(log_models, iris_data):
    mlflow.fastai.autolog(log_models=log_models)
    model = fastai_tabular_model(iris_data)
    model.fit(NUM_EPOCHS)

    client = MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert ("model" in artifacts) == log_models


@pytest.mark.parametrize("fit_variant", ["fit_one_cycle", "fine_tune"])
def test_fastai_autolog_logs_default_params(fastai_random_tabular_data_run, fit_variant):
    # pylint: disable=unused-argument
    client = MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    if fit_variant == "fit_one_cycle":
        for param in ["lr", "mom"]:
            assert any(a.startswith(param + ".") for a in artifacts)
    elif fit_variant == "fine_tune":
        freeze_prefix = "freeze_"
        for prefix in [freeze_prefix, ""]:
            for param in ["lr", "mom"]:
                assert any(a.startswith(prefix + param + ".") for a in artifacts)


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
def test_fastai_autolog_model_can_load_from_artifact(fastai_random_tabular_data_run):
    run_id = fastai_random_tabular_data_run[1].info.run_id
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    model = mlflow.fastai.load_model("runs:/" + run_id + "/model")
    model_wrapper = mlflow.fastai._FastaiModelWrapper(model)
    model_wrapper.predict(iris_dataframe())


def get_fastai_random_data_run_with_callback(iris_data, fit_variant, callback, patience, tmpdir):
    # pylint: disable=unused-argument
    mlflow.fastai.autolog()

    model = fastai_tabular_model(iris_data, model_dir=tmpdir)

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

    client = MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.fixture
def fastai_random_data_run_with_callback(iris_data, fit_variant, callback, patience, tmpdir):
    return get_fastai_random_data_run_with_callback(
        iris_data, fit_variant, callback, patience, tmpdir
    )


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["save_and_early_stop"])
@pytest.mark.parametrize("patience", [0, 1, 5])
def test_fastai_autolog_save_and_early_stop_logs(fastai_random_data_run_with_callback):
    model, run = fastai_random_data_run_with_callback

    client = MlflowClient()
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

    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    num_of_epochs = len(model.recorder.values)

    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [11])
def test_fastai_autolog_early_stop_no_stop_does_not_log(
    fastai_random_data_run_with_callback, patience
):
    model, run = fastai_random_data_run_with_callback
    params = run.data.params
    assert "early_stop_patience" in params
    assert params["early_stop_patience"] == str(patience)
    assert "early_stop_monitor" in params
    assert params["early_stop_monitor"] == "valid_loss"
    assert "early_stop_comp" in params
    assert "early_stop_min_delta" in params
    assert params["early_stop_min_delta"] == "-{}".format(MIN_DELTA)

    num_of_epochs = len(model.recorder.values)
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == NUM_EPOCHS
    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
def test_fastai_autolog_non_early_stop_callback_does_not_log(fastai_random_data_run_with_callback):
    model, run = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "early_stop_patience" not in params
    assert "early_stop_monitor" not in params
    assert "early_stop_comp" not in params
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "early_stop_min_delta" not in params
    num_of_epochs = len(model.recorder.values)
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "valid_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == NUM_EPOCHS
    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("fit_variant", ["fit", "fit_one_cycle"])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
def test_fastai_autolog_batch_metrics_logger_logs_expected_metrics(
    fit_variant, callback, patience, tmpdir, iris_data
):
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
        _, run = get_fastai_random_data_run_with_callback(
            iris_data, fit_variant, callback, patience, tmpdir
        )

    patched_metrics_data = dict(patched_metrics_data)
    original_metrics = run.data.metrics
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data
        assert original_metrics[metric_name] == patched_metrics_data[metric_name]

    assert "train_loss" in original_metrics
    assert "train_loss" in patched_metrics_data


def test_callback_is_picklable():
    cb = __MlflowFastaiCallback(
        BatchMetricsLogger(run_id="1234"), log_models=True, is_fine_tune=False
    )
    pickle.dumps(cb)


def test_autolog_registering_model(iris_data):
    registered_model_name = "test_autolog_registered_model"
    mlflow.fastai.autolog(registered_model_name=registered_model_name)
    with mlflow.start_run():
        model = fastai_tabular_model(iris_data)
        model.fit(NUM_EPOCHS)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name

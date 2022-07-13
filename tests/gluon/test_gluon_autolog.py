from packaging.version import Version
import pickle
import random

import mxnet as mx
import numpy as np
import pytest
from mxnet.gluon import Trainer
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.gluon.nn import HybridSequential, Dense

import mlflow
import mlflow.gluon
from mlflow import MlflowClient
from mlflow.gluon._autolog import __MLflowGluonCallback
from mlflow.utils.autologging_utils import BatchMetricsLogger
from unittest.mock import patch
from tests.gluon.utils import is_mxnet_older_than_1_6_0, get_estimator


if Version(mx.__version__) >= Version("2.0.0"):
    array_module = mx.np
else:
    array_module = mx.nd


class LogsDataset(Dataset):
    def __init__(self):
        self.len = 1000

    def __getitem__(self, idx):
        return (
            array_module.array(np.random.rand(1, 32)),
            array_module.full(1, random.randint(0, 10), dtype="float32"),
        )

    def __len__(self):
        return self.len


def get_train_prefix():
    # training prefix was renamed to `training` in mxnet 1.6.0:
    # https://github.com/apache/incubator-mxnet/pull/17048
    return "train" if is_mxnet_older_than_1_6_0() else "training"


def get_gluon_random_data_run(log_models=True):
    mlflow.gluon.autolog(log_models)

    with mlflow.start_run() as run:
        data = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")
        validation = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")

        model = HybridSequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10))
        model.initialize()
        model.hybridize()
        trainer = Trainer(
            model.collect_params(),
            "adam",
            optimizer_params={"learning_rate": 0.001, "epsilon": 1e-07},
        )
        est = get_estimator(model, trainer)

        est.fit(data, epochs=3, val_data=validation)
    client = MlflowClient()
    return client.get_run(run.info.run_id)


@pytest.fixture
def gluon_random_data_run(log_models=True):
    return get_gluon_random_data_run(log_models)


def test_gluon_autolog_logs_expected_data(gluon_random_data_run):
    data = gluon_random_data_run.data
    train_prefix = get_train_prefix()
    assert "{} accuracy".format(train_prefix) in data.metrics
    assert "validation accuracy" in data.metrics

    # In mxnet >= 1.6.0, `Estimator` monitors `loss` only when `train_metrics` is specified.
    #
    # estimator.Estimator(loss=SomeLoss())  # monitors `loss`
    # estimator.Estimator(loss=SomeLoss(), train_metrics=SomeMetric()) # doesn't monitor `loss`
    if is_mxnet_older_than_1_6_0():
        assert "{} softmaxcrossentropyloss".format(train_prefix) in data.metrics
        assert "validation softmaxcrossentropyloss" in data.metrics
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"
    assert "epsilon" in data.params
    assert data.params["epsilon"] == "1e-07"


def test_gluon_autolog_batch_metrics_logger_logs_expected_metrics():
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
        run = get_gluon_random_data_run()

    patched_metrics_data = dict(patched_metrics_data)
    original_metrics = run.data.metrics
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data
        assert original_metrics[metric_name] == patched_metrics_data[metric_name]

    train_prefix = get_train_prefix()
    assert "{} accuracy".format(train_prefix) in original_metrics
    assert "{} accuracy".format(train_prefix) in patched_metrics_data


def test_gluon_autolog_model_can_load_from_artifact(gluon_random_data_run):
    client = MlflowClient()
    artifacts = client.list_artifacts(gluon_random_data_run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert "model" in artifacts
    ctx = mx.cpu()
    model = mlflow.gluon.load_model("runs:/" + gluon_random_data_run.info.run_id + "/model", ctx)
    model(array_module.array(np.random.rand(1000, 1, 32)))


@pytest.mark.parametrize("log_models", [True, False])
def test_gluon_autolog_log_models_configuration(log_models):
    random_data_run = get_gluon_random_data_run(log_models)
    client = MlflowClient()
    artifacts = client.list_artifacts(random_data_run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert ("model" in artifacts) == log_models


def test_autolog_ends_auto_created_run():
    mlflow.gluon.autolog()

    data = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")

    model = HybridSequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))
    model.initialize()
    model.hybridize()

    trainer = Trainer(
        model.collect_params(), "adam", optimizer_params={"learning_rate": 0.001, "epsilon": 1e-07}
    )
    est = get_estimator(model, trainer)

    est.fit(data, epochs=3)

    assert mlflow.active_run() is None


def test_autolog_persists_manually_created_run():
    mlflow.gluon.autolog()

    data = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")

    with mlflow.start_run() as run:

        model = HybridSequential()
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10))
        model.initialize()
        model.hybridize()
        trainer = Trainer(
            model.collect_params(),
            "adam",
            optimizer_params={"learning_rate": 0.001, "epsilon": 1e-07},
        )
        est = get_estimator(model, trainer)

        est.fit(data, epochs=3)

        assert mlflow.active_run().info.run_id == run.info.run_id


def test_callback_is_callable():
    cb = __MLflowGluonCallback(log_models=True, metrics_logger=BatchMetricsLogger(run_id="1234"))
    pickle.dumps(cb)


def test_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.gluon.autolog(registered_model_name=registered_model_name)

    data = DataLoader(LogsDataset(), batch_size=128, last_batch="discard")

    model = HybridSequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10))
    model.initialize()
    model.hybridize()

    trainer = Trainer(
        model.collect_params(), "adam", optimizer_params={"learning_rate": 0.001, "epsilon": 1e-07}
    )
    est = get_estimator(model, trainer)

    with mlflow.start_run():
        est.fit(data, epochs=3)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name

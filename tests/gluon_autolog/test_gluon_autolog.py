from distutils.version import LooseVersion
import random
import warnings

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pytest
from mxnet.gluon import Trainer
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data import Dataset, DataLoader
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense

import mlflow
import mlflow.gluon
from mlflow.utils.autologging_utils import BatchMetricsLogger
from unittest.mock import patch

if LooseVersion(mx.__version__) >= LooseVersion("2.0.0"):
    from mxnet.gluon.metric import Accuracy  # pylint: disable=import-error
else:
    from mxnet.metric import Accuracy  # pylint: disable=import-error


class LogsDataset(Dataset):
    def __init__(self):
        self.len = 1000

    def __getitem__(self, idx):
        return nd.array(np.random.rand(1, 32)), nd.full(1, random.randint(0, 10), dtype="float32")

    def __len__(self):
        return self.len


def is_mxnet_older_than_1_6_0():
    return LooseVersion(mx.__version__) < LooseVersion("1.6.0")


def get_metrics():
    # `metrics` argument was split into `train_metrics` and `val_metrics` in mxnet 1.6.0:
    # https://github.com/apache/incubator-mxnet/pull/17048
    arg_name = "metrics" if is_mxnet_older_than_1_6_0() else "train_metrics"
    return {arg_name: Accuracy()}


def get_train_prefix():
    # training prefix was renamed to `training` in mxnet 1.6.0:
    # https://github.com/apache/incubator-mxnet/pull/17048
    return "train" if is_mxnet_older_than_1_6_0() else "training"


@pytest.fixture
def gluon_random_data_run(log_models=True):
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
        est = estimator.Estimator(
            net=model, loss=SoftmaxCrossEntropyLoss(), trainer=trainer, **get_metrics()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(data, epochs=3, val_data=validation)
    client = mlflow.tracking.MlflowClient()
    return client.get_run(run.info.run_id)


@pytest.mark.large
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


@pytest.mark.large
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
        run = gluon_random_data_run()

    patched_metrics_data = dict(patched_metrics_data)
    original_metrics = run.data.metrics
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data
        assert original_metrics[metric_name] == patched_metrics_data[metric_name]

    train_prefix = get_train_prefix()
    assert "{} accuracy".format(train_prefix) in original_metrics
    assert "{} accuracy".format(train_prefix) in patched_metrics_data


@pytest.mark.large
def test_gluon_autolog_model_can_load_from_artifact(gluon_random_data_run):
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(gluon_random_data_run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert "model" in artifacts
    ctx = mx.cpu()
    model = mlflow.gluon.load_model("runs:/" + gluon_random_data_run.info.run_id + "/model", ctx)
    model(nd.array(np.random.rand(1000, 1, 32)))


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_gluon_autolog_log_models_configuration(log_models):
    random_data_run = gluon_random_data_run(log_models)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(random_data_run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert ("model" in artifacts) == log_models


@pytest.mark.large
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
    est = estimator.Estimator(
        net=model, loss=SoftmaxCrossEntropyLoss(), trainer=trainer, **get_metrics()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(data, epochs=3)

    assert mlflow.active_run() is None


@pytest.mark.large
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
        est = estimator.Estimator(
            net=model, loss=SoftmaxCrossEntropyLoss(), trainer=trainer, **get_metrics()
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(data, epochs=3)

        assert mlflow.active_run().info.run_id == run.info.run_id

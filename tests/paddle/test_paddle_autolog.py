import pytest
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.static import InputSpec
import mlflow
import mlflow.paddle
from mlflow.utils.file_utils import TempDir
from mlflow.paddle._paddle_autolog import _get_optimizer_name

NUM_EPOCHS = 2

pytestmark = pytest.mark.large


@pytest.fixture
def paddle_model():
    device = paddle.set_device("cpu")  # or 'gpu'

    net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(), nn.Linear(200, 10))

    # inputs and labels are not required for dynamic graph.
    input = InputSpec([None, 784], "float32", "x")
    label = InputSpec([None, 1], "int64", "label")

    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())

    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    data = paddle.vision.datasets.MNIST(mode="train", transform=transform)

    mlflow.paddle.autolog()
    model.fit(data, epochs=NUM_EPOCHS, batch_size=32, verbose=1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return model, run


@pytest.mark.parametrize("log_models", [True, False])
def test_paddle_autolog_log_models_configuration(log_models):
    mlflow.paddle.autolog(log_models=log_models)
    device = paddle.set_device("cpu")  # or 'gpu'

    net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(), nn.Linear(200, 10))

    # inputs and labels are not required for dynamic graph.
    input = InputSpec([None, 784], "float32", "x")
    label = InputSpec([None, 1], "int64", "label")

    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())

    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    data = paddle.vision.datasets.MNIST(mode="train", transform=transform)
    model.fit(data, epochs=NUM_EPOCHS, batch_size=32, verbose=1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("model" in artifacts) == log_models


def test_paddle_autolog_logs_default_params(paddle_model):
    _, run = paddle_model
    data = run.data
    assert "optimizer_name" in data.params


def test_paddle_autolog_logs_expected_data(paddle_model):
    _, run = paddle_model
    data = run.data

    # Checking if metrics are logged
    client = mlflow.tracking.MlflowClient()
    for metric_key in ["acc", "batch_size", "loss", "step"]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "SGD"

    # Testing learning rate are logged
    assert "learning_rate" in data.params
    assert float(data.params["learning_rate"]) == 1e-3

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


def test_paddle_autolog_persists_manually_created_run():
    with mlflow.start_run() as manual_run:
        device = paddle.set_device("cpu")  # or 'gpu'

        net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(), nn.Linear(200, 10))

        # inputs and labels are not required for dynamic graph.
        input = InputSpec([None, 784], "float32", "x")
        label = InputSpec([None, 1], "int64", "label")

        model = paddle.Model(net, input, label)
        optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
        model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())

        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        data = paddle.vision.datasets.MNIST(mode="train", transform=transform)

        mlflow.paddle.autolog()
        model.fit(data, epochs=NUM_EPOCHS, batch_size=32, verbose=1)
        assert mlflow.active_run() is not None
        assert mlflow.active_run().info.run_id == manual_run.info.run_id


def test_paddle_autolog_ends_auto_created_run(paddle_model):
    assert mlflow.active_run() is None


@pytest.fixture
def paddle_model_with_early_stopping(patience):
    device = paddle.set_device("cpu")

    net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(), nn.Linear(200, 10))

    # inputs and labels are not required for dynamic graph.
    input = InputSpec([None, 784], "float32", "x")
    label = InputSpec([None, 1], "int64", "label")

    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())

    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    data = paddle.vision.datasets.MNIST(mode="train", transform=transform)

    callbacks_earlystopping = paddle.callbacks.EarlyStopping(
        "acc",
        mode="auto",
        patience=patience,
        verbose=1,
        min_delta=0,
        baseline=0,
        save_best_model=True,
    )

    mlflow.paddle.autolog()
    model.fit(
        data, epochs=NUM_EPOCHS, batch_size=32, verbose=1, callbacks=[callbacks_earlystopping]
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return model, run


@pytest.mark.parametrize("patience", [2])
def test_paddle_early_stop_params_logged(paddle_model_with_early_stopping, patience):
    _, run = paddle_model_with_early_stopping
    data = run.data
    print(data.params)
    assert "monitor" in data.params
    assert "patience" in data.params
    assert float(data.params["patience"]) == patience
    assert "min_delta" in data.params
    assert "stopped_epoch" in data.params


def test_paddle_autolog_non_early_stop_callback_does_not_log(paddle_model):
    trainer, run = paddle_model
    client = mlflow.tracking.MlflowClient()
    loss_metric_history = client.get_metric_history(run.info.run_id, "loss")
    acc_metric_history = client.get_metric_history(run.info.run_id, "acc")
    step_metric_history = client.get_metric_history(run.info.run_id, "step")
    assert len(loss_metric_history) == NUM_EPOCHS
    assert len(acc_metric_history) == NUM_EPOCHS
    assert len(step_metric_history) == NUM_EPOCHS

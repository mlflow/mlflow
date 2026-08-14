import paddle
import pytest

import mlflow
from mlflow import MlflowClient

NUM_EPOCHS = 6


class LinearRegression(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(13, 1)

    def forward(self, feature):
        return self.fc(feature)


def get_datasets():
    train_dataset = paddle.text.datasets.UCIHousing(mode="train")
    eval_dataset = paddle.text.datasets.UCIHousing(mode="test")
    return train_dataset, eval_dataset


def train_model(**fit_kwargs):
    model = paddle.Model(LinearRegression())
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    model.prepare(optim, paddle.nn.MSELoss())
    train_dataset, eval_dataset = get_datasets()
    model.fit(
        train_dataset, eval_dataset, batch_size=16, epochs=NUM_EPOCHS, verbose=1, **fit_kwargs
    )
    return model


@pytest.mark.parametrize("log_models", [True, False])
def test_autolog_logs_expected_data(log_models):
    mlflow.paddle.autolog(log_models=log_models)

    with mlflow.start_run() as run:
        train_model()

    client = MlflowClient()
    data = client.get_run(run.info.run_id).data

    # Testing params are logged
    for param_key, expected_param_value in [("optimizer_name", "Adam"), ("learning_rate", "0.01")]:
        assert param_key in data.params
        assert data.params[param_key] == expected_param_value

    # Testing metrics are logged
    for metric_key in ["batch_size", "loss", "step", "eval_batch_size", "eval_loss", "eval_step"]:
        assert metric_key in data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    # Testing model_summary.txt is saved
    artifacts = client.list_artifacts(run.info.run_id)
    assert any(x.path == "model_summary.txt" for x in artifacts)

    # Testing metrics are logged to the model
    logged_model = mlflow.last_logged_model()
    if log_models:
        assert logged_model is not None
        assert data.metrics == {m.key: m.value for m in logged_model.metrics}
    else:
        assert logged_model is None


@pytest.mark.parametrize("log_models", [True, False])
def test_autolog_early_stopping_callback(log_models):
    mlflow.paddle.autolog(log_models=log_models)

    early_stopping = paddle.callbacks.EarlyStopping("loss", mode="min", patience=1, min_delta=0)
    with mlflow.start_run() as run:
        train_model(callbacks=[early_stopping])

    client = MlflowClient()
    data = client.get_run(run.info.run_id).data

    for param_key in ["monitor", "patience", "min_delta", "baseline"]:
        assert param_key in data.params
        assert data.params[param_key] == str(getattr(early_stopping, param_key))

    for metric_key in ["stopped_epoch", "best_value"]:
        assert metric_key in data.metrics
        assert float(data.metrics[metric_key]) == getattr(early_stopping, metric_key)

    for metric_key in ["loss", "step"]:
        assert metric_key in data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    logged_model = mlflow.last_logged_model()
    if log_models:
        assert logged_model is not None
        assert data.metrics == {m.key: m.value for m in logged_model.metrics}
    else:
        assert logged_model is None


@pytest.mark.parametrize("log_models", [True, False])
def test_autolog_log_models_configuration(log_models):
    mlflow.paddle.autolog(log_models=log_models)

    with mlflow.start_run() as run:
        train_model()

    MlflowClient().list_artifacts(run.info.run_id)
    assert (mlflow.last_logged_model() is not None) == log_models


def test_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.paddle.autolog(registered_model_name=registered_model_name)

    with mlflow.start_run():
        train_model()

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_extra_tags_paddle_autolog():
    mlflow.paddle.autolog(extra_tags={"test_tag": "paddle_autolog"})
    train_model()

    run = mlflow.last_active_run()
    assert run.data.tags["test_tag"] == "paddle_autolog"
    assert run.data.tags[mlflow.utils.mlflow_tags.MLFLOW_AUTOLOGGING] == "paddle"

import math
import re
from unittest.mock import Mock

import keras
import numpy as np
import pytest

import mlflow
from mlflow.keras.callback import MlflowCallback
from mlflow.tracking.fluent import flush_async_logging


def test_keras_mlflow_callback_log_every_epoch():
    # Prepare data for a 2-class classification.
    data = np.random.uniform(size=(20, 28, 28, 3))
    label = np.random.randint(2, size=20)

    model = keras.Sequential([
        keras.Input([28, 28, 3]),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    num_epochs = 2
    with mlflow.start_run() as run:
        mlflow_callback = MlflowCallback(log_every_epoch=True)
        model.fit(
            data,
            label,
            validation_data=(data, label),
            batch_size=4,
            epochs=num_epochs,
            callbacks=[mlflow_callback],
        )
    flush_async_logging()
    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
    run_metrics = mlflow_run.data.metrics
    model_info = mlflow_run.data.params

    assert "sparse_categorical_accuracy" in run_metrics
    # Keras >= 3.15 uniquifies optimizer names, so the 2nd/3rd optimizer created in
    # the same process is logged as "adam_1"/"adam_2" instead of "adam".
    assert re.fullmatch(r"adam(_\d+)?", model_info["optimizer_name"])
    assert math.isclose(float(model_info["optimizer_learning_rate"]), 0.001, rel_tol=1e-6)
    assert "loss" in run_metrics
    assert "validation_loss" in run_metrics

    loss_history = client.get_metric_history(run_id=run.info.run_id, key="loss")
    assert len(loss_history) == num_epochs

    validation_loss_history = client.get_metric_history(
        run_id=run.info.run_id,
        key="validation_loss",
    )
    assert len(validation_loss_history) == num_epochs
    assert [metric.step for metric in validation_loss_history] == list(range(num_epochs))


def test_keras_mlflow_callback_log_every_n_steps():
    # Prepare data for a 2-class classification.
    data = np.random.uniform(size=(20, 28, 28, 3))
    label = np.random.randint(2, size=20)

    model = keras.Sequential([
        keras.Input([28, 28, 3]),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(0.001),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    log_every_n_steps = 1
    num_epochs = 2
    with mlflow.start_run() as run:
        mlflow_callback = MlflowCallback(log_every_epoch=False, log_every_n_steps=log_every_n_steps)
        model.fit(
            data,
            label,
            validation_data=(data, label),
            batch_size=4,
            epochs=num_epochs,
            callbacks=[mlflow_callback],
        )
    flush_async_logging()
    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
    run_metrics = mlflow_run.data.metrics
    model_info = mlflow_run.data.params

    assert "sparse_categorical_accuracy" in run_metrics
    # Keras >= 3.15 uniquifies optimizer names, so the 2nd/3rd optimizer created in
    # the same process is logged as "adam_1"/"adam_2" instead of "adam".
    assert re.fullmatch(r"adam(_\d+)?", model_info["optimizer_name"])
    assert math.isclose(float(model_info["optimizer_learning_rate"]), 0.001, rel_tol=1e-6)
    assert "loss" in run_metrics
    assert "validation_loss" in run_metrics

    loss_history = client.get_metric_history(run_id=run.info.run_id, key="loss")
    assert len(loss_history) == model.optimizer.iterations.numpy() // log_every_n_steps

    validation_loss_history = client.get_metric_history(
        run_id=run.info.run_id,
        key="validation_loss",
    )
    assert len(validation_loss_history) == num_epochs
    assert [metric.step for metric in validation_loss_history] == [5, 10]


def test_old_callback_still_exists():
    assert mlflow.keras.MLflowCallback is mlflow.keras.MlflowCallback


@pytest.mark.parametrize(
    ("log_every_epoch", "log_every_n_steps", "epoch", "train_ended", "expected_step"),
    [
        pytest.param(True, None, 2, False, 2, id="epoch"),
        pytest.param(False, 1, 2, False, 7, id="iteration"),
        pytest.param(True, None, None, False, None, id="standalone-epoch"),
        pytest.param(False, 1, None, False, None, id="standalone-iteration"),
        pytest.param(True, None, 2, True, None, id="reused-epoch"),
        pytest.param(False, 1, 2, True, None, id="reused-iteration"),
    ],
)
def test_validation_metrics_use_training_step(
    monkeypatch, log_every_epoch, log_every_n_steps, epoch, train_ended, expected_step
):
    callback = MlflowCallback(
        log_every_epoch=log_every_epoch,
        log_every_n_steps=log_every_n_steps,
    )
    if epoch is not None:
        callback.on_epoch_begin(epoch)
        callback.set_model(Mock(optimizer=Mock(iterations=Mock(numpy=Mock(return_value=7)))))
    if train_ended:
        callback.on_train_end()
    log_metrics = Mock()
    monkeypatch.setattr("mlflow.keras.callback.log_metrics", log_metrics)

    callback.on_test_end({"loss": 0.5})

    kwargs = {"synchronous": False, "model_id": None}
    if expected_step is not None:
        kwargs["step"] = expected_step
    log_metrics.assert_called_once_with({"validation_loss": 0.5}, **kwargs)

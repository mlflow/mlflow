import math

import keras
import numpy as np

import mlflow
from mlflow.keras.callback import MlflowCallback
from mlflow.tracking.fluent import flush_async_logging


def test_keras_mlflow_callback_log_every_epoch():
    # Prepare data for a 2-class classification.
    data = np.random.uniform(size=(20, 28, 28, 3))
    label = np.random.randint(2, size=20)

    model = keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )

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
    assert model_info["optimizer_name"] == "adam"
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


def test_keras_mlflow_callback_log_every_n_steps():
    # Prepare data for a 2-class classification.
    data = np.random.uniform(size=(20, 28, 28, 3))
    label = np.random.randint(2, size=20)

    model = keras.Sequential(
        [
            keras.Input([28, 28, 3]),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )

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
    assert model_info["optimizer_name"] == "adam"
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


def test_old_callback_still_exists():
    assert mlflow.keras.MLflowCallback is mlflow.keras.MlflowCallback

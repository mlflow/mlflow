import math

import keras_core as keras
import numpy as np
import pytest

import mlflow
from mlflow.keras_core.callback import MLflowCallback


@pytest.mark.parametrize(("log_every_epoch", "log_every_n_steps"), [(True, None), (False, 1)])
def test_keras_core_mlflow_callback(log_every_epoch, log_every_n_steps):
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

    with mlflow.start_run() as run:
        mlflow_callback = MLflowCallback(
            run=run,
            log_every_epoch=log_every_epoch,
            log_every_n_steps=log_every_n_steps,
        )
        model.fit(
            data,
            label,
            validation_data=(data, label),
            batch_size=4,
            epochs=2,
            callbacks=[mlflow_callback],
        )

    client = mlflow.MlflowClient()
    mlflow_run = client.get_run(run.info.run_id)
    run_metrics = mlflow_run.data.metrics
    model_info = mlflow_run.data.params

    assert "loss" in run_metrics
    assert "sparse_categorical_accuracy" in run_metrics
    assert model_info["optimizer_name"] == "adam"
    assert math.isclose(float(model_info["optimizer_learning_rate"]), 0.001, rel_tol=1e-6)

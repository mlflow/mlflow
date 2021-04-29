import pytest
import numpy as np

np.random.seed(1337)

import keras  # noqa
import keras.layers as layers  # noqa

import mlflow  # noqa
import mlflow.keras  # noqa
from mlflow.utils.autologging_utils import BatchMetricsLogger  # noqa
from unittest.mock import patch  # noqa


@pytest.fixture(autouse=True)
def clear_session():
    yield
    # Release the global state managed by Keras to avoid this issue:
    # https://github.com/keras-team/keras/issues/2102
    keras.backend.clear_session()


@pytest.fixture
def random_train_data():
    return np.random.random((150, 4))


@pytest.fixture
def random_one_hot_labels():
    n, n_class = (150, 3)
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


@pytest.fixture(params=[True, False])
def manual_run(request):
    if request.param:
        mlflow.start_run()
    yield
    mlflow.end_run()


def create_model():
    model = keras.Sequential()

    model.add(layers.Dense(16, activation="relu", input_shape=(4,)))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-07),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    return model


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_keras_autolog_ends_auto_created_run(random_train_data, random_one_hot_labels, fit_variant):
    mlflow.keras.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_model()

    if fit_variant == "fit_generator":

        def generator():
            while True:
                yield data, labels

        model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
    else:
        model.fit(data, labels, epochs=10)

    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_keras_autolog_persists_manually_created_run(
    random_train_data, random_one_hot_labels, fit_variant
):
    mlflow.keras.autolog()

    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels

        model = create_model()

        if fit_variant == "fit_generator":

            def generator():
                while True:
                    yield data, labels

            model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
        else:
            model.fit(data, labels, epochs=10)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def keras_random_data_run(
    random_train_data, fit_variant, random_one_hot_labels, manual_run, initial_epoch
):
    # pylint: disable=unused-argument
    mlflow.keras.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_model()

    if fit_variant == "fit_generator":

        def generator():
            while True:
                yield data, labels

        history = model.fit_generator(
            generator(), epochs=initial_epoch + 10, steps_per_epoch=1, initial_epoch=initial_epoch
        )
    else:
        history = model.fit(
            data, labels, epochs=initial_epoch + 10, steps_per_epoch=1, initial_epoch=initial_epoch
        )

    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id), history


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_logs_expected_data(keras_random_data_run):
    run, history = keras_random_data_run
    data = run.data
    assert "acc" in data.metrics
    assert "loss" in data.metrics
    # Testing explicitly passed parameters are logged correctly
    assert "epochs" in data.params
    assert data.params["epochs"] == str(history.epoch[-1] + 1)
    assert "steps_per_epoch" in data.params
    assert data.params["steps_per_epoch"] == "1"
    # Testing default parameters are logged correctly
    assert "initial_epoch" in data.params
    assert data.params["initial_epoch"] == str(history.epoch[0])
    # Testing unwanted parameters are not logged
    assert "callbacks" not in data.params
    assert "validation_data" not in data.params
    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"
    assert "epsilon" in data.params
    assert data.params["epsilon"] == "1e-07"
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_model_can_load_from_artifact(keras_random_data_run, random_train_data):
    run, _ = keras_random_data_run
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    model = mlflow.keras.load_model("runs:/" + run_id + "/model")
    model.predict(random_train_data)


@pytest.fixture
def keras_random_data_run_with_callback(
    random_train_data,
    fit_variant,
    random_one_hot_labels,
    manual_run,
    callback,
    restore_weights,
    patience,
    initial_epoch,
):
    # pylint: disable=unused-argument
    mlflow.keras.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_model()
    if callback == "early":
        # min_delta is set as such to guarantee early stopping
        callback = keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=patience,
            min_delta=99999999,
            restore_best_weights=restore_weights,
            verbose=1,
        )
    else:

        class CustomCallback(keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                print("Training completed")

        callback = CustomCallback()

    if fit_variant == "fit_generator":

        def generator():
            while True:
                yield data, labels

        history = model.fit_generator(
            generator(),
            epochs=initial_epoch + 10,
            callbacks=[callback],
            steps_per_epoch=1,
            shuffle=False,
            initial_epoch=initial_epoch,
        )
    else:
        history = model.fit(
            data,
            labels,
            epochs=initial_epoch + 10,
            callbacks=[callback],
            initial_epoch=initial_epoch,
        )

    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id), history, callback


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_keras_autolog_log_models_configuration(
    random_train_data, random_one_hot_labels, log_models
):
    mlflow.keras.autolog(log_models=log_models)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_model()
    model.fit(data, labels, epochs=10, steps_per_epoch=1)

    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("model" in artifacts) == log_models


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [0, 1, 5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_early_stop_logs(keras_random_data_run_with_callback):
    run, history, callback = keras_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" in params
    assert params["patience"] == str(callback.patience)
    assert "monitor" in params
    assert params["monitor"] == "loss"
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" in metrics
    assert "restored_epoch" in metrics
    restored_epoch = int(metrics["restored_epoch"])
    assert int(metrics["stopped_epoch"]) - max(1, callback.patience) == restored_epoch
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == max(1, callback.patience) + 1
    # Check that MLflow has logged the metrics of the "best" model
    assert len(metric_history) == num_of_epochs + 1
    # Check that MLflow has logged the correct data
    assert history.history["loss"][history.epoch.index(restored_epoch)] == metric_history[-1].value


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [0, 1, 5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_batch_metrics_logger_logs_expected_metrics(
    fit_variant, callback, restore_weights, patience, initial_epoch
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
        run, _, callback = keras_random_data_run_with_callback(
            random_train_data(),
            fit_variant,
            random_one_hot_labels(),
            manual_run,
            callback,
            restore_weights,
            patience,
            initial_epoch,
        )

    patched_metrics_data = dict(patched_metrics_data)
    original_metrics = run.data.metrics
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data

    restored_epoch = int(patched_metrics_data["restored_epoch"])
    assert int(patched_metrics_data["stopped_epoch"]) - max(1, callback.patience) == restored_epoch
    restored_epoch = int(original_metrics["restored_epoch"])
    assert int(original_metrics["stopped_epoch"]) - max(1, callback.patience) == restored_epoch


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [11])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_early_stop_no_stop_does_not_log(keras_random_data_run_with_callback):
    run, history, callback = keras_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" in params
    assert params["patience"] == str(callback.patience)
    assert "monitor" in params
    assert params["monitor"] == "loss"
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" in metrics
    assert metrics["stopped_epoch"] == 0
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("restore_weights", [False])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_early_stop_no_restore_does_not_log(keras_random_data_run_with_callback):
    run, history, callback = keras_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" in params
    assert params["patience"] == str(callback.patience)
    assert "monitor" in params
    assert params["monitor"] == "loss"
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" in metrics
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == callback.patience + 1
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("restore_weights", [False])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_keras_autolog_non_early_stop_callback_does_not_log(keras_random_data_run_with_callback):
    run, history = keras_random_data_run_with_callback[:-1]
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" not in params
    assert "monitor" not in params
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs

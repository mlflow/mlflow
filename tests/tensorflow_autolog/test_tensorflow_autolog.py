# pep8: disable=E501

import collections
import shutil
import pytest
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import mlflow
import mlflow.tensorflow
import mlflow.keras

import os

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"],
)


@pytest.fixture(autouse=True)
def clear_session():
    yield
    tf.keras.backend.clear_session()


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


def create_tf_keras_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(16, activation="relu", input_shape=(4,)))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-08, name="Eve"),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_tf_keras_autolog_ends_auto_created_run(
    random_train_data, random_one_hot_labels, fit_variant
):
    # pylint: disable=unused-argument
    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    model.fit(data, labels, epochs=10)

    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize("log_models", [True, False])
def test_tf_keras_autolog_log_models_configuration(
    random_train_data, random_one_hot_labels, log_models
):
    # pylint: disable=unused-argument
    mlflow.tensorflow.autolog(log_models=log_models)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    model.fit(data, labels, epochs=10)

    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert ("model" in artifacts) == log_models


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_tf_keras_autolog_persists_manually_created_run(
    random_train_data, random_one_hot_labels, fit_variant
):
    # pylint: disable=unused-argument
    mlflow.tensorflow.autolog()
    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels

        model = create_tf_keras_model()

        model.fit(data, labels, epochs=10)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def tf_keras_random_data_run(
    random_train_data, random_one_hot_labels, manual_run, fit_variant, initial_epoch
):
    # pylint: disable=unused-argument
    mlflow.tensorflow.autolog(every_n_iter=5)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

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
def test_tf_keras_autolog_logs_expected_data(tf_keras_random_data_run):
    run, history = tf_keras_random_data_run
    data = run.data
    assert "epoch_acc" in data.metrics
    assert "epoch_loss" in data.metrics
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
    assert data.params["optimizer_name"] == "Eve"
    assert "learning_rate" in data.params
    assert data.params["learning_rate"] == "0.002"
    assert "epsilon" in data.params
    assert data.params["epsilon"] == "1e-08"
    client = mlflow.tracking.MlflowClient()
    all_epoch_acc = client.get_metric_history(run.info.run_id, "epoch_acc")
    assert all((x.step - 1) % 5 == 0 for x in all_epoch_acc)
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_model_can_load_from_artifact(tf_keras_random_data_run, random_train_data):
    run, _ = tf_keras_random_data_run
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    assert "tensorboard_logs" in artifacts
    model = mlflow.keras.load_model("runs:/" + run.info.run_id + "/model")
    model.predict(random_train_data)


@pytest.fixture
def tf_keras_random_data_run_with_callback(
    random_train_data,
    random_one_hot_labels,
    manual_run,
    callback,
    restore_weights,
    patience,
    initial_epoch,
):
    # pylint: disable=unused-argument
    mlflow.tensorflow.autolog(every_n_iter=1)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()
    if callback == "early":
        # min_delta is set as such to guarantee early stopping
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=patience,
            min_delta=99999999,
            restore_best_weights=restore_weights,
            verbose=1,
        )
    else:
        callback = tf.keras.callbacks.ProgbarLogger(count_mode="samples")

    history = model.fit(
        data, labels, epochs=initial_epoch + 10, callbacks=[callback], initial_epoch=initial_epoch
    )

    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id), history, callback


@pytest.mark.large
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [0, 1, 5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_early_stop_logs(tf_keras_random_data_run_with_callback):
    run, history, callback = tf_keras_random_data_run_with_callback
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
    assert "epoch_loss" in metrics
    restored_epoch = int(metrics["restored_epoch"])
    assert int(metrics["stopped_epoch"]) - max(1, callback.patience) == restored_epoch
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    # TF 1.X TB callback logs loss as `epoch_loss`
    metric_history = client.get_metric_history(run.info.run_id, "epoch_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == max(1, callback.patience) + 1
    # Check that MLflow has logged the metrics of the "best" model
    assert len(metric_history) == num_of_epochs + 1
    # Check that MLflow has logged the correct data
    assert history.history["loss"][history.epoch.index(restored_epoch)] == metric_history[-1].value


@pytest.mark.large
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [11])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_early_stop_no_stop_does_not_log(tf_keras_random_data_run_with_callback):
    run, history, callback = tf_keras_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" in params
    assert params["patience"] == str(callback.patience)
    assert "monitor" in params
    assert params["monitor"] == "loss"
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" in metrics
    assert "epoch_loss" in metrics
    assert metrics["stopped_epoch"] == 0
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "epoch_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("restore_weights", [False])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_early_stop_no_restore_doesnt_log(tf_keras_random_data_run_with_callback):
    run, history, callback = tf_keras_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" in params
    assert params["patience"] == str(callback.patience)
    assert "monitor" in params
    assert params["monitor"] == "loss"
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" in metrics
    assert "epoch_loss" in metrics
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "epoch_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == callback.patience + 1
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("restore_weights", [False])
@pytest.mark.parametrize("callback", ["not-early"])
@pytest.mark.parametrize("patience", [5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_non_early_stop_callback_no_log(tf_keras_random_data_run_with_callback):
    run, history = tf_keras_random_data_run_with_callback[:-1]
    metrics = run.data.metrics
    params = run.data.params
    assert "patience" not in params
    assert "monitor" not in params
    assert "verbose" not in params
    assert "mode" not in params
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "epoch_loss" in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "epoch_loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_tf_keras_autolog_does_not_delete_logging_directory_for_tensorboard_callback(
    tmpdir, random_train_data, random_one_hot_labels, fit_variant
):
    tensorboard_callback_logging_dir_path = str(tmpdir.mkdir("tb_logs"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_callback_logging_dir_path, histogram_freq=0
    )

    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    if fit_variant == "fit_generator":

        def generator():
            while True:
                yield data, labels

        model.fit_generator(
            generator(), epochs=10, steps_per_epoch=1, callbacks=[tensorboard_callback]
        )
    else:
        model.fit(data, labels, epochs=10, callbacks=[tensorboard_callback])

    assert os.path.exists(tensorboard_callback_logging_dir_path)


@pytest.mark.large
@pytest.mark.parametrize("fit_variant", ["fit", "fit_generator"])
def test_tf_keras_autolog_logs_to_and_deletes_temporary_directory_when_tensorboard_callback_absent(
    tmpdir, random_train_data, random_one_hot_labels, fit_variant
):
    from unittest import mock
    from mlflow.tensorflow import _TensorBoardLogDir

    mlflow.tensorflow.autolog()

    mock_log_dir_inst = _TensorBoardLogDir(location=str(tmpdir.mkdir("tb_logging")), is_temp=True)
    with mock.patch("mlflow.tensorflow._TensorBoardLogDir", autospec=True) as mock_log_dir_class:
        mock_log_dir_class.return_value = mock_log_dir_inst

        data = random_train_data
        labels = random_one_hot_labels

        model = create_tf_keras_model()

        if fit_variant == "fit_generator":

            def generator():
                while True:
                    yield data, labels

            model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
        else:
            model.fit(data, labels, epochs=10)

        assert not os.path.exists(mock_log_dir_inst.location)


@pytest.fixture
def tf_core_random_tensors():
    mlflow.tensorflow.autolog(every_n_iter=4)
    with mlflow.start_run() as run:
        sess = tf.Session()
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)
        total = a + b
        tf.summary.scalar("a", a)
        tf.summary.scalar("b", b)
        merged = tf.summary.merge_all()
        directory = tempfile.mkdtemp()
        writer = tf.summary.FileWriter(directory, sess.graph)
        with sess.as_default():
            for i in range(40):
                summary, _ = sess.run([merged, total])
                writer.add_summary(summary, global_step=i)
        shutil.rmtree(directory)
        writer.close()
        sess.close()

    client = mlflow.tracking.MlflowClient()
    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_core_autolog_logs_scalars(tf_core_random_tensors):
    assert "a" in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics["a"] == 3.0
    assert "b" in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics["b"] == 4.0
    client = mlflow.tracking.MlflowClient()
    all_a = client.get_metric_history(tf_core_random_tensors.info.run_id, "a")
    assert all((x.step - 1) % 4 == 0 for x in all_a)
    assert mlflow.active_run() is None


def create_tf_estimator_model(directory, export):
    CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]

    train = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "iris_training.csv"),
        names=CSV_COLUMN_NAMES,
        header=0,
    )

    train_y = train.pop("Species")

    def input_fn(features, labels, training=True, batch_size=256):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    feature_spec = {}
    for feature in CSV_COLUMN_NAMES:
        feature_spec[feature] = tf.placeholder(dtype="float", name=feature, shape=[150])

    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[30, 10],
        # The model must choose between 3 classes.
        n_classes=3,
        model_dir=directory,
    )

    classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=500)
    if export:
        classifier.export_saved_model(directory, receiver_fn)


@pytest.mark.large
@pytest.mark.parametrize("export", [True, False])
def test_tf_estimator_autolog_ends_auto_created_run(tmpdir, export):
    directory = tmpdir.mkdir("test")
    mlflow.tensorflow.autolog()
    create_tf_estimator_model(str(directory), export)
    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize("export", [True, False])
def test_tf_estimator_autolog_persists_manually_created_run(tmpdir, export):
    directory = tmpdir.mkdir("test")
    with mlflow.start_run() as run:
        create_tf_estimator_model(str(directory), export)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def tf_estimator_random_data_run(tmpdir, manual_run, export):
    # pylint: disable=unused-argument
    directory = tmpdir.mkdir("test")
    mlflow.tensorflow.autolog()
    create_tf_estimator_model(str(directory), export)
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("export", [True, False])
def test_tf_estimator_autolog_logs_metrics(tf_estimator_random_data_run):
    assert "loss" in tf_estimator_random_data_run.data.metrics
    assert "steps" in tf_estimator_random_data_run.data.params
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_metric_history(tf_estimator_random_data_run.info.run_id, "loss")
    assert all((x.step - 1) % 100 == 0 for x in metrics)


@pytest.mark.large
@pytest.mark.parametrize("export", [True])
def test_tf_estimator_autolog_model_can_load_from_artifact(tf_estimator_random_data_run):
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(tf_estimator_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    session = tf.Session()
    mlflow.tensorflow.load_model(
        "runs:/" + tf_estimator_random_data_run.info.run_id + "/model", session
    )


@pytest.mark.large
@pytest.mark.parametrize("export", [True, False])
def test_duplicate_autolog_second_overrides(tf_estimator_random_data_run):
    mlflow.tensorflow.autolog(every_n_iter=23)
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_metric_history(tf_estimator_random_data_run.info.run_id, "loss")
    assert all((x.step - 1) % 4 == 0 for x in metrics)

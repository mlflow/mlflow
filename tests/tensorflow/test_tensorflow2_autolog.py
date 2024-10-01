# pep8: disable=E501

import collections
import functools
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
import yaml
from packaging.version import Version
from tensorflow.keras import layers

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.utils import _read_example
from mlflow.tensorflow import load_checkpoint
from mlflow.tensorflow.autologging import _TensorBoard
from mlflow.tensorflow.callback import MlflowCallback
from mlflow.tracking.fluent import _shut_down_async_logging
from mlflow.types.utils import _infer_schema
from mlflow.utils.autologging_utils import (
    AUTOLOGGING_INTEGRATIONS,
    autologging_is_disabled,
)
from mlflow.utils.process import _exec_cmd

np.random.seed(1337)

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"],
)


@pytest.fixture(autouse=True)
def clear_session():
    yield
    _shut_down_async_logging()
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


@pytest.fixture
def random_train_dict_mapping(random_train_data):
    def _generate_features(pos):
        return [v[pos] for v in random_train_data]

    return {
        "a": np.array(_generate_features(0)),
        "b": np.array(_generate_features(1)),
        "c": np.array(_generate_features(2)),
        "d": np.array(_generate_features(3)),
    }


def _create_model_for_dict_mapping():
    inputs = {
        "a": tf.keras.Input(shape=(1,), name="a"),
        "b": tf.keras.Input(shape=(1,), name="b"),
        "c": tf.keras.Input(shape=(1,), name="c"),
        "d": tf.keras.Input(shape=(1,), name="d"),
    }
    concatenated = layers.Concatenate()(list(inputs.values()))
    x = layers.Dense(16, activation="relu", input_shape=(4,))(concatenated)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


@pytest.fixture
def fashion_mnist_tf_dataset():
    train, _ = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = train
    images = images / 255.0
    labels = labels.astype(np.int32)
    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    return fmnist_train_ds.shuffle(5000).batch(32)


@pytest.fixture
def fashion_mnist_tf_dataset_eval():
    _, eval_dataset = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = eval_dataset
    images = images / 255.0
    labels = labels.astype(np.int32)
    fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    return fmnist_train_ds.shuffle(5000).batch(32)


def _create_fashion_mnist_model():
    model = tf.keras.Sequential(
        [tf.keras.Input((28, 28)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10)]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


@pytest.fixture
def keras_data_gen_sequence(random_train_data, random_one_hot_labels):
    class DataGenerator(tf.keras.utils.Sequence):
        def __len__(self):
            return 128

        def __getitem__(self, index):
            x = random_train_data
            y = random_one_hot_labels
            return x, y

    return DataGenerator()


@pytest.fixture(autouse=True)
def clear_fluent_autologging_import_hooks():
    """
    Clears import hooks for MLflow fluent autologging (`mlflow.autolog()`) between tests
    to ensure that interactions between fluent autologging and TensorFlow / tf.keras can
    be tested successfully
    """
    mlflow.utils.import_hooks._post_import_hooks.pop("tensorflow", None)
    mlflow.utils.import_hooks._post_import_hooks.pop("keras", None)


@pytest.fixture(autouse=True)
def clear_autologging_config():
    """
    Clears TensorFlow autologging config, simulating a fresh state where autologging has not
    been previously enabled with any particular configuration
    """
    AUTOLOGGING_INTEGRATIONS.pop(mlflow.tensorflow.FLAVOR_NAME, None)


def create_tf_keras_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(4,), dtype="float64"))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def test_tf_keras_autolog_ends_auto_created_run(random_train_data, random_one_hot_labels):
    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()
    model.fit(data, labels, epochs=10)

    assert mlflow.active_run() is None


def test_extra_tags_tensorflow_autolog(random_train_data, random_one_hot_labels):
    mlflow.tensorflow.autolog(extra_tags={"test_tag": "tf_autolog"})

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()
    model.fit(data, labels, epochs=10)

    run = mlflow.last_active_run()
    assert run.data.tags["test_tag"] == "tf_autolog"
    assert run.data.tags[mlflow.utils.mlflow_tags.MLFLOW_AUTOLOGGING] == "tensorflow"


@pytest.mark.parametrize("log_models", [True, False])
def test_tf_keras_autolog_log_models_configuration(
    random_train_data, random_one_hot_labels, log_models
):
    mlflow.tensorflow.autolog(log_models=log_models)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    model.fit(data, labels, epochs=10)

    client = MlflowClient()
    run_id = client.search_runs(["0"])[0].info.run_id
    artifacts = client.list_artifacts(run_id)
    artifacts = (x.path for x in artifacts)
    assert ("model" in artifacts) == log_models


@pytest.mark.parametrize("log_datasets", [True, False])
def test_tf_keras_autolog_log_datasets_configuration_with_numpy(
    random_train_data, random_one_hot_labels, log_datasets
):
    mlflow.tensorflow.autolog(log_datasets=log_datasets)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    model.fit(data, labels, epochs=10)

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    if log_datasets:
        assert len(dataset_inputs) == 1
        feature_schema = _infer_schema(data)
        target_schema = _infer_schema(labels)
        assert dataset_inputs[0].dataset.schema == json.dumps(
            {
                "mlflow_tensorspec": {
                    "features": feature_schema.to_json(),
                    "targets": target_schema.to_json(),
                }
            }
        )
    else:
        assert len(dataset_inputs) == 0


@pytest.mark.parametrize("log_datasets", [True, False])
def test_tf_keras_autolog_log_datasets_configuration_with_tensor(
    random_train_data, random_one_hot_labels, log_datasets
):
    mlflow.tensorflow.autolog(log_datasets=log_datasets)

    data_as_tensor = tf.convert_to_tensor(random_train_data)
    labels_as_tensor = tf.convert_to_tensor(random_one_hot_labels)

    model = create_tf_keras_model()

    model.fit(data_as_tensor, labels_as_tensor, epochs=10)

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    if log_datasets:
        assert len(dataset_inputs) == 1
        feature_schema = _infer_schema(data_as_tensor.numpy())
        target_schema = _infer_schema(labels_as_tensor.numpy())
        assert dataset_inputs[0].dataset.schema == json.dumps(
            {
                "mlflow_tensorspec": {
                    "features": feature_schema.to_json(),
                    "targets": target_schema.to_json(),
                }
            }
        )
    else:
        assert len(dataset_inputs) == 0


@pytest.mark.parametrize("log_datasets", [True, False])
def test_tf_keras_autolog_log_datasets_configuration_with_tf_dataset(
    fashion_mnist_tf_dataset, log_datasets
):
    mlflow.tensorflow.autolog(log_datasets=log_datasets)
    fashion_mnist_model = _create_fashion_mnist_model()
    fashion_mnist_model.fit(fashion_mnist_tf_dataset)

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    if log_datasets:
        assert len(dataset_inputs) == 1
        numpy_data = next(fashion_mnist_tf_dataset.as_numpy_iterator())
        assert dataset_inputs[0].dataset.schema == json.dumps(
            {
                "mlflow_tensorspec": {
                    "features": _infer_schema(
                        {str(i): data_element for i, data_element in enumerate(numpy_data)}
                    ).to_json(),
                    "targets": None,
                }
            }
        )

    else:
        assert len(dataset_inputs) == 0


def test_tf_keras_autolog_log_datasets_with_validation_data(
    fashion_mnist_tf_dataset, fashion_mnist_tf_dataset_eval
):
    mlflow.tensorflow.autolog(log_datasets=True)
    fashion_mnist_model = _create_fashion_mnist_model()
    fashion_mnist_model.fit(fashion_mnist_tf_dataset, validation_data=fashion_mnist_tf_dataset_eval)

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    assert dataset_inputs[1].tags[0].value == "eval"


def test_tf_keras_autolog_log_datasets_with_validation_data_as_numpy_tuple(
    fashion_mnist_tf_dataset, fashion_mnist_tf_dataset_eval
):
    mlflow.tensorflow.autolog(log_datasets=True)
    fashion_mnist_model = _create_fashion_mnist_model()
    X_eval, y_eval = next(fashion_mnist_tf_dataset_eval.as_numpy_iterator())
    fashion_mnist_model.fit(fashion_mnist_tf_dataset, validation_data=(X_eval, y_eval))

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    assert dataset_inputs[1].tags[0].value == "eval"


def test_tf_keras_autolog_log_datasets_with_validation_data_as_tf_tuple(
    fashion_mnist_tf_dataset, fashion_mnist_tf_dataset_eval
):
    mlflow.tensorflow.autolog(log_datasets=True)
    fashion_mnist_model = _create_fashion_mnist_model()
    # convert tensorflow dataset into tensors
    X_eval, y_eval = next(fashion_mnist_tf_dataset_eval.as_numpy_iterator())
    X_eval_tensor = tf.convert_to_tensor(X_eval)
    y_eval_tensor = tf.convert_to_tensor(y_eval)
    fashion_mnist_model.fit(
        fashion_mnist_tf_dataset, validation_data=(X_eval_tensor, y_eval_tensor)
    )

    client = MlflowClient()
    dataset_inputs = client.get_run(mlflow.last_active_run().info.run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    assert dataset_inputs[1].tags[0].value == "eval"


def test_tf_keras_autolog_persists_manually_created_run(random_train_data, random_one_hot_labels):
    mlflow.tensorflow.autolog()
    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels

        model = create_tf_keras_model()
        model.fit(data, labels, epochs=10)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def tf_keras_random_data_run(random_train_data, random_one_hot_labels, initial_epoch):
    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()
    history = model.fit(
        data, labels, epochs=initial_epoch + 10, steps_per_epoch=1, initial_epoch=initial_epoch
    )

    client = MlflowClient()
    return client.get_run(client.search_runs(["0"])[0].info.run_id), history


@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_logs_expected_data(tf_keras_random_data_run):
    run, history = tf_keras_random_data_run
    data = run.data
    assert "accuracy" in data.metrics
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
    assert "opt_name" in data.params
    assert data.params["opt_name"].lower() == "adam"
    assert "opt_learning_rate" in data.params
    assert "opt_beta_1" in data.params
    assert "opt_beta_2" in data.params
    assert "opt_epsilon" in data.params
    assert "opt_amsgrad" in data.params
    assert data.params["opt_amsgrad"] == "False"
    client = MlflowClient()
    all_epoch_acc = client.get_metric_history(run.info.run_id, "accuracy")
    num_of_epochs = len(history.history["loss"])
    assert len(all_epoch_acc) == num_of_epochs == 10
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = (x.path for x in artifacts)
    assert "model_summary.txt" in artifacts


def __example_tf_dataset(batch_size):
    a = tf.data.Dataset.range(1)
    b = tf.data.Dataset.range(1)
    ds = tf.data.Dataset.zip((a, b))
    return ds.batch(batch_size)


class __ExampleSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, with_sample_weights=False):
        self.batch_size = batch_size
        self.with_sample_weights = with_sample_weights

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        x = np.array([idx] * self.batch_size)
        y = np.array([-idx] * self.batch_size)
        if self.with_sample_weights:
            w = np.array([1] * self.batch_size)
            return x, y, w
        return x, y


def __generator(data, target, batch_size):
    data_batches = np.split(data, data.shape[0] // batch_size)
    target_batches = np.split(target, target.shape[0] // batch_size)
    yield from zip(data_batches, target_batches)


class __GeneratorClass:
    def __init__(self, data, target, batch_size):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.ptr = 0

    def __next__(self):
        if self.ptr >= len(self.data):
            raise StopIteration
        idx = self.ptr % len(self.data)
        self.ptr += 1
        return self.data[idx : idx + self.batch_size], self.target[idx : idx + self.batch_size]

    def __iter__(self):
        return self


@pytest.mark.parametrize(
    "generate_data",
    [
        __example_tf_dataset,
        __ExampleSequence,
        functools.partial(__ExampleSequence, with_sample_weights=True),
        functools.partial(__generator, np.array([[1]] * 10), np.array([[1]] * 10)),
        pytest.param(
            functools.partial(__GeneratorClass, np.array([[1]] * 10), np.array([[1]] * 10)),
            marks=pytest.mark.skipif(
                Version(tf.__version__).release >= (2, 15)
                and "TF_USE_LEGACY_KERAS" not in os.environ,
                reason="does not support",
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [5, 10])
def test_tf_keras_autolog_implicit_batch_size_works(generate_data, batch_size):
    mlflow.autolog()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    model.compile(loss="mse")

    # 'x' passed as arg
    model.fit(generate_data(batch_size), verbose=0)
    assert mlflow.last_active_run().data.params["batch_size"] == str(batch_size)

    # 'x' passed as kwarg
    model.fit(x=generate_data(batch_size), verbose=0)
    assert mlflow.last_active_run().data.params["batch_size"] == str(batch_size)


def __tf_dataset_multi_input(batch_size):
    a = tf.data.Dataset.range(1)
    b = tf.data.Dataset.range(1)
    c = tf.data.Dataset.range(1)
    ds = tf.data.Dataset.zip(((a, b), c))
    return ds.batch(batch_size)


class __SequenceMultiInput(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return (np.random.rand(self.batch_size), np.random.rand(self.batch_size)), np.random.rand(
            self.batch_size
        )


def __generator_multi_input(data, target, batch_size):
    data_batches = np.split(data, data.shape[1] // batch_size, axis=1)
    target_batches = np.split(target, target.shape[0] // batch_size)
    for inputs, output in zip(data_batches, target_batches):
        yield tuple(inputs), output


class __GeneratorClassMultiInput:
    def __init__(self, data, target, batch_size):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.ptr = 0

    def __next__(self):
        if self.ptr >= len(self.data):
            raise StopIteration
        idx = self.ptr % len(self.data)
        self.ptr += 1
        return (
            self.data[idx : idx + self.batch_size, 0],
            self.data[idx : idx + self.batch_size, 1],
        ), self.target[idx : idx + self.batch_size]

    def __iter__(self):
        return self


@pytest.mark.parametrize(
    "generate_data",
    [
        __tf_dataset_multi_input,
        __SequenceMultiInput,
        functools.partial(__generator_multi_input, np.random.rand(2, 10), np.random.rand(10)),
        functools.partial(__GeneratorClassMultiInput, np.random.rand(10, 2), np.random.rand(10, 1)),
    ],
)
@pytest.mark.parametrize("batch_size", [5, 10])
def test_tf_keras_autolog_implicit_batch_size_works_multi_input(generate_data, batch_size):
    mlflow.tensorflow.autolog()

    input1 = tf.keras.Input(shape=(1,))
    input2 = tf.keras.Input(shape=(1,))
    concat = tf.keras.layers.Concatenate()([input1, input2])
    output = tf.keras.layers.Dense(1, activation="sigmoid")(concat)

    model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
    model.compile(loss="mse")

    # 'x' passed as arg
    model.fit(generate_data(batch_size), verbose=0)
    assert mlflow.last_active_run().data.params["batch_size"] == str(batch_size)

    # 'x' passed as kwarg
    model.fit(x=generate_data(batch_size), verbose=0)
    assert mlflow.last_active_run().data.params["batch_size"] == str(batch_size)


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.1.4"),
    reason="Does not support passing of generator classes as `x` in `fit`",
)
@pytest.mark.parametrize(
    "generator",
    [
        __generator,
        pytest.param(
            __GeneratorClass,
            marks=pytest.mark.skipif(
                Version(tf.__version__).release >= (2, 15)
                and "TF_USE_LEGACY_KERAS" not in os.environ,
                reason="does not support",
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [2, 3, 6])
def test_tf_keras_autolog_implicit_batch_size_for_generator_dataset_without_side_effects(
    generator,
    batch_size,
):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    data = np.array([[1, 2, 3], [3, 2, 1], [2, 2, 2], [10, 20, 30], [30, 20, 10], [20, 20, 20]])
    target = np.array([[1], [3], [2], [11], [13], [12]])

    model = Sequential()
    model.add(
        Dense(
            5, input_dim=3, activation="relu", kernel_initializer="zeros", bias_initializer="zeros"
        )
    )
    model.add(Dense(1, kernel_initializer="zeros", bias_initializer="zeros"))
    model.compile(loss="mae", optimizer="adam", metrics=["mse"])

    mlflow.autolog()
    actual_mse = model.fit(generator(data, target, batch_size), verbose=0).history["mse"][-1]

    mlflow.autolog(disable=True)
    expected_mse = model.fit(generator(data, target, batch_size), verbose=0).history["mse"][-1]

    np.testing.assert_allclose(actual_mse, expected_mse, atol=1)
    assert mlflow.last_active_run().data.params["batch_size"] == str(batch_size)


def test_tf_keras_autolog_succeeds_for_tf_datasets_lacking_batch_size_info():
    X_train = np.random.rand(100, 100)
    y_train = np.random.randint(0, 10, 100)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.batch(50)
    train_ds = train_ds.cache().prefetch(buffer_size=5)
    assert not hasattr(train_ds, "_batch_size")

    model = tf.keras.Sequential()
    model.add(tf.keras.Input((100,)))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.4))
    model.add(tf.keras.layers.Dense(10, activation="sigmoid"))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="Adam",
        metrics=["accuracy"],
    )

    mlflow.tensorflow.autolog()
    model.fit(train_ds, epochs=100)

    assert mlflow.last_active_run().data.params["batch_size"] == "None"


def test_tf_keras_autolog_records_metrics_for_last_epoch(random_train_data, random_one_hot_labels):
    num_training_epochs = 17
    mlflow.tensorflow.autolog(log_every_epoch=True)

    model = create_tf_keras_model()
    with mlflow.start_run() as run:
        model.fit(
            random_train_data,
            random_one_hot_labels,
            epochs=num_training_epochs,
            initial_epoch=0,
        )

    client = MlflowClient()
    run_metrics = client.get_run(run.info.run_id).data.metrics
    assert "accuracy" in run_metrics
    all_epoch_acc = client.get_metric_history(run.info.run_id, "accuracy")
    assert len(all_epoch_acc) == num_training_epochs


def test_tf_keras_autolog_logs_metrics_for_single_epoch_training(
    random_train_data, random_one_hot_labels
):
    """
    tf.Keras exhibits inconsistent epoch indexing behavior in comparison with other
    TF2 APIs (e.g., tf.Estimator). tf.Keras uses zero-indexing for epochs,
    while other APIs use one-indexing. Accordingly, this test verifies that metrics are
    produced in the boundary case where a model is trained for a single epoch, ensuring
    that we don't miss the zero index in the tf.Keras case.
    """
    mlflow.tensorflow.autolog()

    model = create_tf_keras_model()
    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, epochs=1)

    client = MlflowClient()
    run_metrics = client.get_run(run.info.run_id).data.metrics
    assert "accuracy" in run_metrics
    assert "loss" in run_metrics


def test_tf_keras_autolog_names_positional_parameters_correctly(
    random_train_data, random_one_hot_labels
):
    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    with mlflow.start_run():
        # Pass `batch_size` as a positional argument for testing purposes
        model.fit(data, labels, 8, epochs=10, steps_per_epoch=1)
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    run_info = client.get_run(run_id)
    assert run_info.data.params.get("batch_size") == "8"


@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_model_can_load_from_artifact(tf_keras_random_data_run, random_train_data):
    run, _ = tf_keras_random_data_run

    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = (x.path for x in artifacts)
    assert "model" in artifacts
    assert "tensorboard_logs" in artifacts
    model = mlflow.tensorflow.load_model("runs:/" + run.info.run_id + "/model")
    model.predict(random_train_data)


def get_tf_keras_random_data_run_with_callback(
    random_train_data,
    random_one_hot_labels,
    callback,
    restore_weights,
    patience,
    initial_epoch,
    log_models,
):
    mlflow.tensorflow.autolog(every_n_iter=1, log_models=log_models)

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

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_train_end(self, logs=None):
                pass

        callback = CustomCallback()

    history = model.fit(
        data, labels, epochs=initial_epoch + 10, callbacks=[callback], initial_epoch=initial_epoch
    )

    client = MlflowClient()
    return client.get_run(client.search_runs(["0"])[0].info.run_id), history, callback


@pytest.fixture
def tf_keras_random_data_run_with_callback(
    random_train_data,
    random_one_hot_labels,
    callback,
    restore_weights,
    patience,
    initial_epoch,
    log_models,
):
    return get_tf_keras_random_data_run_with_callback(
        random_train_data,
        random_one_hot_labels,
        callback,
        restore_weights,
        patience,
        initial_epoch,
        log_models=log_models,
    )


@pytest.mark.parametrize("log_models", [True, False])
@pytest.mark.parametrize("restore_weights", [True])
@pytest.mark.parametrize("callback", ["early"])
@pytest.mark.parametrize("patience", [0, 1, 5])
@pytest.mark.parametrize("initial_epoch", [0, 10])
def test_tf_keras_autolog_early_stop_logs(tf_keras_random_data_run_with_callback, initial_epoch):
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
    restored_epoch = int(metrics["restored_epoch"])
    # In this test, the best epoch is always the first epoch because the early stopping callback
    # never observes a loss improvement due to an extremely large `min_delta` value
    assert restored_epoch == initial_epoch
    assert "loss" in history.history
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check that MLflow has logged the metrics of the "best" model, in addition to per-epoch metrics
    loss = history.history["loss"]
    assert len(metric_history) == len(loss) + 1
    steps, values = map(list, zip(*[(m.step, m.value) for m in metric_history]))
    # Check that MLflow has logged the correct steps
    assert steps == [*history.epoch, callback.stopped_epoch + 1]
    # Check that MLflow has logged the correct metric values
    np.testing.assert_allclose(values, [*loss, callback.best])

    artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
    assert "tensorboard_logs" in artifacts


@pytest.mark.parametrize("log_models", [False])
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
    assert "stopped_epoch" not in metrics
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("log_models", [False])
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
    assert "restored_epoch" not in metrics
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == callback.patience + 1
    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("log_models", [False])
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
    assert "loss" in history.history
    num_of_epochs = len(history.history["loss"])
    client = MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.parametrize("positional", [True, False])
def test_tf_keras_autolog_does_not_mutate_original_callbacks_list(
    tmp_path, random_train_data, random_one_hot_labels, positional
):
    """
    TensorFlow autologging passes new callbacks to the `fit()` / `fit_generator()` function. If
    preexisting user-defined callbacks already exist, these new callbacks are added to the
    user-specified ones. This test verifies that the new callbacks are added to the without
    permanently mutating the original list of callbacks.
    """
    mlflow.tensorflow.autolog()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tmp_path)
    callbacks = [tensorboard_callback]

    model = create_tf_keras_model()
    data = random_train_data
    labels = random_one_hot_labels

    if positional:
        model.fit(data, labels, None, 10, 1, callbacks)
    else:
        model.fit(data, labels, epochs=10, callbacks=callbacks)

    assert len(callbacks) == 1
    assert callbacks == [tensorboard_callback]


def test_tf_keras_autolog_does_not_delete_logging_directory_for_tensorboard_callback(
    tmp_path, random_train_data, random_one_hot_labels
):
    tensorboard_callback_logging_dir_path = str(tmp_path.joinpath("tb_logs"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        tensorboard_callback_logging_dir_path, histogram_freq=0
    )

    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()
    model.fit(data, labels, epochs=10, callbacks=[tensorboard_callback])

    assert os.path.exists(tensorboard_callback_logging_dir_path)


def test_tf_keras_autolog_logs_to_and_deletes_temporary_directory_when_tensorboard_callback_absent(
    tmp_path, random_train_data, random_one_hot_labels
):
    from mlflow.tensorflow import _TensorBoardLogDir

    mlflow.tensorflow.autolog()

    mock_log_dir_inst = _TensorBoardLogDir(
        location=str(tmp_path.joinpath("tb_logging")), is_temp=True
    )
    with patch("mlflow.tensorflow._TensorBoardLogDir", autospec=True) as mock_log_dir_class:
        mock_log_dir_class.return_value = mock_log_dir_inst

        data = random_train_data
        labels = random_one_hot_labels

        model = create_tf_keras_model()
        model.fit(data, labels, epochs=10)

        assert not os.path.exists(mock_log_dir_inst.location)


def get_text_vec_model(train_samples):
    # Taken from: https://github.com/mlflow/mlflow/issues/3910

    try:
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    except ModuleNotFoundError:
        from tensorflow.keras.layers import TextVectorization

    VOCAB_SIZE = 10
    SEQUENCE_LENGTH = 16
    EMBEDDING_DIM = 16

    vectorizer_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer_layer.adapt(train_samples)
    model = tf.keras.Sequential(
        [
            vectorizer_layer,
            tf.keras.layers.Embedding(
                VOCAB_SIZE,
                EMBEDDING_DIM,
                name="embedding",
                mask_zero=True,
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="tanh"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.3.0"),
    reason=(
        "Deserializing a model with `TextVectorization` and `Embedding` "
        "fails in tensorflow < 2.3.0. See this issue: "
        "https://github.com/tensorflow/tensorflow/issues/38250"
    ),
)
def test_autolog_text_vec_model(tmp_path):
    """
    Verifies autolog successfully saves a model that can't be saved in the H5 format
    """
    mlflow.tensorflow.autolog()

    train_samples = np.array(["this is an example", "another example"], dtype=object)
    train_labels = np.array([0.4, 0.2])
    model = get_text_vec_model(train_samples)

    with mlflow.start_run() as run:
        model.fit(train_samples, train_labels, epochs=1)

    loaded_model = mlflow.tensorflow.load_model("runs:/" + run.info.run_id + "/model")
    np.testing.assert_array_equal(loaded_model.predict(train_samples), model.predict(train_samples))


def test_tf_keras_model_autolog_registering_model(random_train_data, random_one_hot_labels):
    registered_model_name = "test_autolog_registered_model"
    mlflow.tensorflow.autolog(registered_model_name=registered_model_name)
    with mlflow.start_run():
        model = create_tf_keras_model()
        model.fit(random_train_data, random_one_hot_labels, epochs=10)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_fluent_autolog_with_tf_keras_logs_expected_content(
    random_train_data, random_one_hot_labels
):
    """
    Guards against previously-exhibited issues where using the fluent `mlflow.autolog()` API with
    `tf.keras` Models did not work due to conflicting patches set by both the
    `mlflow.tensorflow.autolog()` and the `mlflow.keras.autolog()` APIs.
    """
    mlflow.autolog()

    model = create_tf_keras_model()

    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, epochs=10)

    client = MlflowClient()
    run_data = client.get_run(run.info.run_id).data
    assert "accuracy" in run_data.metrics
    assert "epochs" in run_data.params

    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = (x.path for x in artifacts)
    assert "model" in artifacts


def test_callback_is_picklable():
    cb = MlflowCallback()
    pickle.dumps(cb)

    tb = _TensorBoard()
    pickle.dumps(tb)


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.1.0"), reason="This test requires tensorflow >= 2.1.0"
)
def test_tf_keras_autolog_distributed_training(random_train_data, random_one_hot_labels):
    # Ref: https://www.tensorflow.org/tutorials/distribute/keras
    mlflow.tensorflow.autolog()

    with tf.distribute.MirroredStrategy().scope():
        model = create_tf_keras_model()
    fit_params = {"epochs": 10, "batch_size": 10}
    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, **fit_params)
    client = MlflowClient()
    assert client.get_run(run.info.run_id).data.params.keys() >= fit_params.keys()


def test_import_tensorflow_with_fluent_autolog_enables_tensorflow_autologging():
    mlflow.autolog()

    import tensorflow  # noqa: F401

    assert not autologging_is_disabled(mlflow.tensorflow.FLAVOR_NAME)


def _assert_autolog_infers_model_signature_correctly(run, input_sig_spec, output_sig_spec):
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run.info.run_id, "model")]
    ml_model_filename = "MLmodel"
    assert str(os.path.join("model", ml_model_filename)) in artifacts
    ml_model_path = os.path.join(artifacts_dir, "model", ml_model_filename)
    with open(ml_model_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        assert data is not None
        assert "signature" in data
        signature = data["signature"]
        assert signature is not None
        assert "inputs" in signature
        assert "outputs" in signature
        assert json.loads(signature["inputs"]) == input_sig_spec
        assert json.loads(signature["outputs"]) == output_sig_spec


def _assert_keras_autolog_input_example_load_and_predict_with_nparray(run, random_train_data):
    model_path = os.path.join(run.info.artifact_uri, "model")
    model_conf = Model.load(os.path.join(model_path, "MLmodel"))
    input_example = _read_example(model_conf, model_path)
    np.testing.assert_array_almost_equal(input_example, random_train_data[:5])
    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))
    pyfunc_model.predict(input_example)


def test_keras_autolog_input_example_load_and_predict_with_nparray(
    random_train_data, random_one_hot_labels
):
    mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True)
    initial_model = create_tf_keras_model()
    with mlflow.start_run() as run:
        initial_model.fit(random_train_data, random_one_hot_labels)
        _assert_keras_autolog_input_example_load_and_predict_with_nparray(run, random_train_data)


def test_keras_autolog_infers_model_signature_correctly_with_nparray(
    random_train_data, random_one_hot_labels
):
    mlflow.tensorflow.autolog(log_model_signatures=True)
    initial_model = create_tf_keras_model()
    with mlflow.start_run() as run:
        initial_model.fit(random_train_data, random_one_hot_labels)
        _assert_autolog_infers_model_signature_correctly(
            run,
            [{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 4]}}],
            [{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}}],
        )


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.1.0"),
    reason="tf.data.Dataset inputs are unsupported for input example logging in TensorFlow < 2.1.0",
)
def test_keras_autolog_input_example_load_and_predict_with_tf_dataset(fashion_mnist_tf_dataset):
    mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True)
    fashion_mnist_model = _create_fashion_mnist_model()
    with mlflow.start_run() as run:
        fashion_mnist_model.fit(fashion_mnist_tf_dataset)
        model_path = os.path.join(run.info.artifact_uri, "model")
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        input_example = _read_example(model_conf, model_path)
        pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))
        pyfunc_model.predict(input_example)


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.1.0"),
    reason="tf.data.Dataset inputs are unsupported for signature logging in TensorFlow < 2.1.0",
)
def test_keras_autolog_infers_model_signature_correctly_with_tf_dataset(fashion_mnist_tf_dataset):
    mlflow.tensorflow.autolog(log_model_signatures=True)
    fashion_mnist_model = _create_fashion_mnist_model()
    with mlflow.start_run() as run:
        fashion_mnist_model.fit(fashion_mnist_tf_dataset)
        _assert_autolog_infers_model_signature_correctly(
            run,
            [{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 28, 28]}}],
            [{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 10]}}],
        )


def test_keras_autolog_input_example_load_and_predict_with_dict(
    random_train_dict_mapping, random_one_hot_labels
):
    mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True)
    model = _create_model_for_dict_mapping()
    with mlflow.start_run() as run:
        model.fit(random_train_dict_mapping, random_one_hot_labels)
        model_path = os.path.join(run.info.artifact_uri, "model")
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        input_example = _read_example(model_conf, model_path)
        for k, v in random_train_dict_mapping.items():
            np.testing.assert_array_almost_equal(input_example[k], np.take(v, range(0, 5)))
        pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))
        pyfunc_model.predict(input_example)


def test_keras_autolog_infers_model_signature_correctly_with_dict(
    random_train_dict_mapping, random_one_hot_labels
):
    mlflow.tensorflow.autolog(log_model_signatures=True)
    model = _create_model_for_dict_mapping()
    with mlflow.start_run() as run:
        model.fit(random_train_dict_mapping, random_one_hot_labels)
        _assert_autolog_infers_model_signature_correctly(
            run,
            [
                {"name": "a", "type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}},
                {"name": "b", "type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}},
                {"name": "c", "type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}},
                {"name": "d", "type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1]}},
            ],
            [{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}}],
        )


def test_keras_autolog_input_example_load_and_predict_with_keras_sequence(keras_data_gen_sequence):
    mlflow.tensorflow.autolog(log_input_examples=True, log_model_signatures=True)
    model = create_tf_keras_model()
    with mlflow.start_run() as run:
        model.fit(keras_data_gen_sequence)
        _assert_keras_autolog_input_example_load_and_predict_with_nparray(
            run, keras_data_gen_sequence[:][0][:5]
        )


def test_keras_autolog_infers_model_signature_correctly_with_keras_sequence(
    keras_data_gen_sequence,
):
    mlflow.tensorflow.autolog(log_model_signatures=True)
    initial_model = create_tf_keras_model()
    with mlflow.start_run() as run:
        initial_model.fit(keras_data_gen_sequence)
        _assert_autolog_infers_model_signature_correctly(
            run,
            [{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 4]}}],
            [{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}}],
        )


def test_keras_autolog_load_saved_hdf5_model(keras_data_gen_sequence):
    mlflow.tensorflow.autolog(keras_model_kwargs={"save_format": "h5"})
    model = create_tf_keras_model()
    with mlflow.start_run() as run:
        model.fit(keras_data_gen_sequence)
        mlflow.tensorflow.load_model(f"runs:/{run.info.run_id}/model")
        assert Path(run.info.artifact_uri, "model", "data", "model.h5").exists()


def test_keras_autolog_logs_model_signature_by_default(keras_data_gen_sequence):
    mlflow.autolog()
    initial_model = create_tf_keras_model()
    initial_model.fit(keras_data_gen_sequence)

    mlmodel_path = mlflow.artifacts.download_artifacts(
        f"runs:/{mlflow.last_active_run().info.run_id}/model/MLmodel"
    )
    with open(mlmodel_path) as f:
        mlmodel_contents = yaml.safe_load(f)
    assert "signature" in mlmodel_contents.keys()
    signature = mlmodel_contents["signature"]
    assert signature is not None
    assert "inputs" in signature
    assert "outputs" in signature
    assert json.loads(signature["inputs"]) == [
        {"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 4]}}
    ]
    assert json.loads(signature["outputs"]) == [
        {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}}
    ]


def test_extract_tf_keras_input_example_unsupported_type_returns_None():
    from mlflow.tensorflow.autologging import extract_tf_keras_input_example

    extracted_data = extract_tf_keras_input_example([1, 2, 4, 5])
    assert extracted_data is None, (
        "Keras input data extraction function should have "
        "returned None as input type is not supported."
    )


def test_extract_input_example_from_tf_input_fn_unsupported_type_returns_None():
    from mlflow.tensorflow.autologging import extract_tf_keras_input_example

    extracted_data = extract_tf_keras_input_example(lambda: [1, 2, 4, 5])
    assert extracted_data is None, (
        "Tensorflow's input_fn training data extraction should have"
        " returned None as input type is not supported."
    )


@pytest.mark.skipif(
    Version(tf.__version__) < Version("2.6.0"),
    reason=("TensorFlow only has a hard dependency on Keras in version >= 2.6.0"),
)
def test_import_keras_model_trigger_import_tensorflow():
    # This test is for guarding importing keras model will trigger importing tensorflow
    # Because in Keras>=2.6, the keras autologging patching is installed by
    # `mlflow.tensorflow.autolog`, suppose user enable autolog by `mlflow.autolog()`,
    # and then import keras, if keras does not trigger importing tensorflow,
    # then the keras autologging patching cannot be installed.
    py_executable = sys.executable
    _exec_cmd(
        [
            py_executable,
            "-c",
            "from keras import Model; import sys; assert 'tensorflow' in sys.modules",
        ]
    )


def test_autolog_throw_error_on_explicit_mlflow_callback(keras_data_gen_sequence):
    mlflow.tensorflow.autolog()

    model = create_tf_keras_model()
    with mlflow.start_run() as run:
        with pytest.raises(MlflowException, match="MLflow autologging must be turned off*"):
            model.fit(keras_data_gen_sequence, callbacks=[MlflowCallback(run)])


def test_autolog_correct_logging_frequency(random_train_data, random_one_hot_labels):
    logging_freq = 5
    num_epochs = 2
    batch_size = 10
    mlflow.tensorflow.autolog(log_every_epoch=False, log_every_n_steps=logging_freq)
    initial_model = create_tf_keras_model()
    with mlflow.start_run() as run:
        initial_model.fit(
            random_train_data,
            random_one_hot_labels,
            batch_size=batch_size,
            epochs=num_epochs,
        )

    client = MlflowClient()
    loss_history = client.get_metric_history(run.info.run_id, "loss")
    assert len(loss_history) == num_epochs * (len(random_train_data) // batch_size) // logging_freq


def test_automatic_checkpoint_per_epoch_callback(random_train_data, random_one_hot_labels):
    mlflow.tensorflow.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq="epoch",
    )

    model = create_tf_keras_model()

    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, epochs=1)
    run_id = run.info.run_id

    logged_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/epoch_0/checkpoint_metrics.json"
    )
    assert set(logged_metrics) == {"epoch", "loss", "accuracy", "global_step"}
    assert logged_metrics["epoch"] == 0
    assert logged_metrics["global_step"] == 5

    pred_result = model.predict(random_train_data)
    pred_result2 = load_checkpoint(run_id=run_id).predict(random_train_data)
    np.testing.assert_array_almost_equal(pred_result, pred_result2)

    pred_result3 = load_checkpoint(run_id=run_id, epoch=0).predict(random_train_data)
    np.testing.assert_array_almost_equal(pred_result, pred_result3)


def test_automatic_checkpoint_per_epoch_save_weight_only_callback(
    random_train_data, random_one_hot_labels
):
    mlflow.tensorflow.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq="epoch",
    )

    model = create_tf_keras_model()

    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, epochs=1)
    run_id = run.info.run_id

    logged_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/epoch_0/checkpoint_metrics.json"
    )
    assert set(logged_metrics) == {"epoch", "loss", "accuracy", "global_step"}
    assert logged_metrics["epoch"] == 0
    assert logged_metrics["global_step"] == 5

    model2 = create_tf_keras_model()
    pred_result = model.predict(random_train_data)
    pred_result2 = load_checkpoint(model=model2, run_id=run_id).predict(random_train_data)
    np.testing.assert_array_almost_equal(pred_result, pred_result2)


def test_automatic_checkpoint_per_3_steps_callback(random_train_data, random_one_hot_labels):
    mlflow.tensorflow.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq=3,
    )
    model = create_tf_keras_model()

    with mlflow.start_run() as run:
        model.fit(random_train_data, random_one_hot_labels, epochs=1)
    run_id = run.info.run_id
    logged_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/global_step_3/checkpoint_metrics.json"
    )
    assert set(logged_metrics) == {"epoch", "loss", "accuracy", "global_step"}
    assert logged_metrics["epoch"] == 0
    assert logged_metrics["global_step"] == 3

    assert isinstance(load_checkpoint(run_id=run_id), tf.keras.Sequential)
    assert isinstance(load_checkpoint(run_id=run_id, global_step=3), tf.keras.Sequential)


def test_automatic_checkpoint_per_3_steps_save_best_only_callback(
    random_train_data, random_one_hot_labels
):
    mlflow.tensorflow.autolog(
        checkpoint=True,
        checkpoint_monitor="loss",
        checkpoint_mode="min",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq=3,
    )

    model = create_tf_keras_model()

    with mlflow.start_run() as run:
        model.fit(
            random_train_data,
            random_one_hot_labels,
            epochs=1,
        )
    run_id = run.info.run_id
    logged_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json"
    )
    assert set(logged_metrics) == {"epoch", "loss", "accuracy", "global_step"}
    assert logged_metrics["epoch"] == 0
    assert logged_metrics["global_step"] == 3

    assert isinstance(load_checkpoint(run_id=run_id), tf.keras.Sequential)

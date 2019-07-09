# pep8: disable=E501

from __future__ import print_function

import collections
import shutil
import pytest
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import layers

import mlflow
import mlflow.tensorflow

SavedModelInfo = collections.namedtuple(
        "SavedModelInfo",
        ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"])


@pytest.fixture
def tf_keras_random_data():
    mlflow.tensorflow.autolog(metrics_every_n_steps=2)
    def random_one_hot_labels(shape):
        n, n_class = shape
        classes = np.random.randint(0, n_class, n)
        labels = np.zeros((n, n_class))
        labels[np.arange(n), classes] = 1
        return labels

    data = np.random.random((1000, 32))
    labels = random_one_hot_labels((1000, 10))

    model = tf.keras.Sequential()

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    run = mlflow.start_run()
    model.fit(data, labels, epochs=10)
    mlflow.end_run()
    return run.info.run_id


@pytest.fixture
def get_tf_keras_random_data_run_instance(tf_keras_random_data):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(tf_keras_random_data)
    return run


@pytest.mark.small
def test_tf_keras_autolog_logs_epoch_acc(get_tf_keras_random_data_run_instance):
    assert 'epoch_acc' in get_tf_keras_random_data_run_instance.data.metrics


@pytest.mark.small
def test_tf_keras_autolog_logs_epoch_loss(get_tf_keras_random_data_run_instance):
    assert 'epoch_loss' in get_tf_keras_random_data_run_instance.data.metrics


@pytest.mark.small
def test_tf_keras_autolog_logs_optimizer_name(get_tf_keras_random_data_run_instance):
    assert 'optimizer_name' in get_tf_keras_random_data_run_instance.data.params


@pytest.mark.small
def test_tf_keras_autolog_logs_model_summary(get_tf_keras_random_data_run_instance):
    assert 'summary' in get_tf_keras_random_data_run_instance.data.tags
    assert get_tf_keras_random_data_run_instance.data.tags['summary'] is not ''


@pytest.mark.large
def test_tf_keras_autolog_saves_model(get_tf_keras_random_data_run_instance):
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(get_tf_keras_random_data_run_instance.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    assert 'tensorboard_logs' in artifacts


@pytest.fixture
def tf_core_random_tensors():
    mlflow.tensorflow.autolog(metrics_every_n_steps=1)
    run = mlflow.start_run()
    sess = tf.Session()

    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)  # also tf.float32 implicitly
    total = a + b
    tf.summary.scalar('a', a)
    tf.summary.scalar('b', b)
    merged = tf.summary.merge_all()
    dir = tempfile.mkdtemp()
    writer = tf.summary.FileWriter(dir, sess.graph)
    with sess.as_default():
        summary, _ = sess.run([merged, total])

    writer.add_summary(summary)

    shutil.rmtree(dir)

    writer.close()

    mlflow.end_run()

    return run.info.run_id


@pytest.fixture
def get_tf_core_random_tensors_run_instance(tf_core_random_tensors):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(tf_core_random_tensors)
    return run


@pytest.mark.small
def test_tf_core_autolog_logs_scalars(get_tf_core_random_tensors_run_instance):
    assert 'a' in get_tf_core_random_tensors_run_instance.data.metrics
    assert get_tf_core_random_tensors_run_instance.data.metrics['a'] == 3.0
    assert 'b' in get_tf_core_random_tensors_run_instance.data.metrics
    assert get_tf_core_random_tensors_run_instance.data.metrics['b'] == 4.0


@pytest.fixture
def tf_estimator_random_data():
    mlflow.tensorflow.autolog(metrics_every_n_steps=1)
    run = mlflow.start_run()
    dir = tempfile.mkdtemp()
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    train_path = tf.keras.utils.get_file(
        "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file(
        "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    train_y = train.pop('Species')
    test_y = test.pop('Species')

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
        model_dir=dir)

    classifier.train(
        input_fn=lambda: input_fn(train, train_y, training=True),
        steps=5000)
    classifier.export_saved_model(dir, receiver_fn)
    mlflow.end_run()
    shutil.rmtree(dir)
    return run.info.run_id


@pytest.fixture
def get_tf_estimator_random_data_run_instance(tf_estimator_random_data):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(tf_estimator_random_data)
    return run


@pytest.mark.small
def test_tf_estimator_autolog_logs_metrics(get_tf_estimator_random_data_run_instance):
    metrics = get_tf_estimator_random_data_run_instance.data.metrics
    assert len(metrics) > 0

@pytest.mark.large
def test_tf_estimator_autolog_saves_model(get_tf_estimator_random_data_run_instance):
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(get_tf_estimator_random_data_run_instance.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts









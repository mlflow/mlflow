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
import mlflow.keras

import os

SavedModelInfo = collections.namedtuple(
    "SavedModelInfo",
    ["path", "meta_graph_tags", "signature_def_key", "inference_df", "expected_results_df"])

client = mlflow.tracking.MlflowClient()


@pytest.fixture
def random_train_data():
    return np.random.random((1000, 32))


@pytest.fixture
def tf_keras_random_data_run(random_train_data):
    mlflow.tensorflow.autolog(every_n_iter=5)

    def random_one_hot_labels(shape):
        n, n_class = shape
        classes = np.random.randint(0, n_class, n)
        labels = np.zeros((n, n_class))
        labels[np.arange(n), classes] = 1
        return labels

    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels((1000, 10))

        model = tf.keras.Sequential()

        model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=10)

    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_keras_autolog_logs_expected_data(tf_keras_random_data_run):
    data = tf_keras_random_data_run.data
    assert 'accuracy' in data.metrics
    assert 'loss' in data.metrics
    assert 'opt_name' in data.params
    assert data.params['opt_name'] == 'Adam'
    assert 'opt_learning_rate' in data.params
    assert 'opt_decay' in data.params
    assert 'opt_beta_1' in data.params
    assert 'opt_beta_2' in data.params
    assert 'opt_epsilon' in data.params
    assert 'opt_amsgrad' in data.params
    assert data.params['opt_amsgrad'] == 'False'
    assert 'summary' in tf_keras_random_data_run.data.tags
    assert 'Total params: 6,922' in tf_keras_random_data_run.data.tags['summary']
    all_epoch_acc = client.get_metric_history(tf_keras_random_data_run.info.run_id, 'accuracy')
    assert all((x.step - 1) % 5 == 0 for x in all_epoch_acc)


@pytest.mark.large
def test_tf_keras_autolog_model_can_load_from_artifact(tf_keras_random_data_run, random_train_data):
    artifacts = client.list_artifacts(tf_keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    assert 'tensorboard_logs' in artifacts
    model = mlflow.keras.load_model("runs:/" + tf_keras_random_data_run.info.run_id +
                                    "/model")
    model.predict(random_train_data)


@pytest.fixture
def tf_estimator_random_data_run():
    mlflow.tensorflow.autolog()
    with mlflow.start_run() as run:
        dir = tempfile.mkdtemp()
        CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        SPECIES = ['Setosa', 'Versicolor', 'Virginica']

        train = pd.read_csv(os.path.join(os.path.dirname(__file__), "iris_training.csv"),
                            names=CSV_COLUMN_NAMES, header=0)
        test = pd.read_csv(os.path.join(os.path.dirname(__file__), "iris_test.csv"),
                           names=CSV_COLUMN_NAMES, header=0)

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
            feature_spec[feature] = tf.Variable([], dtype=tf.float64, name=feature)

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
            steps=500)
        classifier.export_saved_model(dir, receiver_fn)

    shutil.rmtree(dir)
    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_estimator_autolog_logs_metrics(tf_estimator_random_data_run):
    assert 'loss' in tf_estimator_random_data_run.data.metrics
    metrics = client.get_metric_history(tf_estimator_random_data_run.info.run_id, 'loss')
    assert all((x.step-1) % 100 == 0 for x in metrics)


@pytest.mark.large
def test_tf_estimator_autolog_model_can_load_from_artifact(tf_estimator_random_data_run):
    artifacts = client.list_artifacts(tf_estimator_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    model = mlflow.tensorflow.load_model("runs:/" + tf_estimator_random_data_run.info.run_id +
                                         "/model")


@pytest.fixture
def duplicate_autolog_tf_estimator_run():
    mlflow.tensorflow.autolog(every_n_iter=23)  # 23 is prime; no false positives in test
    run = tf_estimator_random_data_run()
    return run  # should be autologged every 4 steps


@pytest.mark.large
def test_duplicate_autolog_second_overrides(duplicate_autolog_tf_estimator_run):
    metrics = client.get_metric_history(duplicate_autolog_tf_estimator_run.info.run_id, 'loss')
    assert all((x.step - 1) % 4 == 0 for x in metrics)

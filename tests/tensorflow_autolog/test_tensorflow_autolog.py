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
def random_one_hot_labels():
    n, n_class = (1000, 10)
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

    model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_tf_keras_autolog_ends_auto_created_run(random_train_data, random_one_hot_labels,
                                                fit_variant):
    mlflow.tensorflow.autolog()

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    model.fit(data, labels, epochs=10)

    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_tf_keras_autolog_persists_manually_created_run(random_train_data, random_one_hot_labels,
                                                        fit_variant):
    mlflow.tensorflow.autolog()
    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels

        model = create_tf_keras_model()

        model.fit(data, labels, epochs=10)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def tf_keras_random_data_run(random_train_data, random_one_hot_labels, manual_run, fit_variant):
    mlflow.tensorflow.autolog(every_n_iter=5)

    data = random_train_data
    labels = random_one_hot_labels

    model = create_tf_keras_model()

    if fit_variant == 'fit_generator':
        def generator():
            while True:
                yield data, labels
        model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
    else:
        model.fit(data, labels, epochs=10)

    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_tf_keras_autolog_logs_expected_data(tf_keras_random_data_run):
    data = tf_keras_random_data_run.data

    assert 'epoch_acc' in data.metrics
    assert 'epoch_loss' in data.metrics
    assert 'optimizer_name' in data.params
    assert data.params['optimizer_name'] == 'AdamOptimizer'
    assert 'model_summary' in data.tags
    assert 'Total params: 6,922' in data.tags['model_summary']
    all_epoch_acc = client.get_metric_history(tf_keras_random_data_run.info.run_id, 'epoch_acc')
    assert all((x.step - 1) % 5 == 0 for x in all_epoch_acc)
    artifacts = client.list_artifacts(tf_keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model_summary.txt' in artifacts


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_tf_keras_autolog_model_can_load_from_artifact(tf_keras_random_data_run, random_train_data):
    artifacts = client.list_artifacts(tf_keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    assert 'tensorboard_logs' in artifacts
    model = mlflow.keras.load_model("runs:/" + tf_keras_random_data_run.info.run_id +
                                    "/model")
    model.predict(random_train_data)


@pytest.fixture
def tf_core_random_tensors():
    mlflow.tensorflow.autolog(every_n_iter=4)
    with mlflow.start_run() as run:
        sess = tf.Session()
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)
        total = a + b
        tf.summary.scalar('a', a)
        tf.summary.scalar('b', b)
        merged = tf.summary.merge_all()
        dir = tempfile.mkdtemp()
        writer = tf.summary.FileWriter(dir, sess.graph)
        with sess.as_default():
            for i in range(40):
                summary, _ = sess.run([merged, total])
                writer.add_summary(summary, global_step=i)
        shutil.rmtree(dir)
        writer.close()
        sess.close()

    return client.get_run(run.info.run_id)


@pytest.mark.large
def test_tf_core_autolog_logs_scalars(tf_core_random_tensors):
    assert 'a' in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics['a'] == 3.0
    assert 'b' in tf_core_random_tensors.data.metrics
    assert tf_core_random_tensors.data.metrics['b'] == 4.0
    all_a = client.get_metric_history(tf_core_random_tensors.info.run_id, 'a')
    assert all((x.step - 1) % 4 == 0 for x in all_a)
    assert mlflow.active_run() is None


def create_tf_estimator_model(dir, export):
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
        steps=500)
    if export:
        classifier.export_saved_model(dir, receiver_fn)


@pytest.mark.large
@pytest.mark.parametrize('export', [True, False])
def test_tf_estimator_autolog_ends_auto_created_run(tmpdir, export):
    dir = tmpdir.mkdir("test")
    mlflow.tensorflow.autolog()
    create_tf_estimator_model(str(dir), export)
    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize('export', [True, False])
def test_tf_estimator_autolog_persists_manually_created_run(tmpdir, export):
    dir = tmpdir.mkdir("test")
    with mlflow.start_run() as run:
        create_tf_estimator_model(str(dir), export)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def tf_estimator_random_data_run(tmpdir, manual_run, export):
    dir = tmpdir.mkdir("test")
    mlflow.tensorflow.autolog()
    create_tf_estimator_model(str(dir), export)
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize('export', [True, False])
def test_tf_estimator_autolog_logs_metrics(tf_estimator_random_data_run):
    assert 'loss' in tf_estimator_random_data_run.data.metrics
    assert 'steps' in tf_estimator_random_data_run.data.params
    metrics = client.get_metric_history(tf_estimator_random_data_run.info.run_id, 'loss')
    assert all((x.step-1) % 100 == 0 for x in metrics)


@pytest.mark.large
@pytest.mark.parametrize('export', [True])
def test_tf_estimator_autolog_model_can_load_from_artifact(tf_estimator_random_data_run):
    artifacts = client.list_artifacts(tf_estimator_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    session = tf.Session()
    model = mlflow.tensorflow.load_model("runs:/" + tf_estimator_random_data_run.info.run_id +
                                         "/model", session)


@pytest.fixture
def duplicate_autolog_tf_estimator_run(tmpdir, manual_run, export):
    mlflow.tensorflow.autolog(every_n_iter=23)  # 23 is prime; no false positives in test
    run = tf_estimator_random_data_run(tmpdir, manual_run, export)
    return run  # should be autologged every 4 steps


@pytest.mark.large
@pytest.mark.parametrize('export', [True, False])
def test_duplicate_autolog_second_overrides(duplicate_autolog_tf_estimator_run):
    metrics = client.get_metric_history(duplicate_autolog_tf_estimator_run.info.run_id, 'loss')
    assert all((x.step - 1) % 4 == 0 for x in metrics)

# pep8: disable=E501

from __future__ import print_function

import collections
import os
import shutil
import pytest
import yaml
import json

import numpy as np
import pandas as pd
import pandas.testing
import sklearn.datasets as datasets
import tensorflow as tf
from tensorflow.python.keras import layers

import mlflow
import mlflow.tensorflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow.exceptions import MlflowException
from mlflow import pyfunc
from mlflow.store.s3_artifact_repo import S3ArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import score_model_in_sagemaker_docker_container
from tests.helper_functions import set_boto_credentials  # pylint: disable=unused-import
from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

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
def get_run_instance(tf_keras_random_data):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(tf_keras_random_data)
    return run

@pytest.mark.small
def test_tf_keras_autolog_logs_epoch_acc(get_run_instance):
    assert 'epoch_acc' in get_run_instance.data.metrics

@pytest.mark.small
def test_tf_keras_autolog_logs_epoch_loss(get_run_instance):
    assert 'epoch_loss' in get_run_instance.data.metrics



import pytest
from unittest import mock

import mlflow
from mlflow import tracking
from mlflow.tracking.fluent import start_run
from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracking.metric_value_conversion_utils import *

import tensorflow as tf


def test_reraised_value_errors():
    multi_item_tf_tensor = tf.random.uniform([2, 2])

    with pytest.raises(MlflowException) as e:
        convert_metric_value_to_float_if_possible(multi_item_tf_tensor)

    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_convert_metric_value_to_float():
    tf_tensor_val = tf.random.uniform([])
    assert float(tf_tensor_val.numpy()) == convert_metric_value_to_float_if_possible(tf_tensor_val)


def test_log_tf_tensor_as_metric():
    tf_tensor_val = tf.random.uniform([])
    tf_tensor_float_val = float(tf_tensor_val.numpy())

    with start_run() as run:
        mlflow.log_metric("name_tf", tf_tensor_val)

    finished_run = tracking.MlflowClient().get_run(run.info.run_id)
    expected_pairs = {"name_tf": tf_tensor_float_val}

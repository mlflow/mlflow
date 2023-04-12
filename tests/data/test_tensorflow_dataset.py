import json
import os

import pandas as pd
import numpy as np
import pytest

import mlflow.data
from mlflow.data.tensorflow_dataset import TensorflowDataset
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema

import tensorflow as tf


def test_from_tensorflow_dataset_constructs_expected_dataset():
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    mlflow_ds = mlflow.data.from_tensorflow(tf_dataset, source="test")
    assert isinstance(mlflow_ds, TensorflowDataset)
    assert mlflow_ds.data == tf_dataset
    assert mlflow_ds.schema == _infer_schema(next(tf_dataset.as_numpy_iterator()))
    assert mlflow_ds.profile == {
        "num_rows": len(tf_dataset),
        "num_elements": tf_dataset.cardinality().numpy(),
    }

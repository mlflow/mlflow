import functools
import inspect
from unittest import mock
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.models import Model
from mlflow.models.signature import infer_signature
from mlflow.models.utils import _read_example
from mlflow.entities import RunStatus
from mlflow.sklearn.utils import (
    _is_supported_version,
    _is_metric_supported,
    _is_plotting_supported,
    _get_arg_names,
    _truncate_dict,
)
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_AUTOLOGGING
from mlflow.utils.validation import (
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)

from tests.autologging.fixtures import test_mode_off


from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from mlflow.pyspark.ml import _get_estimator_param_map

MODEL_DIR = "model"


def truncate_dict(d):
    return _truncate_dict(d, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)


def get_expected_class_tags(model):
    return {
        'estimator_name': model.__class__.__name__,
        'estimator_class': model.__class__.__module__ + "." + model.__class__.__name__,
    }


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return data.params, data.metrics, tags, artifacts


def load_model_by_run_id(run_id):
    return mlflow.sklearn.load_model("runs:/{}/{}".format(run_id, MODEL_DIR))


def get_training_dataset(spark_session):
    return spark_session.createDataFrame(
        [(1.0, 2.0, Vectors.dense(1.0)),
         (0.0, 2.0, Vectors.sparse(1, [], []))] * 100,
        ["label", "weight", "features"])


def test_basic_estimator(spark_session):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression(regParam=0.0, solver="normal", weightCol="weight")
    training_dataset = get_training_dataset(spark_session)
    with mlflow.start_run() as run:
        lr_model = lr.fit(training_dataset)
    run_id = run.info.run_id
    params, metrics, tags, artifacts = get_run_data(run_id)
    assert params == truncate_dict(_get_estimator_param_map(lr))
    assert tags == get_expected_class_tags(lr_model)
    assert MODEL_DIR in artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert loaded_model.uid == lr_model.uid

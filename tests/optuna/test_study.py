import logging
import os
import random
import tempfile
import time
import pyspark
from datetime import datetime
from packaging.version import Version
from time import sleep
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import optuna
import pytest


import mlflow
from mlflow.entities import Metric, Param, RunTag

from mlflow.pyspark.optuna.storage import MlflowStorage
from mlflow.pyspark.optuna.study import MLFlowSparkStudy

from tests.pyfunc.test_spark import get_spark_session
from tests.pyspark.optuna.test_storage import setup_storage

_logger = logging.getLogger(__name__)


def _get_spark_session_with_retry(max_tries=3):
    conf = pyspark.SparkConf()
    for attempt in range(max_tries):
        try:
            return get_spark_session(conf)
        except Exception as e:
            if attempt >= max_tries - 1:
                raise
            _logger.exception(
                f"Attempt {attempt} to create a SparkSession failed ({e!r}), retrying..."
            )


# Specify `autouse=True` to ensure that a context is created
# before any tests are executed. This ensures that the Hadoop filesystem
# does not create its own SparkContext without the MLeap libraries required by
# other tests.
@pytest.fixture(scope="module")
def spark():
    if Version(pyspark.__version__) < Version("3.1"):
        spark_home = (
            os.environ.get("SPARK_HOME")
            if "SPARK_HOME" in os.environ
            else os.path.dirname(pyspark.__file__)
        )
        conf_dir = os.path.join(spark_home, "conf")
        os.makedirs(conf_dir, exist_ok=True)
        with open(os.path.join(conf_dir, "spark-defaults.conf"), "w") as f:
            conf = """
spark.driver.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
spark.executor.extraJavaOptions="-Dio.netty.tryReflectionSetAccessible=true"
"""
            f.write(conf)

    with _get_spark_session_with_retry() as spark:
        yield spark


@pytest.mark.usefixtures("spark")
def test_study_optimize_run(setup_storage):
    storage = setup_storage
    study_name = "test-study"
    mlflow_study = MLFlowSparkStudy(study_name, storage)

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2

    mlflow_study.optimize(objective, n_trials=1)

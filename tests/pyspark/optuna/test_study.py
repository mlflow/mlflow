import logging
import os

import numpy as np
import pyspark
import pytest
from optuna.samplers import TPESampler
from packaging.version import Version

import mlflow
from mlflow.exceptions import ExecutionException
from mlflow.pyspark.optuna.study import MlflowSparkStudy

from tests.optuna.test_storage import setup_storage  # noqa: F401
from tests.pyfunc.test_spark import get_spark_session

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
@pytest.fixture(scope="module", autouse=True)
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


@pytest.mark.skip(reason="TODO: Deflake this test")
def test_study_optimize_run(setup_storage):
    storage = setup_storage
    study_name = "test-study"
    sampler = TPESampler(seed=10)
    mlflow_study = MlflowSparkStudy(
        study_name, storage, sampler=sampler, mlflow_tracking_uri=mlflow.get_tracking_uri()
    )

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    mlflow_study.optimize(objective, n_trials=8, n_jobs=4)
    assert sorted(mlflow_study.best_params.keys()) == ["x"]
    assert len(mlflow_study.trials) == 8
    np.testing.assert_allclose(mlflow_study.best_params["x"], 5.426412865334919, rtol=1e-6)


def test_study_with_failed_objective(setup_storage):
    storage = setup_storage
    study_name = "test-study"
    sampler = TPESampler(seed=10)
    mlflow_study = MlflowSparkStudy(
        study_name, storage, sampler=sampler, mlflow_tracking_uri=mlflow.get_tracking_uri()
    )

    def fail_objective(_):
        raise ValueError()

    with pytest.raises(
        ExecutionException,
        match="Optimization run for Optuna MlflowSparkStudy failed",
    ):
        mlflow_study.optimize(fail_objective, n_trials=4)

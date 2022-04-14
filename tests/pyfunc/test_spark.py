import os
import sys
import time
from typing import Iterator
import threading
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import pyspark
from pyspark.sql.types import ArrayType, DoubleType, LongType, StringType, FloatType, IntegerType
from pyspark.sql.utils import AnalysisException

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.pyfunc import spark_udf, PythonModel, PyFuncModel
from mlflow.pyfunc.spark_model_cache import SparkModelCache
from mlflow.utils.environment import _EnvManager

import tests
from mlflow.types import Schema, ColSpec

import sklearn.datasets as datasets
from collections import namedtuple
from sklearn.neighbors import KNeighborsClassifier

from pyspark.sql.functions import pandas_udf


prediction = [int(1), int(2), "class1", float(0.1), 0.2]
types = [np.int32, int, str, np.float32, np.double]


def score_model_as_udf(model_uri, pandas_df, result_type="double"):
    spark = get_spark_session(pyspark.SparkConf())
    spark_df = spark.createDataFrame(pandas_df)
    pyfunc_udf = spark_udf(spark=spark, model_uri=model_uri, result_type=result_type)
    new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
    return [x["prediction"] for x in new_df.collect()]


class ConstantPyfuncWrapper:
    @staticmethod
    def predict(model_input):
        m, _ = model_input.shape
        prediction_df = pd.DataFrame(
            data={
                str(i): np.array([prediction[i] for j in range(m)], dtype=types[i])
                for i in range(len(prediction))
            },
            columns=[str(i) for i in range(len(prediction))],
        )
        return prediction_df


def _load_pyfunc(_):
    return ConstantPyfuncWrapper()


@pytest.fixture(autouse=True)
def configure_environment():
    os.environ["PYSPARK_PYTHON"] = sys.executable


def get_spark_session(conf):
    conf.set(key="spark_session.python.worker.reuse", value=True)
    # when local run test_spark.py
    # you can set SPARK_MASTER=local[1]
    # so that executor log will be printed as test process output
    # which make debug easiser.
    spark_master = os.environ.get("SPARK_MASTER", "local-cluster[2, 1, 1024]")
    return (
        pyspark.sql.SparkSession.builder.config(conf=conf)
        .master(spark_master)
        .config("spark.task.maxFailures", "1")  # avoid retry failed spark tasks
        .getOrCreate()
    )


@pytest.fixture(scope="module")
def spark():
    conf = pyspark.SparkConf()
    session = get_spark_session(conf)
    yield session
    session.stop()


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def sklearn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.mark.large
def test_spark_udf(spark, model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        loader_module=__name__,
        code_path=[os.path.dirname(tests.__file__)],
    )

    with mock.patch("mlflow.pyfunc._warn_dependency_requirement_mismatches") as mock_check_fn:
        reloaded_pyfunc_model = mlflow.pyfunc.load_model(model_path)
        mock_check_fn.assert_called_once()

    pandas_df = pd.DataFrame(data=np.ones((10, 10)), columns=[str(i) for i in range(10)])
    spark_df = spark.createDataFrame(pandas_df)

    # Test all supported return types
    type_map = {
        "float": (FloatType(), np.number),
        "int": (IntegerType(), np.int32),
        "double": (DoubleType(), np.number),
        "long": (LongType(), int),
        "string": (StringType(), None),
    }

    for tname, tdef in type_map.items():
        spark_type, np_type = tdef
        prediction_df = reloaded_pyfunc_model.predict(pandas_df)
        for is_array in [True, False]:
            t = ArrayType(spark_type) if is_array else spark_type
            if tname == "string":
                expected = prediction_df.applymap(str)
            else:
                expected = prediction_df.select_dtypes(np_type)
                if tname == "float":
                    expected = expected.astype(np.float32)

            expected = [list(row[1]) if is_array else row[1][0] for row in expected.iterrows()]
            pyfunc_udf = spark_udf(spark, model_path, result_type=t)
            new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
            actual = list(new_df.select("prediction").toPandas()["prediction"])
            assert expected == actual
            if not is_array:
                pyfunc_udf = spark_udf(spark, model_path, result_type=tname)
                new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
                actual = list(new_df.select("prediction").toPandas()["prediction"])
                assert expected == actual


@pytest.mark.parametrize("sklearn_version", ["0.22.1", "0.24.0"])
@pytest.mark.parametrize("env_manager", ["virtualenv", "conda"])
def test_spark_udf_env_manager_can_restore_env(spark, model_path, sklearn_version, env_manager):
    class EnvRestoringTestModel(mlflow.pyfunc.PythonModel):
        def __init__(self):
            pass

        def predict(self, context, model_input):
            import sklearn

            if sklearn.__version__ == sklearn_version:
                pred_value = 1
            else:
                pred_value = 0

            return model_input.apply(lambda row: pred_value, axis=1)

    infer_spark_df = spark.createDataFrame(pd.DataFrame(data=[[1, 2]], columns=["a", "b"]))

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=EnvRestoringTestModel(),
        pip_requirements=[
            "pyspark==3.2.0",
            "pandas==1.3.0",
            f"scikit-learn=={sklearn_version}",
            "pytest==6.2.5",
        ],
    )

    python_udf = mlflow.pyfunc.spark_udf(spark, model_path, env_manager=env_manager)
    result = infer_spark_df.select(python_udf("a", "b").alias("result")).toPandas().result[0]

    assert result == 1


@pytest.mark.parametrize("env_manager", ["virtualenv", "conda"])
def test_spark_udf_env_manager_predict_sklearn_model(spark, sklearn_model, model_path, env_manager):
    model, inference_data = sklearn_model

    mlflow.sklearn.save_model(model, model_path)
    expected_pred_result = model.predict(inference_data)

    infer_data = pd.DataFrame(inference_data, columns=["a", "b"])
    infer_spark_df = spark.createDataFrame(infer_data)

    pyfunc_udf = spark_udf(spark, model_path, env_manager=env_manager)
    result = (
        infer_spark_df.select(pyfunc_udf("a", "b").alias("predictions"))
        .toPandas()
        .predictions.to_numpy()
    )

    np.testing.assert_allclose(result, expected_pred_result, rtol=1e-5)


def test_spark_udf_with_single_arg(spark):
    from pyspark.sql.functions import struct

    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [",".join(model_input.columns.tolist())] * len(model_input)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=StringType()
        )

        data1 = spark.createDataFrame(pd.DataFrame({"a": [1], "b": [4]})).repartition(1)

        result = data1.withColumn("res", udf("a")).select("res").toPandas()
        assert result.res[0] == "0"

        data2 = data1.select(struct("a", "b").alias("ab"))
        result = data2.withColumn("res", udf("ab")).select("res").toPandas()
        assert result.res[0] == "a,b"


def test_spark_udf_autofills_no_arguments(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "a"), ColSpec("long", "b"), ColSpec("long", "c")]),
        outputs=Schema([ColSpec("integer")]),
    )

    good_data = spark.createDataFrame(
        pd.DataFrame(columns=["a", "b", "c", "d"], data={"a": [1], "b": [2], "c": [3], "d": [4]})
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        res = good_data.withColumn("res", udf()).select("res").toPandas()
        assert res["res"][0] == ["a", "b", "c"]

        with pytest.raises(
            pyspark.sql.utils.PythonException,
            match=r"Model input is missing columns. Expected 3 input columns",
        ):
            res = good_data.withColumn("res", udf("b", "c")).select("res").toPandas()

        # this dataframe won't work because it's missing column a
        bad_data = spark.createDataFrame(
            pd.DataFrame(
                columns=["x", "b", "c", "d"], data={"x": [1], "b": [2], "c": [3], "d": [4]}
            )
        )
        with pytest.raises(AnalysisException, match=r"cannot resolve 'a' given input columns"):
            bad_data.withColumn("res", udf())

    nameless_signature = ModelSignature(
        inputs=Schema([ColSpec("long"), ColSpec("long"), ColSpec("long")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=nameless_signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        with pytest.raises(
            MlflowException,
            match=r"Cannot apply udf because no column names specified",
        ):
            good_data.withColumn("res", udf())

    with mlflow.start_run() as run:
        # model without signature
        mlflow.pyfunc.log_model("model", python_model=TestModel())
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        with pytest.raises(MlflowException, match="Attempting to apply udf on zero columns"):
            res = good_data.withColumn("res", udf()).select("res").toPandas()


def test_spark_udf_autofills_column_names_with_schema(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "a"), ColSpec("long", "b"), ColSpec("long", "c")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        data = spark.createDataFrame(
            pd.DataFrame(
                columns=["a", "b", "c", "d"], data={"a": [1], "b": [2], "c": [3], "d": [4]}
            )
        )
        with pytest.raises(pyspark.sql.utils.PythonException, match=r".+"):
            res = data.withColumn("res1", udf("a", "b")).select("res1").toPandas()

        res = data.withColumn("res2", udf("a", "b", "c")).select("res2").toPandas()
        assert res["res2"][0] == ["a", "b", "c"]
        res = data.withColumn("res4", udf("a", "b", "c", "d")).select("res4").toPandas()
        assert res["res4"][0] == ["a", "b", "c"]


def test_spark_udf_with_datetime_columns(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [model_input.columns] * len(model_input)

    signature = ModelSignature(
        inputs=Schema([ColSpec("datetime", "timestamp"), ColSpec("datetime", "date")]),
        outputs=Schema([ColSpec("integer")]),
    )
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=ArrayType(StringType())
        )
        data = spark.range(10).selectExpr(
            "current_timestamp() as timestamp", "current_date() as date"
        )

        res = data.withColumn("res", udf("timestamp", "date")).select("res")
        res = res.toPandas()
        assert res["res"][0] == ["timestamp", "date"]


@pytest.mark.large
def test_model_cache(spark, model_path):
    mlflow.pyfunc.save_model(
        path=model_path,
        loader_module=__name__,
        code_path=[os.path.dirname(tests.__file__)],
    )

    archive_path = SparkModelCache.add_local_model(spark, model_path)
    assert archive_path != model_path

    # Define the model class name as a string so that each Spark executor can reference it
    # without attempting to resolve ConstantPyfuncWrapper, which is only available on the driver.
    constant_model_name = ConstantPyfuncWrapper.__name__

    def check_get_or_load_return_value(model_from_cache, model_path_from_cache):
        assert model_path_from_cache != model_path
        assert os.path.isdir(model_path_from_cache)
        model2 = mlflow.pyfunc.load_model(model_path_from_cache)
        for model in [model_from_cache, model2]:
            assert isinstance(model, PyFuncModel)
            # NB: Can not use instanceof test as remote does not know about ConstantPyfuncWrapper
            # class.
            assert type(model._model_impl).__name__ == constant_model_name

    # Ensure we can use the model locally.
    local_model, local_model_path = SparkModelCache.get_or_load(archive_path)

    check_get_or_load_return_value(local_model, local_model_path)

    # Request the model on all executors, and see how many times we got cache hits.
    def get_model(_):
        executor_model, executor_model_path = SparkModelCache.get_or_load(archive_path)
        check_get_or_load_return_value(executor_model, executor_model_path)
        return SparkModelCache._cache_hits

    # This will run 30 distinct tasks, and we expect most to reuse an already-loaded model.
    # Note that we can't necessarily expect an even split, or even that there were only
    # exactly 2 python processes launched, due to Spark and its mysterious ways, but we do
    # expect significant reuse.
    results = spark.sparkContext.parallelize(range(100), 30).map(get_model).collect()
    assert max(results) > 10
    # Running again should see no newly-loaded models.
    results2 = spark.sparkContext.parallelize(range(100), 30).map(get_model).collect()
    assert min(results2) > 0


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Only Linux system support setting  parent process death signal via prctl lib.",
)
@pytest.mark.large
def test_spark_udf_embedded_model_server_killed_when_job_canceled(spark, sklearn_model, model_path):
    from mlflow.pyfunc.scoring_server.client import ScoringServerClient
    from mlflow.models.cli import _get_flavor_backend

    mlflow.sklearn.save_model(sklearn_model.model, model_path)

    server_port = 51234

    @pandas_udf("int")
    def udf_with_model_server(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        from mlflow.models.cli import _get_flavor_backend

        _get_flavor_backend(
            model_path, env_manager=_EnvManager.CONDA, workers=1, install_mlflow=False
        ).serve(
            model_uri=model_path,
            port=server_port,
            host="127.0.0.1",
            enable_mlserver=False,
            synchronous=False,
        )

        time.sleep(120)
        for x in it:
            yield x

    def run_job():
        # Start a spark job with only one UDF task,
        # and the udf task starts a mlflow model server process.
        spark.range(1).repartition(1).select(udf_with_model_server("id")).collect()

    _get_flavor_backend(
        model_path, env_manager=_EnvManager.CONDA, install_mlflow=False
    ).prepare_env(model_uri=model_path)

    job_thread = threading.Thread(target=run_job)
    job_thread.start()

    client = ScoringServerClient("127.0.0.1", server_port)
    client.wait_server_ready(timeout=20)
    spark.sparkContext.cancelAllJobs()
    job_thread.join()

    time.sleep(10)  # waiting server to exit and release the port.

    # assert ping failed, i.e. the server process is killed successfully.
    with pytest.raises(Exception):  # pylint: disable=pytest-raises-without-match
        client.ping()

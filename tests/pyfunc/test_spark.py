import datetime
import os
import random
import sys
import time
from typing import Iterator
import threading
from collections import namedtuple
from unittest import mock
import pytest
from packaging.version import Version

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import pandas_udf, col, struct
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    FloatType,
    IntegerType,
    BooleanType,
    StructType,
    StructField,
)
from pyspark.sql.utils import AnalysisException
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import tests

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.pyfunc import spark_udf, PythonModel, PyFuncModel
from mlflow.pyfunc.spark_model_cache import SparkModelCache
from mlflow.types import Schema, ColSpec


prediction = [int(1), int(2), "class1", float(0.1), 0.2, True]
types = [np.int32, int, str, np.float32, np.double, bool]


def score_model_as_udf(model_uri, pandas_df, result_type="double"):
    spark = get_spark_session(pyspark.SparkConf())
    spark_df = spark.createDataFrame(pandas_df).coalesce(1)
    pyfunc_udf = spark_udf(
        spark=spark, model_uri=model_uri, result_type=result_type, env_manager="local"
    )
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
    conf.set(key="spark.python.worker.reuse", value="true")
    # disable task retry (i.e. make it fast fail)
    conf.set(key="spark.task.maxFailures", value="1")
    # Disable simplifying traceback from Python UDFs
    conf.set(key="spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled", value="false")
    # Show jvm side stack trace.
    conf.set(key="spark.sql.pyspark.jvmStacktrace.enabled", value="true")
    # when local run test_spark.py
    # you can set SPARK_MASTER=local[1]
    # so that executor log will be printed as test process output
    # which make debug easier.
    # If running in local mode on certain OS configurations (M1 Mac ARM CPUs)
    # adding `.config("spark.driver.bindAddress", "127.0.0.1")` to the SparkSession
    # builder configuration will enable a SparkSession to start.
    # For local testing, uncomment the following line:
    # spark_master = os.environ.get("SPARK_MASTER", "local[1]")
    # If doing local testing, comment-out the following line.
    spark_master = os.environ.get("SPARK_MASTER", "local-cluster[2, 1, 1024]")
    # Don't forget to revert these changes prior to pushing a branch!
    return (
        pyspark.sql.SparkSession.builder.config(conf=conf)
        .master(spark_master)
        # Uncomment the following line for testing on Apple silicon locally
        # .config("spark.driver.bindAddress", "127.0.0.1")
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
        "bool": (BooleanType(), bool),
        "boolean": (BooleanType(), bool),
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
                if tname == "bool" or tname == "boolean":
                    expected = expected.astype(bool)

            expected = [list(row[1]) if is_array else row[1][0] for row in expected.iterrows()]
            pyfunc_udf = spark_udf(spark, model_path, result_type=t, env_manager="local")
            new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))
            actual = list(new_df.select("prediction").toPandas()["prediction"])
            assert expected == actual
            if not is_array:
                pyfunc_udf = spark_udf(spark, model_path, result_type=tname, env_manager="local")
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

            return model_input.apply(lambda row: sklearn.__version__, axis=1)

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
    # tests/helper_functions.py
    from tests.helper_functions import _get_mlflow_home

    os.environ["MLFLOW_HOME"] = _get_mlflow_home()
    python_udf = mlflow.pyfunc.spark_udf(
        spark, model_path, env_manager=env_manager, result_type="string"
    )
    result = infer_spark_df.select(python_udf("a", "b").alias("result")).toPandas().result[0]
    assert result == sklearn_version


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
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [",".join(map(str, model_input.columns.tolist()))] * len(model_input)

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


def test_spark_udf_with_struct_return_type(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            input_len = len(model_input)
            return {
                "r1": [1] * input_len,
                "r2": [1.5] * input_len,
                "r3": [[1, 2]] * input_len,
                "r4": [np.array([1.5, 2.5])] * input_len,
                "r5": np.vstack([np.array([1.5, 2.5])] * input_len),
                "r6": [True] * input_len,
                "r7": ["abc"] * input_len,
            }

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

        udf = mlflow.pyfunc.spark_udf(
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type=(
                "r1 int, r2 float, r3 array<long>, r4 array<double>, "
                "r5 array<double>, r6 boolean, r7 string"
            ),
        )

        data1 = spark.range(2).repartition(1)
        result = (
            data1.withColumn("res", udf("id"))
            .select("res.r1", "res.r2", "res.r3", "res.r4", "res.r5", "res.r6", "res.r7")
            .toPandas()
        )
        assert result.r1.tolist() == [1] * 2
        np.testing.assert_almost_equal(result.r2.tolist(), [1.5] * 2)
        assert result.r3.tolist() == [[1, 2]] * 2
        np.testing.assert_almost_equal(
            np.vstack(result.r4.tolist()), np.array([[1.5, 2.5], [1.5, 2.5]])
        )
        np.testing.assert_almost_equal(
            np.vstack(result.r5.tolist()), np.array([[1.5, 2.5], [1.5, 2.5]])
        )
        assert result.r6.tolist() == [True] * 2
        assert result.r7.tolist() == ["abc"] * 2


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
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type=ArrayType(StringType()),
            env_manager="local",
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
        with pytest.raises(
            AnalysisException,
            match=(
                # PySpark < 3.3
                r"cannot resolve 'a' given input columns|"
                # PySpark 3.3
                r"Column 'a' does not exist|"
                # PySpark 3.4
                r"A column or function parameter with name `a` cannot be resolved"
            ),
        ):
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
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type=ArrayType(StringType()),
            env_manager="local",
        )
        data = spark.createDataFrame(
            pd.DataFrame(
                columns=["a", "b", "c", "d"], data={"a": [1], "b": [2], "c": [3], "d": [4]}
            )
        )

        res = data.withColumn("res2", udf("a", "b", "c")).select("res2").toPandas()
        assert res["res2"][0] == ["a", "b", "c"]
        res = data.withColumn("res4", udf("a", "b", "c", "d")).select("res4").toPandas()
        assert res["res4"][0] == ["a", "b", "c"]

        # Exception being thrown in udf process intermittently causes the SparkSession to crash
        # which results in a `java.net.SocketException: Socket is closed` failure in subsequent
        # tests if tests are conducted after this exception capture validation.
        # Keep this at the end of this suite so that executor sockets don't get closed while
        # processing is still being conducted.
        with pytest.raises(pyspark.sql.utils.PythonException, match=r".+"):
            data.withColumn("res1", udf("a", "b")).select("res1").toPandas()


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
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type=ArrayType(StringType()),
            env_manager="local",
        )
        data = spark.range(10).selectExpr(
            "current_timestamp() as timestamp", "current_date() as date"
        )

        res = data.withColumn("res", udf("timestamp", "date")).select("res")
        res = res.toPandas()
        assert res["res"][0] == ["timestamp", "date"]


def test_spark_udf_over_empty_partition(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            if len(model_input) == 0:
                raise ValueError("Empty input is not allowed.")
            else:
                return model_input.a + model_input.b

    signature = ModelSignature(
        inputs=Schema([ColSpec("long", "a"), ColSpec("long", "b")]),
        outputs=Schema([ColSpec("long")]),
    )

    # Create a spark dataframe with 2 partitions, one partition has one record and
    # the other partition is empty.
    spark_df = spark.createDataFrame(
        pd.DataFrame(columns=["x", "y"], data={"x": [11], "y": [21]})
    ).repartition(2)
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        python_udf = mlflow.pyfunc.spark_udf(
            spark, "runs:/{}/model".format(run.info.run_id), result_type=LongType()
        )
        res_df = spark_df.withColumn("res", python_udf("x", "y")).select("res").toPandas()
        assert res_df.res[0] == 32

        res_df2 = (
            spark_df.withColumn("res", python_udf(struct(col("x").alias("a"), col("y").alias("b"))))
            .select("res")
            .toPandas()
        )
        assert res_df2.res[0] == 32


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
@pytest.mark.parametrize("env_manager", ["virtualenv", "conda"])
def test_spark_udf_embedded_model_server_killed_when_job_canceled(
    spark, sklearn_model, model_path, env_manager
):
    from mlflow.pyfunc.scoring_server.client import ScoringServerClient
    from mlflow.models.flavor_backend_registry import get_flavor_backend

    mlflow.sklearn.save_model(sklearn_model.model, model_path)

    server_port = 51234
    timeout = 60

    @pandas_udf("int")
    def udf_with_model_server(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        from mlflow.models.flavor_backend_registry import get_flavor_backend

        get_flavor_backend(
            model_path, env_manager=env_manager, workers=1, install_mlflow=False
        ).serve(
            model_uri=model_path,
            port=server_port,
            host="127.0.0.1",
            timeout=timeout,
            enable_mlserver=False,
            synchronous=False,
        )

        time.sleep(120)
        yield from it

    def run_job():
        # Start a spark job with only one UDF task,
        # and the udf task starts a mlflow model server process.
        spark.range(1).repartition(1).select(udf_with_model_server("id")).collect()

    get_flavor_backend(model_path, env_manager=env_manager, install_mlflow=False).prepare_env(
        model_uri=model_path
    )

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


def test_spark_udf_datetime_with_model_schema(spark):
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    X = X.assign(
        timestamp=[datetime.datetime(2022, random.randint(1, 12), 1) for _ in range(len(X))]
    )

    month_extractor = FunctionTransformer(
        lambda df: df.assign(month=df["timestamp"].map(lambda d: d.month)), validate=False
    )
    timestamp_remover = ColumnTransformer(
        [("selector", "passthrough", X.columns.drop("timestamp"))], remainder="drop"
    )
    model = Pipeline(
        [
            ("month_extractor", month_extractor),
            ("timestamp_remover", timestamp_remover),
            ("knn", KNeighborsClassifier()),
        ]
    )
    model.fit(X, y)

    timestamp_dtype = {"timestamp": "datetime64[ns]"}
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(X.astype(timestamp_dtype), y)
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    inference_sample = X.sample(n=10, random_state=42)
    infer_spark_df = spark.createDataFrame(inference_sample.astype(timestamp_dtype))
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="conda")
    result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()
    np.testing.assert_almost_equal(result.to_numpy().squeeze(), model.predict(inference_sample))


def test_spark_udf_string_datetime_with_model_schema(spark):
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    X = X.assign(timestamp=["2022-{:02d}-01".format(random.randint(1, 12)) for _ in range(len(X))])

    month_extractor = FunctionTransformer(
        lambda df: df.assign(month=df["timestamp"].str.extract(r"^2022-0?(\d{1,2})-").astype(int)),
        validate=False,
    )
    timestamp_remover = ColumnTransformer(
        [("selector", "passthrough", X.columns.drop("timestamp"))], remainder="drop"
    )
    model = Pipeline(
        [
            ("month_extractor", month_extractor),
            ("timestamp_remover", timestamp_remover),
            ("knn", KNeighborsClassifier()),
        ]
    )
    model.fit(X, y)

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(X, y)
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    inference_sample = X.sample(n=10, random_state=42)
    infer_spark_df = spark.createDataFrame(inference_sample)
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="conda")
    result = infer_spark_df.select(pyfunc_udf(*X.columns).alias("predictions")).toPandas()
    np.testing.assert_almost_equal(result.to_numpy().squeeze(), model.predict(inference_sample))


def test_spark_udf_with_col_spec_type_input(spark):
    input_pdf = pd.DataFrame(
        {
            "c_bool": [True],
            "c_int": [10],
            "c_long": [20],
            "c_float": [1.5],
            "c_double": [2.5],
            "c_str": ["abc"],
            "c_binary": [b"xyz"],
            "c_datetime": [pd.to_datetime("2018-01-01")],
        }
    )

    class TestModel(PythonModel):
        def predict(self, context, model_input):
            assert model_input.to_dict() == input_pdf.to_dict()
            return model_input[["c_int", "c_float"]]

    signature = ModelSignature(
        inputs=Schema(
            [
                ColSpec("boolean", "c_bool"),
                ColSpec("integer", "c_int"),
                ColSpec("long", "c_long"),
                ColSpec("float", "c_float"),
                ColSpec("double", "c_double"),
                ColSpec("string", "c_str"),
                ColSpec("binary", "c_binary"),
                ColSpec("datetime", "c_datetime"),
            ]
        ),
    )

    spark_schema = (
        "c_bool boolean, c_int int, c_long long, c_float float, c_double double, "
        "c_str string, c_binary binary, c_datetime timestamp"
    )
    data = spark.createDataFrame(
        data=input_pdf,
        schema=spark_schema,
    ).repartition(1)

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel(), signature=signature)
        udf = mlflow.pyfunc.spark_udf(
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type="c_int int, c_float float",
            env_manager="local",
        )
        res = data.withColumn("res", udf()).select("res.c_int", "res.c_float").toPandas()
        assert res.c_int.tolist() == [10]
        np.testing.assert_almost_equal(res.c_float.tolist(), [1.5])


def test_spark_udf_stdin_scoring_server(spark, monkeypatch):
    X, y = datasets.load_iris(return_X_y=True, as_frame=True)
    X = X[::5]
    y = y[::5]
    model = LogisticRegression().fit(X, y)
    model.fit(X, y)

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(X, y)
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)

    with mock.patch("mlflow.pyfunc.check_port_connectivity", return_value=False):
        udf = mlflow.pyfunc.spark_udf(
            spark,
            model_info.model_uri,
            env_manager="virtualenv",
        )
        df = spark.createDataFrame(X)
        result = df.select(udf(*X.columns)).toPandas()
        np.testing.assert_almost_equal(result.to_numpy().squeeze(), model.predict(X))


# TODO: Remove `skipif` once pyspark 3.4 is released
@pytest.mark.skipif(
    Version(pyspark.__version__) < Version("3.4.0"), reason="requires spark >= 3.4.0"
)
def test_spark_udf_array_of_structs(spark):
    class TestModel(PythonModel):
        def predict(self, context, model_input):
            return [[("str", 0, 1, 0.0, 0.1, True)]] * len(model_input)

    signature = ModelSignature(inputs=Schema([ColSpec("long", "a")]))
    good_data = spark.createDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            "model",
            python_model=TestModel(),
            signature=signature,
        )
        udf = mlflow.pyfunc.spark_udf(
            spark,
            "runs:/{}/model".format(run.info.run_id),
            result_type=ArrayType(
                StructType(
                    [
                        StructField("str", StringType()),
                        StructField("int", IntegerType()),
                        StructField("long", LongType()),
                        StructField("float", FloatType()),
                        StructField("double", DoubleType()),
                        StructField("bool", BooleanType()),
                    ]
                )
            ),
        )
        res = good_data.withColumn("res", udf("a")).select("res").toPandas()
        assert res["res"][0] == [("str", 0, 1, 0.0, 0.1, True)]

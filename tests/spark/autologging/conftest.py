import os
import tempfile

import pytest
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from mlflow.spark.autologging import clear_table_infos

from tests.spark.autologging.utils import _get_or_create_spark_session


# Session-scoped version of pytest monkeypatch fixture. Original monkeypatch in pytest
# is function-scoped, thus we need a larger scoped one to use that in module/session
# scoped fixtures.
@pytest.fixture(scope="session")
def monkeypatch_session(request):
    mpatch = pytest.MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(autouse=True, scope="session")
def disable_pyspark_pin_thread(monkeypatch_session):
    # PYSPARK_PIN_THREAD is set to true by default since Pyspark 3.2.0, which causes
    # issues with Py4J callbacks, so we ask users to set it to false.
    # We have to set this before creating the SparkSession, hence setting it session
    # -scoped, which is applied before module-scoped spark_session fixture
    monkeypatch_session.setenv("PYSPARK_PIN_THREAD", "false")


@pytest.fixture(scope="module")
def spark_session():
    with _get_or_create_spark_session() as session:
        yield session


@pytest.fixture
def data_format(format_to_file_path):
    res, _ = sorted(format_to_file_path.items())[0]
    return res


@pytest.fixture
def file_path(format_to_file_path):
    _, file_path = sorted(format_to_file_path.items())[0]
    return file_path


@pytest.fixture
def format_to_file_path(spark_session):
    rows = [Row(8, 32, "bat"), Row(64, 40, "mouse"), Row(-27, 55, "horse")]
    schema = StructType(
        [
            StructField("number2", IntegerType()),
            StructField("number1", IntegerType()),
            StructField("word", StringType()),
        ]
    )
    rdd = spark_session.sparkContext.parallelize(rows)
    df = spark_session.createDataFrame(rdd, schema)
    res = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for data_format in ["csv", "parquet", "json"]:
            res[data_format] = os.path.join(tempdir, f"test-data-{data_format}")

        for data_format, file_path in res.items():
            df.write.option("header", "true").format(data_format).save(file_path)
        yield res


@pytest.fixture(autouse=True)
def tear_down():
    yield

    # Clear cached table infos. When the datasource event from Spark arrives but there is no
    # active run (e.g. the even comes with some delay), MLflow keep them in memory and logs them to
    # the next **and any successive active run** (ref: PR #4086).
    # However, this behavior is not desirable during tests, as we don't want any tests to be
    # affected by the previous test. Hence, this fixture is executed on every test function
    # to clear the accumulated table infos stored in the global context.
    clear_table_infos()

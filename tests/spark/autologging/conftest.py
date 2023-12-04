import os
import tempfile

import pytest
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from tests.spark.autologging.utils import _get_or_create_spark_session



@pytest.fixture(scope="module")
def spark_session():
    with _get_or_create_spark_session() as session:
        yield session


@pytest.fixture(scope="module")
def data_format(format_to_file_path):
    res, _ = sorted(format_to_file_path.items())[0]
    return res


@pytest.fixture(scope="module")
def file_path(format_to_file_path):
    _, file_path = sorted(format_to_file_path.items())[0]
    return file_path


@pytest.fixture(scope="module")
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

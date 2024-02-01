from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from datetime import datetime
import pytest

from unittest import mock
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _enforce_schema
from mlflow.types import ColSpec, DataType, Schema


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.getOrCreate() as spark:
        yield spark


def test_enforce_schema_spark_dataframe(spark):
    spark_df_schema = StructType(
        [
            StructField("smallint", ShortType(), True),
            StructField("int", IntegerType(), True),
            StructField("bigint", LongType(), True),
            StructField("float", FloatType(), True),
            StructField("double", DoubleType(), True),
            StructField("boolean", BooleanType(), True),
            StructField("date", DateType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("string", StringType(), True),
            StructField("binary", BinaryType(), True),
        ]
    )

    data = [
        (
            1,  # smallint
            2,  # int
            1234567890123456789,  # bigint
            1.23,  # float
            3.456789,  # double
            True,  # boolean
            datetime(2020, 1, 1),  # date
            datetime.now(),  # timestamp
            "example string",  # string
            bytearray("example binary", "utf-8"),  # binary
        )
    ]

    df = spark.createDataFrame(data, spark_df_schema)
    input_schema = Schema(
        [
            ColSpec(DataType.integer, "smallint"),
            ColSpec(DataType.integer, "int"),
            ColSpec(DataType.long, "bigint"),
            ColSpec(DataType.float, "float"),
            ColSpec(DataType.double, "double"),
            ColSpec(DataType.boolean, "boolean"),
            ColSpec(DataType.datetime, "date"),
            ColSpec(DataType.datetime, "timestamp"),
            ColSpec(DataType.string, "string"),
            ColSpec(DataType.binary, "binary"),
        ]
    )
    _enforce_schema(df, input_schema)


def test_enforce_schema_spark_dataframe_missing_col(spark):
    spark_df_schema = StructType(
        [StructField("smallint", ShortType(), True), StructField("int", IntegerType(), True)]
    )

    data = [
        (
            1,  # smallint
            2,  # int
        )
    ]

    df = spark.createDataFrame(data, spark_df_schema)
    input_schema = Schema(
        [
            ColSpec(DataType.integer, "smallint"),
            ColSpec(DataType.integer, "int"),
            ColSpec(DataType.long, "bigint"),
        ]
    )
    with pytest.raises(MlflowException, match="Model is missing inputs"):
        _enforce_schema(df, input_schema)


def test_enforce_schema_spark_dataframe_incompatible_type(spark):
    spark_df_schema = StructType(
        [StructField("a", ShortType(), True), StructField("b", DoubleType(), True)]
    )

    data = [
        (
            1,  # a
            2.3,  # b
        )
    ]

    df = spark.createDataFrame(data, spark_df_schema)
    input_schema = Schema(
        [
            ColSpec(DataType.integer, "a"),
            ColSpec(DataType.integer, "b"),
        ]
    )
    with pytest.raises(MlflowException, match="Incompatible input types"):
        _enforce_schema(df, input_schema)


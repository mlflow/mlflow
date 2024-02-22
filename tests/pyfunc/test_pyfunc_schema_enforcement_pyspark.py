from datetime import datetime

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.testing import assertDataFrameEqual

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _enforce_schema
from mlflow.types import ColSpec, DataType, Schema
from mlflow.types.schema import Array, Object, Property


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

    input_df = spark.createDataFrame(data, spark_df_schema)
    result = _enforce_schema(input_df, input_schema)
    assertDataFrameEqual(input_df, result)


@pytest.mark.parametrize(
    ("spark_df_schema", "data", "input_schema"),
    [
        (
            StructType([StructField("query", ArrayType(StringType()), True)]),
            [(["sentence_1", "sentence_2"],)],
            Schema([ColSpec(Array(DataType.string), name="query")]),
        ),
        (
            StructType(
                [
                    StructField(
                        "teststruct",
                        StructType(
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
                        ),
                        True,
                    )
                ]
            ),
            [
                Row(
                    teststruct=Row(
                        smallint=100,
                        int=1000,
                        bigint=10000000000,
                        float=10.5,
                        double=20.5,
                        boolean=True,
                        date=datetime(2020, 1, 1),
                        timestamp=datetime.now(),
                        string="example",
                        binary=b"binary_data",
                    )
                ),
                Row(
                    teststruct=Row(
                        smallint=200,
                        int=2000,
                        bigint=20000000000,
                        float=20.5,
                        double=30.5,
                        boolean=False,
                        date=datetime(2020, 1, 1),
                        timestamp=datetime.now(),
                        string="sample",
                        binary=b"sample_data",
                    )
                ),
                Row(
                    teststruct=Row(
                        smallint=300,
                        int=3000,
                        bigint=30000000000,
                        float=30.5,
                        double=40.5,
                        boolean=True,
                        date=datetime(2020, 1, 1),
                        timestamp=datetime.now(),
                        string="data",
                        binary=b"data_binary",
                    )
                ),
            ],
            Schema(
                [
                    ColSpec(
                        Object(
                            [
                                Property("smallint", DataType.integer),
                                Property("int", DataType.integer),
                                Property("bigint", DataType.long),
                                Property("float", DataType.float),
                                Property("double", DataType.double),
                                Property("boolean", DataType.boolean),
                                Property("date", DataType.datetime),
                                Property("timestamp", DataType.datetime),
                                Property("string", DataType.string),
                                Property("binary", DataType.binary),
                            ]
                        ),
                        "teststruct",
                    )
                ]
            ),
        ),
        (
            StructType(
                [
                    StructField(
                        "array",
                        ArrayType(
                            StructType(
                                [
                                    StructField("name", StringType(), True),
                                    StructField("age", DoubleType(), True),
                                ]
                            )
                        ),
                        True,
                    )
                ]
            ),
            [
                (
                    [
                        Row(name="Alice", age=30.0),
                        Row(name="Bob", age=25.0),
                        Row(name="Catherine", age=35.0),
                    ],
                )
            ],
            Schema(
                [
                    ColSpec(
                        Array(
                            Object(
                                [
                                    Property("name", DataType.string),
                                    Property("age", DataType.double),
                                ]
                            )
                        ),
                        name="array",
                    ),
                ]
            ),
        ),
        (
            StructType([StructField("nested_list", ArrayType(ArrayType(IntegerType())), True)]),
            [
                ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],),
                ([[10, 11], [12, 13, 14]],),
            ],
            Schema([ColSpec(Array(Array(DataType.integer)), name="nested_list")]),
        ),
    ],
)
def test_enforce_schema_spark_dataframe_complex(spark_df_schema, data, input_schema, spark):
    input_df = spark.createDataFrame(data, spark_df_schema)
    result = _enforce_schema(input_df, input_schema)
    assertDataFrameEqual(input_df, result)


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

    input_schema = Schema(
        [
            ColSpec(DataType.integer, "smallint"),
            ColSpec(DataType.integer, "int"),
            ColSpec(DataType.long, "bigint"),
        ]
    )

    df = spark.createDataFrame(data, spark_df_schema)
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

    input_schema = Schema(
        [
            ColSpec(DataType.integer, "a"),
            ColSpec(DataType.integer, "b"),
        ]
    )

    df = spark.createDataFrame(data, spark_df_schema)
    with pytest.raises(MlflowException, match="Incompatible input types"):
        _enforce_schema(df, input_schema)


def test_enforce_schema_spark_dataframe_incompatible_type_complex(spark):
    spark_df_schema = StructType(
        [
            StructField(
                "teststruct",
                StructType(
                    [
                        StructField("int", IntegerType(), True),
                        StructField("double", DoubleType(), True),
                    ]
                ),
            )
        ]
    )

    data = [
        Row(
            teststruct=Row(
                int=1000,
                double=20.5,
            )
        )
    ]

    input_schema = Schema(
        [
            ColSpec(
                Object(
                    [
                        Property("int", DataType.integer),
                        Property("double", DataType.string),
                    ]
                )
            )
        ]
    )

    df = spark.createDataFrame(data, spark_df_schema)
    with pytest.raises(MlflowException, match="Failed to enforce schema"):
        _enforce_schema(df, input_schema)


def test_enforce_schema_spark_dataframe_extra_col(spark):
    spark_df_schema = StructType(
        [StructField("a", ShortType(), True), StructField("b", DoubleType(), True)]
    )

    data = [
        (
            1,  # a
            2.3,  # b
        )
    ]

    input_schema = Schema([ColSpec(DataType.integer, "a")])

    df = spark.createDataFrame(data, spark_df_schema)
    result = _enforce_schema(df, input_schema)
    expected_result = df.drop("b")
    assertDataFrameEqual(result, expected_result)


def test_enforce_schema_spark_dataframe_no_schema(spark):
    data = [
        (
            1,  # a
            2.3,  # b
        )
    ]

    input_schema = Schema(
        [
            ColSpec(DataType.integer, "a"),
            ColSpec(DataType.double, "b"),
        ]
    )

    df = spark.createDataFrame(data, ["a", "b"])
    with pytest.raises(MlflowException, match="Incompatible input types"):
        _enforce_schema(df, input_schema)

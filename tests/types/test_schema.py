import json
import math
import numpy as np
import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema
from mlflow.types.utils import _infer_schema


def test_col_spec():
    a1 = ColSpec("string", "a")
    a2 = ColSpec(DataType.string, "a")
    a3 = ColSpec(DataType.integer, "a")
    assert a1 != a3
    b1 = ColSpec(DataType.string, "b")
    assert b1 != a1
    assert a1 == a2
    with pytest.raises(MlflowException) as ex:
        ColSpec("unsupported")
    assert "Unsupported type 'unsupported'" in ex.value.message
    a4 = ColSpec(**a1.to_dict())
    assert a4 == a1
    assert ColSpec(**json.loads(json.dumps(a1.to_dict()))) == a1
    a5 = ColSpec("string")
    a6 = ColSpec("string", None)
    assert a5 == a6
    assert ColSpec(**json.loads(json.dumps(a5.to_dict()))) == a5


@pytest.fixture
def pandas_df_with_all_types():
    return pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9]),],
            "string": ["a", "b", "c"],
        }
    )


def test_schema_inference_on_dataframe(pandas_df_with_all_types):
    schema = _infer_schema(pandas_df_with_all_types)
    assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])


def test_schema_inference_on_dictionary(pandas_df_with_all_types):
    # test dictionary
    d = {c: pandas_df_with_all_types[c].values for c in pandas_df_with_all_types.columns}
    schema = _infer_schema(d)
    assert dict(zip(schema.column_names(), schema.column_types())) == {
        c: DataType[c] for c in pandas_df_with_all_types.columns
    }
    # test exception is raised if non-numpy data in dictionary
    with pytest.raises(TypeError):
        _infer_schema({"x": 1})
    with pytest.raises(TypeError):
        _infer_schema({"x": [1]})


def test_schema_inference_on_numpy_array(pandas_df_with_all_types):
    # drop int and float as we lose type size information when storing as objects and defaults are
    # 64b.
    pandas_df_with_all_types = pandas_df_with_all_types.drop(columns=["integer", "float"])
    schema = _infer_schema(pandas_df_with_all_types.values)
    assert schema == Schema([ColSpec(x) for x in pandas_df_with_all_types.columns])

    # test objects
    schema = _infer_schema(np.array(["a"], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.string)])
    schema = _infer_schema(np.array([bytes([1])], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(np.array([bytearray([1]), None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(np.array([True, None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.boolean)])
    schema = _infer_schema(np.array([1.1, None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.double)])

    # test bytes
    schema = _infer_schema(np.array([bytes([1])], dtype=np.bytes_))

    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(np.array([bytearray([1])], dtype=np.bytes_))
    assert schema == Schema([ColSpec(DataType.binary)])

    # test string
    schema = _infer_schema(np.array(["a"], dtype=np.str))
    assert schema == Schema([ColSpec(DataType.string)])

    # test boolean
    schema = _infer_schema(np.array([True], dtype=np.bool))
    assert schema == Schema([ColSpec(DataType.boolean)])

    # test ints
    for t in [np.uint8, np.uint16, np.int8, np.int16, np.int32]:
        schema = _infer_schema(np.array([1, 2, 3], dtype=t))
        assert schema == Schema([ColSpec("integer")])

    # test longs
    for t in [np.uint32, np.int64]:
        schema = _infer_schema(np.array([1, 2, 3], dtype=t))
        assert schema == Schema([ColSpec("long")])

    # unsigned long is unsupported
    with pytest.raises(MlflowException):
        _infer_schema(np.array([1, 2, 3], dtype=np.uint64))

    # test floats
    for t in [np.float16, np.float32]:
        schema = _infer_schema(np.array([1.1, 2.2, 3.3], dtype=t))
        assert schema == Schema([ColSpec("float")])

    # test doubles
    schema = _infer_schema(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    assert schema == Schema([ColSpec("double")])

    # unsupported
    if hasattr(np, "float128"):
        with pytest.raises(MlflowException):
            _infer_schema(np.array([1, 2, 3], dtype=np.float128))


def test_that_schema_inference_with_tensors_raises_exception():
    with pytest.raises(MlflowException):
        _infer_schema(np.array([[[1, 2, 3]]], dtype=np.int64))
    with pytest.raises(MlflowException):
        _infer_schema(pd.DataFrame({"x": [np.array([[1, 2, 3]], dtype=np.int64)]}))
    with pytest.raises(MlflowException):
        _infer_schema({"x": np.array([[1, 2, 3]], dtype=np.int64)})


@pytest.mark.large
def test_spark_schema_inference(pandas_df_with_all_types):
    import pyspark
    from pyspark.sql.types import _parse_datatype_string, StructField, StructType

    schema = _infer_schema(pandas_df_with_all_types)
    assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
    spark_session = pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate())
    spark_schema = StructType(
        [StructField(t.name, _parse_datatype_string(t.name), True) for t in schema.column_types()]
    )
    sparkdf = spark_session.createDataFrame(pandas_df_with_all_types, schema=spark_schema)
    schema = _infer_schema(sparkdf)
    assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])


@pytest.mark.large
def test_spark_type_mapping(pandas_df_with_all_types):
    import pyspark
    from pyspark.sql.types import (
        BooleanType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        StringType,
        BinaryType,
    )
    from pyspark.sql.types import StructField, StructType

    assert isinstance(DataType.boolean.to_spark(), BooleanType)
    assert isinstance(DataType.integer.to_spark(), IntegerType)
    assert isinstance(DataType.long.to_spark(), LongType)
    assert isinstance(DataType.float.to_spark(), FloatType)
    assert isinstance(DataType.double.to_spark(), DoubleType)
    assert isinstance(DataType.string.to_spark(), StringType)
    assert isinstance(DataType.binary.to_spark(), BinaryType)
    schema = _infer_schema(pandas_df_with_all_types)
    expected_spark_schema = StructType(
        [StructField(t.name, t.to_spark(), True) for t in schema.column_types()]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()
    spark_session = pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate())
    sparkdf = spark_session.createDataFrame(pandas_df_with_all_types, schema=actual_spark_schema)
    schema2 = _infer_schema(sparkdf)
    assert schema == schema2

    # test unnamed columns
    schema = Schema([ColSpec(col.type) for col in schema.columns])
    expected_spark_schema = StructType(
        [StructField(str(i), t.to_spark(), True) for i, t in enumerate(schema.column_types())]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()

    # test single unnamed column is mapped to just a single spark type
    schema = Schema([ColSpec(DataType.integer)])
    spark_type = schema.as_spark_schema()
    assert isinstance(spark_type, IntegerType)

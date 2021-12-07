import json
import math
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, csc_matrix

from mlflow.exceptions import MlflowException
from mlflow.pyfunc import _enforce_tensor_spec
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from mlflow.types.utils import _infer_schema, _get_tensor_shape


def test_col_spec():
    a1 = ColSpec("string", "a")
    a2 = ColSpec(DataType.string, "a")
    a3 = ColSpec(DataType.integer, "a")
    assert a1 != a3
    b1 = ColSpec(DataType.string, "b")
    assert b1 != a1
    assert a1 == a2
    with pytest.raises(MlflowException, match="Unsupported type 'unsupported'"):
        ColSpec("unsupported")
    a4 = ColSpec(**a1.to_dict())
    assert a4 == a1
    assert ColSpec(**json.loads(json.dumps(a1.to_dict()))) == a1
    a5 = ColSpec("string")
    a6 = ColSpec("string", None)
    assert a5 == a6
    assert ColSpec(**json.loads(json.dumps(a5.to_dict()))) == a5


def test_tensor_spec():
    a1 = TensorSpec(np.dtype("float64"), (-1, 3, 3), "a")
    a2 = TensorSpec(np.dtype("float"), (-1, 3, 3), "a")  # float defaults to float64
    a3 = TensorSpec(np.dtype("float"), [-1, 3, 3], "a")
    a4 = TensorSpec(np.dtype("int"), (-1, 3, 3), "a")
    assert a1 == a2
    assert a1 == a3
    assert a1 != a4
    b1 = TensorSpec(np.dtype("float64"), (-1, 3, 3), "b")
    assert b1 != a1
    with pytest.raises(TypeError, match="Expected `type` to be instance"):
        TensorSpec("Unsupported", (-1, 3, 3), "a")
    with pytest.raises(TypeError, match="Expected `shape` to be instance"):
        TensorSpec(np.dtype("float64"), np.array([-1, 2, 3]), "b")
    with pytest.raises(
        MlflowException,
        match="MLflow does not support size information in flexible numpy data types",
    ):
        TensorSpec(np.dtype("<U10"), (-1,), "b")

    a5 = TensorSpec.from_json_dict(**a1.to_dict())
    assert a5 == a1
    assert TensorSpec.from_json_dict(**json.loads(json.dumps(a1.to_dict()))) == a1
    a6 = TensorSpec(np.dtype("float64"), (-1, 3, 3))
    a7 = TensorSpec(np.dtype("float64"), (-1, 3, 3), None)
    assert a6 == a7
    assert TensorSpec.from_json_dict(**json.loads(json.dumps(a6.to_dict()))) == a6


@pytest.fixture
def pandas_df_with_all_types():
    df = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9])],
            "string": ["a", "b", "c"],
            "datetime": [
                np.datetime64("2021-01-01"),
                np.datetime64("2021-02-02"),
                np.datetime64("2021-03-03"),
            ],
            "boolean_ext": [True, False, True],
            "integer_ext": [1, 2, 3],
            "string_ext": ["a", "b", "c"],
        }
    )
    df["boolean_ext"] = df["boolean_ext"].astype("boolean")
    df["integer_ext"] = df["integer_ext"].astype("Int64")
    df["string_ext"] = df["string_ext"].astype("string")
    return df


@pytest.fixture
def dict_of_ndarrays():
    return {
        "1D": np.arange(0, 12, 0.5),
        "2D": np.arange(0, 12, 0.5).reshape(3, 8),
        "3D": np.arange(0, 12, 0.5).reshape(2, 3, 4),
        "4D": np.arange(0, 12, 0.5).reshape(3, 2, 2, 2),
    }


def test_schema_creation():
    # can create schema with named col specs
    Schema([ColSpec("double", "a"), ColSpec("integer", "b")])

    # can create schema with unnamed col specs
    Schema([ColSpec("double"), ColSpec("integer")])

    # can create schema with multiple named tensor specs
    Schema([TensorSpec(np.dtype("float64"), (-1,), "a"), TensorSpec(np.dtype("uint8"), (-1,), "b")])

    # can create schema with single unnamed tensor spec
    Schema([TensorSpec(np.dtype("float64"), (-1,))])

    # combination of tensor and col spec is not allowed
    with pytest.raises(MlflowException, match="Please choose one of"):
        Schema([TensorSpec(np.dtype("float64"), (-1,)), ColSpec("double")])

    # combination of named and unnamed inputs is not allowed
    with pytest.raises(
        MlflowException, match="Creating Schema with a combination of named and unnamed inputs"
    ):
        Schema(
            [TensorSpec(np.dtype("float64"), (-1,), "blah"), TensorSpec(np.dtype("float64"), (-1,))]
        )

    with pytest.raises(
        MlflowException, match="Creating Schema with a combination of named and unnamed inputs"
    ):
        Schema([ColSpec("double", "blah"), ColSpec("double")])

    # multiple unnamed tensor specs is not allowed
    with pytest.raises(
        MlflowException, match="Creating Schema with multiple unnamed TensorSpecs is not supported"
    ):
        Schema([TensorSpec(np.dtype("double"), (-1,)), TensorSpec(np.dtype("double"), (-1,))])


def test_get_schema_type(dict_of_ndarrays):
    schema = _infer_schema(dict_of_ndarrays)
    assert ["float64"] * 4 == schema.numpy_types()
    with pytest.raises(MlflowException, match="TensorSpec only supports numpy types"):
        schema.column_types()
    with pytest.raises(MlflowException, match="TensorSpec only supports numpy types"):
        schema.pandas_types()
    with pytest.raises(MlflowException, match="TensorSpec cannot be converted to spark dataframe"):
        schema.as_spark_schema()


def test_schema_inference_on_dataframe(pandas_df_with_all_types):
    basic_types = pandas_df_with_all_types.drop(
        columns=["boolean_ext", "integer_ext", "string_ext"]
    )
    schema = _infer_schema(basic_types)
    assert schema == Schema([ColSpec(x, x) for x in basic_types.columns])

    ext_types = pandas_df_with_all_types[["boolean_ext", "integer_ext", "string_ext"]].copy()
    expected_schema = Schema(
        [
            ColSpec(DataType.boolean, "boolean_ext"),
            ColSpec(DataType.long, "integer_ext"),
            ColSpec(DataType.string, "string_ext"),
        ]
    )
    schema = _infer_schema(ext_types)
    assert schema == expected_schema


def test_schema_inference_on_pandas_series():
    # test objects
    schema = _infer_schema(pd.Series(np.array(["a"], dtype=np.object)))
    assert schema == Schema([ColSpec(DataType.string)])
    schema = _infer_schema(pd.Series(np.array([bytes([1])], dtype=np.object)))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(pd.Series(np.array([bytearray([1]), None], dtype=np.object)))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(pd.Series(np.array([True, None], dtype=np.object)))
    assert schema == Schema([ColSpec(DataType.string)])
    schema = _infer_schema(pd.Series(np.array([1.1, None], dtype=np.object)))
    assert schema == Schema([ColSpec(DataType.double)])

    # test bytes
    schema = _infer_schema(pd.Series(np.array([bytes([1])], dtype=np.bytes_)))
    assert schema == Schema([ColSpec(DataType.binary)])

    # test string
    schema = _infer_schema(pd.Series(np.array(["a"], dtype=np.str)))
    assert schema == Schema([ColSpec(DataType.string)])

    # test boolean
    schema = _infer_schema(pd.Series(np.array([True], dtype=np.bool)))
    assert schema == Schema([ColSpec(DataType.boolean)])

    # test ints
    for t in [np.uint8, np.uint16, np.int8, np.int16, np.int32]:
        schema = _infer_schema(pd.Series(np.array([1, 2, 3], dtype=t)))
        assert schema == Schema([ColSpec("integer")])

    # test longs
    for t in [np.uint32, np.int64]:
        schema = _infer_schema(pd.Series(np.array([1, 2, 3], dtype=t)))
        assert schema == Schema([ColSpec("long")])

    # unsigned long is unsupported
    with pytest.raises(MlflowException, match="Unsupported numpy data type"):
        _infer_schema(pd.Series(np.array([1, 2, 3], dtype=np.uint64)))

    # test floats
    for t in [np.float16, np.float32]:
        schema = _infer_schema(pd.Series(np.array([1.1, 2.2, 3.3], dtype=t)))
        assert schema == Schema([ColSpec("float")])

    # test doubles
    schema = _infer_schema(pd.Series(np.array([1.1, 2.2, 3.3], dtype=np.float64)))
    assert schema == Schema([ColSpec("double")])

    # test datetime
    schema = _infer_schema(
        pd.Series(
            np.array(
                ["2021-01-01 00:00:00", "2021-02-02 00:00:00", "2021-03-03 12:00:00"],
                dtype="datetime64",
            )
        )
    )
    assert schema == Schema([ColSpec("datetime")])

    # unsupported
    if hasattr(np, "float128"):
        with pytest.raises(MlflowException, match="Unsupported numpy data type"):
            _infer_schema(pd.Series(np.array([1, 2, 3], dtype=np.float128)))


def test_get_tensor_shape(dict_of_ndarrays):
    assert all([-1 == _get_tensor_shape(tensor)[0] for tensor in dict_of_ndarrays.values()])

    data = dict_of_ndarrays["4D"]
    # Specify variable dimension
    for i in range(-4, 4):
        assert _get_tensor_shape(data, i)[i] == -1

    # Specify None
    assert all([_get_tensor_shape(data, None) != -1])

    # Out of bounds
    with pytest.raises(
        MlflowException, match="The specified variable_dimension 10 is out of bounds"
    ):
        _get_tensor_shape(data, 10)
    with pytest.raises(
        MlflowException, match="The specified variable_dimension -10 is out of bounds"
    ):
        _get_tensor_shape(data, -10)

    with pytest.raises(TypeError, match="Data in the dictionary must be of type numpy.ndarray"):
        _infer_schema({"x": 1})


@pytest.fixture
def dict_of_sparse_matrix():
    return {
        "csc": csc_matrix(np.arange(0, 12, 0.5).reshape(3, 8)),
        "csr": csr_matrix(np.arange(0, 12, 0.5).reshape(3, 8)),
    }


def test_get_sparse_matrix_data_type_and_shape(dict_of_sparse_matrix):
    for sparse_matrix in dict_of_sparse_matrix.values():
        schema = _infer_schema(sparse_matrix)
        assert schema.numpy_types() == ["float64"]
        assert _get_tensor_shape(sparse_matrix) == (-1, 8)


def test_schema_inference_on_dictionary(dict_of_ndarrays):
    # test dictionary
    schema = _infer_schema(dict_of_ndarrays)
    assert schema == Schema(
        [
            TensorSpec(tensor.dtype, _get_tensor_shape(tensor), name)
            for name, tensor in dict_of_ndarrays.items()
        ]
    )
    # test exception is raised if non-numpy data in dictionary
    match = "Data in the dictionary must be of type numpy.ndarray"
    with pytest.raises(TypeError, match=match):
        _infer_schema({"x": 1})
    with pytest.raises(TypeError, match=match):
        _infer_schema({"x": [1]})


def test_schema_inference_on_basic_numpy(pandas_df_with_all_types):
    for col in pandas_df_with_all_types:
        data = pandas_df_with_all_types[col].to_numpy()
        schema = _infer_schema(data)
        assert schema == Schema([TensorSpec(type=data.dtype, shape=(-1,))])


# Todo: arjundc : Remove _enforce_tensor_spec and move to its own test file.
def test_all_numpy_dtypes():
    def test_dtype(nparray, dtype):
        schema = _infer_schema(nparray)
        assert schema == Schema([TensorSpec(np.dtype(dtype), (-1,))])
        spec = schema.inputs[0]
        recreated_spec = TensorSpec.from_json_dict(**spec.to_dict())
        assert spec == recreated_spec
        enforced_array = _enforce_tensor_spec(nparray, spec)
        assert isinstance(enforced_array, np.ndarray)

    bool_ = ["bool", "bool_", "bool8"]
    object_ = ["object"]
    signed_int = [
        "byte",
        "int8",
        "short",
        "int16",
        "intc",
        "int32",
        "int_",
        "int",
        "intp",
        "int64",
        "longlong",
    ]
    unsigned_int = [
        "ubyte",
        "uint8",
        "ushort",
        "uint16",
        "uintc",
        "uint32",
        "uint",
        "uintp",
        "uint64",
        "ulonglong",
    ]
    floating = ["half", "float16", "single", "float32", "double", "float_", "float64"]
    complex_ = [
        "csingle",
        "singlecomplex",
        "complex64",
        "cdouble",
        "cfloat",
        "complex_",
        "complex128",
    ]
    bytes_ = ["bytes_", "string_"]
    str_ = ["str_", "unicode_"]
    platform_dependent = [
        # Complex
        "clongdouble",
        "clongfloat",
        "longcomplex",
        "complex256",
        # Float
        "longdouble",
        "longfloat",
        "float128",
    ]

    # test boolean
    for dtype in bool_:
        test_dtype(np.array([True, False, True], dtype=dtype), dtype)
        test_dtype(np.array([123, 0, -123], dtype=dtype), dtype)

    # test object
    for dtype in object_:
        test_dtype(np.array([True, False, True], dtype=dtype), dtype)
        test_dtype(np.array([123, 0, -123.544], dtype=dtype), dtype)
        test_dtype(np.array(["test", "this", "type"], dtype=dtype), dtype)
        test_dtype(np.array(["test", 123, "type"], dtype=dtype), dtype)
        test_dtype(np.array(["test", 123, 234 + 543j], dtype=dtype), dtype)

    # test signedInt_
    for dtype in signed_int:
        test_dtype(np.array([1, 2, 3, -5], dtype=dtype), dtype)

    # test unsignedInt_
    for dtype in unsigned_int:
        test_dtype(np.array([1, 2, 3, 5], dtype=dtype), dtype)

    # test floating
    for dtype in floating:
        test_dtype(np.array([1.1, -2.2, 3.3, 5.12], dtype=dtype), dtype)

    # test complex
    for dtype in complex_:
        test_dtype(np.array([1 + 2j, -2.2 - 3.6j], dtype=dtype), dtype)

    # test bytes_
    for dtype in bytes_:
        test_dtype(np.array([bytes([1, 255, 12, 34])], dtype=dtype), dtype)
    # Explicitly giving size information for flexible dtype bytes
    test_dtype(np.array([bytes([1, 255, 12, 34])], dtype="S10"), "S")
    test_dtype(np.array([bytes([1, 255, 12, 34])], dtype="S10"), "bytes")

    # str_
    for dtype in str_:
        test_dtype(np.array(["m", "l", "f", "l", "o", "w"], dtype=dtype), dtype)
        test_dtype(np.array(["mlflow"], dtype=dtype), dtype)
        test_dtype(np.array(["mlflow is the best"], dtype=dtype), dtype)
    # Explicitly giving size information for flexible dtype str_
    test_dtype(np.array(["a", "bc", "def"], dtype="U16"), "str")
    test_dtype(np.array(["a", "bc", "def"], dtype="U16"), "U")

    # test datetime
    test_dtype(
        np.array(
            ["2021-01-01 00:00:00", "2021-02-02 00:00:00", "2021-03-03 12:00:00"],
            dtype="datetime64",
        ),
        "datetime64[s]",
    )

    # platform_dependent
    for dtype in platform_dependent:
        if hasattr(np, dtype):
            test_dtype(np.array([1.1, -2.2, 3.3, 5.12], dtype=dtype), dtype)


@pytest.mark.large
def test_spark_schema_inference(pandas_df_with_all_types):
    import pyspark
    from pyspark.sql.types import _parse_datatype_string, StructField, StructType

    pandas_df_with_all_types = pandas_df_with_all_types.drop(
        columns=["boolean_ext", "integer_ext", "string_ext"]
    )
    schema = _infer_schema(pandas_df_with_all_types)
    assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
    spark_session = pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate())

    struct_fields = []
    for t in schema.column_types():
        # pyspark _parse_datatype_string() expects "timestamp" instead of "datetime"
        if t == DataType.datetime:
            struct_fields.append(StructField("datetime", _parse_datatype_string("timestamp"), True))
        else:
            struct_fields.append(StructField(t.name, _parse_datatype_string(t.name), True))
    spark_schema = StructType(struct_fields)
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
        TimestampType,
    )
    from pyspark.sql.types import StructField, StructType

    assert isinstance(DataType.boolean.to_spark(), BooleanType)
    assert isinstance(DataType.integer.to_spark(), IntegerType)
    assert isinstance(DataType.long.to_spark(), LongType)
    assert isinstance(DataType.float.to_spark(), FloatType)
    assert isinstance(DataType.double.to_spark(), DoubleType)
    assert isinstance(DataType.string.to_spark(), StringType)
    assert isinstance(DataType.binary.to_spark(), BinaryType)
    assert isinstance(DataType.datetime.to_spark(), TimestampType)
    pandas_df_with_all_types = pandas_df_with_all_types.drop(
        columns=["boolean_ext", "integer_ext", "string_ext"]
    )
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
    schema = Schema([ColSpec(col.type) for col in schema.inputs])
    expected_spark_schema = StructType(
        [StructField(str(i), t.to_spark(), True) for i, t in enumerate(schema.column_types())]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()

    # test single unnamed column is mapped to just a single spark type
    schema = Schema([ColSpec(DataType.integer)])
    spark_type = schema.as_spark_schema()
    assert isinstance(spark_type, IntegerType)

import datetime
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.types as T
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from mlflow.exceptions import MlflowException
from mlflow.models import rag_signatures
from mlflow.models.utils import _enforce_tensor_spec
from mlflow.pyfunc import _parse_spark_datatype
from mlflow.types import DataType
from mlflow.types.schema import (
    AnyType,
    Array,
    ColSpec,
    Object,
    ParamSchema,
    ParamSpec,
    Property,
    Schema,
    TensorSpec,
    convert_dataclass_to_schema,
)
from mlflow.types.utils import (
    MULTIPLE_TYPES_ERROR_MSG,
    _get_tensor_shape,
    _infer_colspec_type,
    _infer_param_schema,
    _infer_schema,
    _validate_input_dictionary_contains_only_strings_and_lists_of_strings,
)


@pytest.fixture(scope="module")
def spark():
    with pyspark.sql.SparkSession.builder.getOrCreate() as s:
        yield s


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
    with pytest.raises(TypeError, match="Expected `dtype` to be instance"):
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


def test_schema_creation_errors():
    with pytest.raises(MlflowException, match=r"Creating Schema with empty inputs is not allowed."):
        Schema([])

    with pytest.raises(MlflowException, match=r"Inputs of Schema must be a list, got type dict"):
        Schema({"col1": ColSpec(DataType.string)})


def test_get_schema_type(dict_of_ndarrays):
    schema = _infer_schema(dict_of_ndarrays)
    assert ["float64"] * 4 == schema.numpy_types()
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
    schema = _infer_schema(pd.Series(np.array(["a"], dtype=object)))
    assert schema == Schema([ColSpec(DataType.string)])
    schema = _infer_schema(pd.Series(np.array([bytes([1])], dtype=object)))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = _infer_schema(pd.Series(np.array([bytearray([1]), None], dtype=object), name="input"))
    assert schema == Schema([ColSpec(DataType.binary, name="input", required=False)])
    schema = _infer_schema(pd.Series(np.array([1.1, None], dtype=object), name="input"))
    assert schema == Schema([ColSpec(DataType.double, name="input", required=False)])

    # test bytes
    schema = _infer_schema(pd.Series(np.array([bytes([1])], dtype=np.bytes_)))
    assert schema == Schema([ColSpec(DataType.binary)])

    # test string
    schema = _infer_schema(pd.Series(np.array(["a"], dtype=str)))
    assert schema == Schema([ColSpec(DataType.string)])

    # test boolean
    schema = _infer_schema(pd.Series(np.array([True], dtype=bool)))
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

    # test names
    s = pd.Series([1, 2, 3])
    if hasattr(s, "name"):
        s.rename("test", inplace=True)
        assert "test" in _infer_schema(s).input_names()
        assert len(_infer_schema(s).input_names()) == 1


def test_get_tensor_shape(dict_of_ndarrays):
    assert all(-1 == _get_tensor_shape(tensor)[0] for tensor in dict_of_ndarrays.values())

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


@pytest.mark.parametrize(
    ("data", "schema"),
    [
        (
            {"a": "b", "c": "d"},
            Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "c")]),
        ),
        (
            {"a": ["a", "b"], "b": ["c", "d"]},
            Schema([ColSpec(Array(DataType.string), "a"), ColSpec(Array(DataType.string), "b")]),
        ),
        (
            {"a": "a", "b": ["a", "b"]},
            Schema([ColSpec(DataType.string, "a"), ColSpec(Array(DataType.string), "b")]),
        ),
        (
            [{"a": 1}, {"a": 1, "b": ["a", "b"]}],
            Schema(
                [ColSpec(DataType.long, "a"), ColSpec(Array(DataType.string), "b", required=False)]
            ),
        ),
    ],
)
def test_schema_inference_on_dictionary_of_strings(data, schema):
    assert _infer_schema(data) == schema


def test_schema_inference_on_list_with_errors():
    data = [{"a": 1, "b": "c"}, {"a": 1, "b": ["a", "b"]}]
    with pytest.raises(MlflowException, match=re.escape(MULTIPLE_TYPES_ERROR_MSG)):
        _infer_schema(data)


def test_schema_inference_validating_dictionary_keys():
    for data in [{1.7: "a", "b": "c"}, {12.4: "c", "d": "e"}]:
        with pytest.raises(MlflowException, match="The dictionary keys are not all strings."):
            _infer_schema(data)


def test_schema_inference_on_lists_with_errors():
    with pytest.raises(MlflowException, match=re.escape(MULTIPLE_TYPES_ERROR_MSG)):
        _infer_schema(["a", 1])

    with pytest.raises(MlflowException, match=re.escape(MULTIPLE_TYPES_ERROR_MSG)):
        _infer_schema(["a", ["b", "c"]])


def test_schema_inference_on_list_of_dicts():
    schema = _infer_schema([{"a": "a", "b": "b"}, {"a": "a", "b": "b"}])
    assert schema == Schema([ColSpec(DataType.string, "a"), ColSpec(DataType.string, "b")])

    schema = _infer_schema([{"a": 1}, {"b": "a"}])
    assert schema == Schema(
        [
            ColSpec(DataType.long, "a", required=False),
            ColSpec(DataType.string, "b", required=False),
        ]
    )


def test_dict_input_valid_checks_on_keys():
    match = "The dictionary keys are not all strings or "
    # User-defined keys

    class Hashable:
        def __init__(self, x: str, y: str):
            self.x = x
            self.y = y

        def __hash__(self):
            return hash((self.x, self.y))

        def __eq__(self, other):
            return isinstance(other, Hashable) and self.x == other.x and self.y == other.y

    hash_obj_1 = Hashable("some", "custom")
    hash_obj_2 = Hashable("some", "other")
    custom_hashable_dict = {hash_obj_1: "value", hash_obj_2: "other_value"}

    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(custom_hashable_dict)

    # keys are floats
    float_keys_dict = {1.1: "a", 2.2: "b"}

    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(float_keys_dict)

    # keys are bool
    bool_keys_dict = {True: "a", False: "b"}

    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(bool_keys_dict)
    # keys are tuples
    tuple_keys_dict = {("a", "b"): "a", ("a", "c"): "b"}

    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(tuple_keys_dict)

    # keys are frozenset
    frozen_set_dict = {frozenset({"a", "b"}): "a", frozenset({"b", "c"}): "b"}

    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(frozen_set_dict)


def test_dict_input_valid_checks_on_values():
    match = "Invalid values in dictionary. If passing a dictionary containing strings"

    list_of_ints = {"a": [1, 2, 3], "b": [1, 2, 3]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_ints)

    list_of_floats = {"a": [1.1, 1.2], "b": [1.1, 2.2]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_floats)

    list_of_dics = {"a": [{"a": "b"}, {"b": "c"}], "b": [{"a": "c"}, {"b": "d"}]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_dics)

    list_of_lists = {"a": [["b", "c"], ["d", "e"]], "b": [["e", "f"], ["g", "h"]]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_lists)

    list_of_set = {"a": [{"b", "c"}, {"d", "e"}], "b": [{"a", "c"}, {"d", "f"}]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_set)

    list_of_frozen_set = {
        "a": [frozenset({"b", "c"}), frozenset({"d", "e"})],
        "b": [frozenset({"a", "c"}), frozenset({"d", "f"})],
    }
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_frozen_set)

    list_of_bool = {"a": [True, True, False], "b": [False, False, True]}
    with pytest.raises(MlflowException, match=match):
        _validate_input_dictionary_contains_only_strings_and_lists_of_strings(list_of_bool)


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

    bool_ = ["bool", "bool_"]
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
    floating = ["half", "float16", "single", "float32", "double", "float64"]
    complex_ = [
        "csingle",
        "complex64",
        "cdouble",
        "complex128",
    ]
    bytes_ = ["bytes_"]
    str_ = ["str_"]
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


def test_spark_schema_inference(pandas_df_with_all_types):
    import pyspark
    from pyspark.sql.types import StructField, StructType

    pandas_df_with_all_types = pandas_df_with_all_types.drop(
        columns=["boolean_ext", "integer_ext", "string_ext"]
    )
    schema = _infer_schema(pandas_df_with_all_types)
    assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
    with pyspark.sql.SparkSession.builder.getOrCreate() as spark:
        struct_fields = []
        for t in schema.input_types():
            if t == DataType.datetime:
                struct_fields.append(
                    StructField("datetime", _parse_spark_datatype("timestamp"), True)
                )
            else:
                struct_fields.append(StructField(t.name, _parse_spark_datatype(t.name), True))
        spark_schema = StructType(struct_fields)
        sparkdf = spark.createDataFrame(pandas_df_with_all_types, schema=spark_schema)
        schema = _infer_schema(sparkdf)
        assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])


def test_spark_schema_inference_complex(spark):
    df = spark.createDataFrame(
        [(["arr"],)],
        schema=T.StructType([T.StructField("arr", T.ArrayType(T.StringType()))]),
    )
    schema = _infer_schema(df)
    assert schema == Schema([ColSpec(Array(DataType.string), "arr")])

    df = spark.createDataFrame(
        [({"str": "s", "str_nullable": None},)],
        schema=T.StructType(
            [
                T.StructField(
                    "obj",
                    T.StructType(
                        [
                            T.StructField("str", T.StringType(), nullable=False),
                            T.StructField("str_nullable", T.StringType(), nullable=True),
                        ]
                    ),
                )
            ]
        ),
    )
    schema = _infer_schema(df)
    assert schema == Schema(
        [
            ColSpec(
                Object(
                    [
                        Property("str", DataType.string, required=True),
                        Property("str_nullable", DataType.string, required=False),
                    ]
                ),
                "obj",
            )
        ]
    )

    df = spark.createDataFrame(
        [([{"str": "s"}],)],
        schema=T.StructType(
            [
                T.StructField(
                    "obj_arr",
                    T.ArrayType(
                        T.StructType([T.StructField("str", T.StringType(), nullable=False)])
                    ),
                )
            ]
        ),
    )

    schema = _infer_schema(df)
    assert schema == Schema(
        [
            ColSpec(
                Array(Object([Property("str", DataType.string, required=True)])),
                "obj_arr",
            )
        ]
    )


def test_spark_type_mapping(pandas_df_with_all_types):
    import pyspark
    from pyspark.sql.types import (
        BinaryType,
        BooleanType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

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
        [StructField(t.name, t.to_spark(), False) for t in schema.input_types()]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()
    with pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate()) as spark_session:
        sparkdf = spark_session.createDataFrame(
            pandas_df_with_all_types, schema=actual_spark_schema
        )
        schema2 = _infer_schema(sparkdf)
        assert schema == schema2

    # test unnamed columns
    schema = Schema([ColSpec(col.type) for col in schema.inputs])
    expected_spark_schema = StructType(
        [StructField(str(i), t.to_spark(), False) for i, t in enumerate(schema.input_types())]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()

    # test single unnamed column is mapped to just a single spark type
    schema = Schema([ColSpec(DataType.integer)])
    spark_type = schema.as_spark_schema()
    assert isinstance(spark_type, IntegerType)


def test_enforce_tensor_spec_variable_signature():
    standard_array = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]], dtype=np.int32)
    ragged_array = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3]]], dtype=object)
    inferred_schema = _infer_schema(ragged_array)
    inferred_spec = inferred_schema.inputs[0]
    assert inferred_spec.shape == (-1,)
    assert inferred_spec.type == np.dtype(object)

    result_array = _enforce_tensor_spec(standard_array, inferred_spec)
    np.testing.assert_array_equal(standard_array, result_array)
    result_array = _enforce_tensor_spec(ragged_array, inferred_spec)
    np.testing.assert_array_equal(ragged_array, result_array)

    manual_spec = TensorSpec(np.dtype(np.int32), (-1, -1, 3))
    result_array = _enforce_tensor_spec(standard_array, manual_spec)
    np.testing.assert_array_equal(standard_array, result_array)
    result_array = _enforce_tensor_spec(ragged_array, manual_spec)
    np.testing.assert_array_equal(ragged_array, result_array)

    standard_spec = _infer_schema(standard_array).inputs[0]
    assert standard_spec.shape == (-1, 2, 3)

    result_array = _enforce_tensor_spec(standard_array, standard_spec)
    np.testing.assert_array_equal(standard_array, result_array)
    with pytest.raises(
        MlflowException,
        match=re.escape(r"Shape of input (2,) does not match expected shape (-1, 2, 3)."),
    ):
        _enforce_tensor_spec(ragged_array, standard_spec)


def test_datatype_type_check():
    assert DataType.check_type(DataType.string, "string")

    integer_type = DataType.integer
    assert DataType.check_type(integer_type, 1)
    assert DataType.check_type(integer_type, np.int32(1))
    assert not DataType.check_type(integer_type, np.int64(1))
    # Note that isinstance(True, int) returns True
    assert not DataType.check_type(integer_type, True)

    long_type = DataType.long
    assert DataType.check_type(long_type, 1)
    assert DataType.check_type(long_type, np.int64(1))
    assert not DataType.check_type(long_type, np.int32(1))

    bool_type = DataType.boolean
    assert DataType.check_type(bool_type, True)
    assert DataType.check_type(bool_type, np.bool_(True))
    assert not DataType.check_type(bool_type, 1)

    double_type = DataType.double
    assert DataType.check_type(double_type, 1.0)
    assert DataType.check_type(double_type, np.float64(1.0))
    assert not DataType.check_type(double_type, np.float32(1.0))

    float_type = DataType.float
    assert DataType.check_type(float_type, 1.0)
    assert DataType.check_type(float_type, np.float32(1.0))
    assert not DataType.check_type(float_type, np.float64(1.0))

    datetime_type = DataType.datetime
    assert DataType.check_type(datetime_type, datetime.date(2023, 6, 26))
    assert DataType.check_type(datetime_type, np.datetime64("2023-06-26 00:00:00"))
    assert not DataType.check_type(datetime_type, "2023-06-26 00:00:00")


def test_param_schema_find_duplicates():
    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param1']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
            ]
        )

    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param1']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
                ParamSpec("param1", DataType.string, "default1", None),
            ]
        )

    with pytest.raises(
        MlflowException, match=re.escape("Duplicated parameters found in schema: ['param3']")
    ):
        ParamSchema(
            [
                ParamSpec("param1", DataType.string, "default1", None),
                ParamSpec("param2", DataType.string, "default2", None),
                ParamSpec("param3", DataType.string, "default3", None),
                ParamSpec("param3", DataType.string, "default3", None),
            ]
        )


def test_param_spec_to_and_from_dict():
    spec = ParamSpec("str_param", DataType.string, "str_a", None)
    assert spec.to_dict() == {
        "name": "str_param",
        "type": "string",
        "default": "str_a",
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("str_array", DataType.string, ["str_a", "str_b"], (-1,))
    assert spec.to_dict() == {
        "name": "str_array",
        "type": "string",
        "default": ["str_a", "str_b"],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("int_param", DataType.integer, np.int32(1), None)
    assert spec.to_dict() == {
        "name": "int_param",
        "type": "integer",
        "default": 1,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("int_array", DataType.integer, [np.int32(1), np.int32(2)], (-1,))
    assert spec.to_dict() == {
        "name": "int_array",
        "type": "integer",
        "default": [1, 2],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("bool_param", DataType.boolean, True, None)
    assert spec.to_dict() == {
        "name": "bool_param",
        "type": "boolean",
        "default": True,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("bool_array", DataType.boolean, [True, False], (-1,))
    assert spec.to_dict() == {
        "name": "bool_array",
        "type": "boolean",
        "default": [True, False],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("double_param", DataType.double, 1.0, None)
    assert spec.to_dict() == {
        "name": "double_param",
        "type": "double",
        "default": 1.0,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("double_array", DataType.double, [1.0, 2.0], (-1,))
    assert spec.to_dict() == {
        "name": "double_array",
        "type": "double",
        "default": [1.0, 2.0],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("float_param", DataType.float, np.float32(0.1), None)
    assert spec.to_dict() == {
        "name": "float_param",
        "type": "float",
        "default": float(np.float32(0.1)),
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("float_array", DataType.float, [np.float32(0.1), np.float32(0.2)], (-1,))
    assert spec.to_dict() == {
        "name": "float_array",
        "type": "float",
        "default": [float(np.float32(0.1)), float(np.float32(0.2))],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("long_param", DataType.long, 100, None)
    assert spec.to_dict() == {
        "name": "long_param",
        "type": "long",
        "default": 100,
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec("long_array", DataType.long, [100, 200], (-1,))
    assert spec.to_dict() == {
        "name": "long_array",
        "type": "long",
        "default": [100, 200],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec(
        "datetime_param", DataType.datetime, np.datetime64("2023-06-26 00:00:00"), None
    )
    assert spec.to_dict() == {
        "name": "datetime_param",
        "type": "datetime",
        "default": "2023-06-26T00:00:00",
        "shape": None,
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec

    spec = ParamSpec(
        "datetime_array",
        DataType.datetime,
        [np.datetime64("2023-06-26 00:00:00"), np.datetime64("2023-06-27 00:00:00")],
        (-1,),
    )
    assert spec.to_dict() == {
        "name": "datetime_array",
        "type": "datetime",
        "default": ["2023-06-26T00:00:00", "2023-06-27T00:00:00"],
        "shape": (-1,),
    }
    assert ParamSpec.from_json_dict(**json.loads(json.dumps(spec.to_dict()))) == spec


def test_param_spec_from_dict_backward_compatibility():
    spec = ParamSpec("str_param", DataType.string, "str_a", None)
    spec_json = json.dumps(
        {
            "name": "str_param",
            "dtype": "string",
            "default": "str_a",
            "shape": None,
        }
    )
    assert ParamSpec.from_json_dict(**json.loads(spec_json)) == spec


def test_infer_param_schema():
    test_params = {
        "str_param": "str_a",
        "int_param": np.int32(1),
        "bool_param": True,
        "double_param": 1.0,
        "float_param": np.float32(0.1),
        "long_param": np.int64(100),
        "datetime_param": np.datetime64("2023-06-26 00:00:00"),
        "str_list": ["a", "b", "c"],
        "bool_list": [True, False],
        "double_array": np.array([1.0, 2.0]),
        "float_array": np.array([np.float32(0.1), np.float32(0.2)]),
        "long_array": np.array([np.int64(100), np.int64(200)]),
        "datetime_array": np.array([datetime.date(2023, 6, 26)]),
        "str_array": np.array(["a", "b", "c"]),
        "bool_array": np.array([True, False]),
        "int_array": np.array([np.int32(1), np.int32(2)]),
    }
    test_schema = ParamSchema(
        [
            ParamSpec("str_param", DataType.string, "str_a", None),
            ParamSpec("int_param", DataType.integer, np.int32(1), None),
            ParamSpec("bool_param", DataType.boolean, True, None),
            ParamSpec("double_param", DataType.double, 1.0, None),
            ParamSpec("float_param", DataType.float, np.float32(0.1), None),
            ParamSpec("long_param", DataType.long, 100, None),
            ParamSpec(
                "datetime_param", DataType.datetime, np.datetime64("2023-06-26 00:00:00"), None
            ),
            ParamSpec("str_list", DataType.string, ["a", "b", "c"], (-1,)),
            ParamSpec("bool_list", DataType.boolean, [True, False], (-1,)),
            ParamSpec("double_array", DataType.double, [1.0, 2.0], (-1,)),
            ParamSpec("float_array", DataType.float, [np.float32(0.1), np.float32(0.2)], (-1,)),
            ParamSpec("long_array", DataType.long, [100, 200], (-1,)),
            ParamSpec("datetime_array", DataType.datetime, [datetime.date(2023, 6, 26)], (-1,)),
            ParamSpec("str_array", DataType.string, ["a", "b", "c"], (-1,)),
            ParamSpec("bool_array", DataType.boolean, [True, False], (-1,)),
            ParamSpec("int_array", DataType.integer, [1, 2], (-1,)),
        ]
    )
    assert _infer_param_schema(test_params) == test_schema

    assert _infer_param_schema({"datetime_param": datetime.date(2023, 6, 26)}) == ParamSchema(
        [ParamSpec("datetime_param", DataType.datetime, datetime.date(2023, 6, 26), None)]
    )

    # Raise error if parameters is not dictionary
    with pytest.raises(MlflowException, match=r"Expected parameters to be dict, got list"):
        _infer_param_schema(["a", "str_a", "b", 1])

    # Raise error if parameter is bytes
    with pytest.raises(MlflowException, match=r"Binary type is not supported for parameters"):
        _infer_param_schema({"a": b"str_a"})

    # Raise error for invalid parameters types - tuple, 2D array
    test_parameters = {
        "a": "str_a",
        "b": (1, 2, 3),
        "c": True,
        "d": [[1, 2], [3, 4]],
        "e": [[[1, 2], [3], []]],
    }
    with pytest.raises(MlflowException, match=r".*") as e:
        _infer_param_schema(test_parameters)
    assert e.match(r"Failed to infer schema for parameters: ")
    assert e.match(
        re.escape(
            "('b', (1, 2, 3), MlflowException('Expected parameters "
            "to be 1D array or scalar, got tuple'))"
        )
    )
    assert e.match(
        re.escape(
            "('d', [[1, 2], [3, 4]], MlflowException('Expected parameters "
            "to be 1D array or scalar, got 2D array'))"
        )
    )
    assert e.match(
        re.escape(
            "('e', [[[1, 2], [3], []]], MlflowException('Expected parameters "
            "to be 1D array or scalar, got 3D array'))"
        )
    )


def test_param_schema_for_dicts():
    data = {
        "config": {
            "thread_id": "1",
            "batch_size": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
    }
    schema = _infer_param_schema(data)
    object_schema = Object(
        [
            Property("thread_id", DataType.string),
            Property("batch_size", DataType.long),
            Property(
                "messages",
                Array(
                    Object(
                        [
                            Property("role", DataType.string),
                            Property("content", DataType.string),
                        ]
                    )
                ),
            ),
        ]
    )
    assert schema == ParamSchema(
        [
            ParamSpec(
                "config",
                object_schema,
                data["config"],
            )
        ]
    )
    assert schema.to_dict() == [
        {
            "name": "config",
            "default": data["config"],
            "shape": None,
        }
        | object_schema.to_dict()
    ]
    assert ParamSchema.from_json(json.dumps(schema.to_dict())) == schema


def test_infer_param_schema_with_errors():
    with pytest.raises(
        MlflowException, match=r"Expected parameters to be 1D array or scalar, got Series"
    ):
        _infer_param_schema({"a": pd.Series([1, 2, 3])})


def test_object_construction_with_errors():
    with pytest.raises(MlflowException, match=r"Expected properties to be a list, got type dict"):
        Object({"p1": Property("p1", DataType.string)})

    with pytest.raises(
        MlflowException, match=r"Creating Object with empty properties is not allowed."
    ):
        Object([])

    properties = [
        Property("p1", DataType.string),
        Property("p2", DataType.binary),
        {"invalid_type": "value"},
    ]
    with pytest.raises(MlflowException, match=r"Expected values to be instance of Property"):
        Object(properties)

    properties = [
        Property("p1", DataType.string),
        Property("p2", DataType.binary),
        Property("p2", DataType.boolean),
    ]
    with pytest.raises(MlflowException, match=r"Found duplicated property names: `p2`"):
        Object(properties)


@pytest.mark.parametrize("data_type", DataType)
def test_property_to_and_from_dict(data_type):
    prop = Property("data", data_type, True)
    assert prop.to_dict() == {"data": {"type": data_type.name, "required": True}}
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop

    # test array
    prop = Property("arr", Array(data_type), False)
    assert prop.to_dict() == {
        "arr": {
            "type": "array",
            "items": {"type": data_type.name},
            "required": False,
        },
    }
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop

    # test object
    prop = Property("data", Object([Property("p", data_type)]))
    assert prop.to_dict() == {
        "data": {
            "type": "object",
            "properties": {"p": {"type": data_type.name, "required": True}},
            "required": True,
        },
    }
    assert Property.from_json_dict(**json.loads(json.dumps(prop.to_dict()))) == prop


def test_property_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Expected Property JSON to contain a single key as name, got 2 keys.",
    ):
        Property.from_json_dict(**{"p1": {}, "p2": {}})

    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Property `p`. Expected to find key `type`",
    ):
        Property.from_json_dict(**{"p": {}})

    with pytest.raises(
        MlflowException,
        match=(
            r"Unsupported type 'invalid_type', expected instance of DataType, Array, "
            r"Object, Map or "
        ),
    ):
        Property.from_json_dict(**{"p": {"type": "invalid_type"}})

    # test array
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Array JSON. Expected to find keys `items` and `type`",
    ):
        Property.from_json_dict(**{"p": {"type": "array"}})

    with pytest.raises(
        MlflowException,
        match=(
            r"Unsupported type 'invalid_type', expected instance of DataType, Array, "
            r"Object, Map or "
        ),
    ):
        Property.from_json_dict(**{"p": {"type": "array", "items": {"type": "invalid_type"}}})

    with pytest.raises(
        MlflowException,
        match=r"Expected items to be a dictionary of Object JSON",
    ):
        Property.from_json_dict(**{"p": {"type": "array", "items": "invalid_items_type"}})

    # test object
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Object JSON. Expected to find keys `properties` and `type`",
    ):
        Property.from_json_dict(**{"p": {"type": "object"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Property.from_json_dict(**{"p": {"type": "object", "properties": "invalid_type"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Property.from_json_dict(
            **{
                "p": {
                    "type": "object",
                    "properties": {"p1": {"type": "string"}, "p2": "invalid_type"},
                }
            }
        )


def test_object_to_and_from_dict():
    properties = []
    dict_prop = {}
    for data_type in DataType:
        properties.append(Property(f"name_{data_type.name}", data_type))
        dict_prop[f"name_{data_type.name}"] = {"type": data_type.name, "required": True}
    obj = Object(properties)
    assert obj.to_dict() == {
        "type": "object",
        "properties": dict(sorted(dict_prop.items())),
    }
    assert Object.from_json_dict(**json.loads(json.dumps(obj.to_dict()))) == obj


def test_object_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Object JSON. Expected to find keys `properties` and `type`",
    ):
        Object.from_json_dict(**{"type": "object"})

    with pytest.raises(
        MlflowException, match=r"Type mismatch, Object expects `object` as the type"
    ):
        Object.from_json_dict(**{"type": "array", "properties": {}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Object.from_json_dict(**{"type": "object", "properties": "invalid_type"})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Object.from_json_dict(
            **{
                "type": "object",
                "properties": {"p1": {"type": "string"}, "p2": "invalid_type"},
            }
        )


@pytest.mark.parametrize("data_type", DataType)
def test_array_to_and_from_dict(data_type):
    arr = Array(data_type)
    assert arr.to_dict() == {"type": "array", "items": {"type": data_type.name}}
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr

    # test object
    arr = Array(Object([Property("p", data_type)]))
    assert arr.to_dict() == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"p": {"type": data_type.name, "required": True}},
        },
    }
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr


def test_array_from_dict_with_errors():
    with pytest.raises(
        MlflowException,
        match=r"Missing keys in Array JSON. Expected to find keys `items` and `type`",
    ):
        Array.from_json_dict(**{"type": "array"})

    with pytest.raises(MlflowException, match=r"Type mismatch, Array expects `array` as the type"):
        Array.from_json_dict(**{"type": "object", "items": "string"})

    with pytest.raises(MlflowException, match=r"Expected items to be a dictionary of Object JSON"):
        Array.from_json_dict(**{"type": "array", "items": "string"})

    with pytest.raises(
        MlflowException,
        match=(
            r"Unsupported type 'invalid_type', expected instance of DataType, Array, "
            r"Object, Map or "
        ),
    ):
        Array.from_json_dict(**{"type": "array", "items": {"type": "invalid_type"}})

    with pytest.raises(
        MlflowException, match=r"Expected properties to be a dictionary of Property JSON"
    ):
        Array.from_json_dict(**{"type": "array", "items": {"type": "object", "properties": []}})


def test_nested_array_object_to_and_from_dict():
    data = [{"p": "a", "arr": [{"p2": True, "arr2": [1, 2, 3]}, {"arr2": [2, 3, 4]}]}]
    col_spec = _infer_colspec_type(data)
    arr = Array(
        Object(
            [
                Property("p", DataType.string),
                Property(
                    "arr",
                    Array(
                        Object(
                            [
                                Property("p2", DataType.boolean, required=False),
                                Property("arr2", Array(DataType.long)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )
    assert col_spec == arr
    assert arr.to_dict() == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "p": {"type": "string", "required": True},
                "arr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "p2": {"type": "boolean", "required": False},
                            "arr2": {
                                "type": "array",
                                "items": {"type": "long"},
                                "required": True,
                            },
                        },
                    },
                    "required": True,
                },
            },
        },
    }
    assert Array.from_json_dict(**json.loads(json.dumps(arr.to_dict()))) == arr


def test_merge_object_example():
    obj1 = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double),
        ]
    )
    obj1_dict = obj1.to_dict()
    obj2 = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="c", dtype=DataType.boolean),
        ]
    )
    obj2_dict = obj2.to_dict()
    updated_obj = obj1._merge(obj2)
    assert updated_obj == Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double, required=False),
            Property(name="c", dtype=DataType.boolean, required=False),
        ]
    )
    assert obj1.to_dict() == obj1_dict
    assert obj2.to_dict() == obj2_dict


def test_merge_property_example():
    prop1 = Property(
        name="a",
        dtype=Object(properties=[Property(name="a", dtype=DataType.string, required=False)]),
    )
    prop1_dict = prop1.to_dict()
    prop2 = Property(
        name="a",
        dtype=Object(
            properties=[
                Property(name="a", dtype=DataType.string),
                Property(name="b", dtype=DataType.double),
            ]
        ),
    )
    prop2_dict = prop2.to_dict()
    updated_prop = prop1._merge(prop2)
    assert updated_prop == Property(
        name="a",
        dtype=Object(
            properties=[
                Property(name="a", dtype=DataType.string, required=False),
                Property(name="b", dtype=DataType.double, required=False),
            ]
        ),
    )
    assert prop1.to_dict() == prop1_dict
    assert prop2.to_dict() == prop2_dict


def test_infer_colspec_type():
    data = {"role": "system", "content": "Translate every message you receive to French."}
    dtype = _infer_colspec_type(data)
    assert dtype == Object(
        [Property("role", DataType.string), Property("content", DataType.string)]
    )

    data = {
        "messages": [
            {"role": "system", "content": "Translate every message you receive to French."},
            {"role": "user", "content": "I like machine learning", "name": "John Doe"},
        ],
    }
    dtype = _infer_colspec_type(data)
    assert dtype == Object(
        [
            Property(
                "messages",
                Array(
                    Object(
                        [
                            Property("role", DataType.string),
                            Property("content", DataType.string),
                            Property("name", DataType.string, required=False),
                        ]
                    )
                ),
            )
        ]
    )

    data = [
        {
            "messages": [
                {"role": "system", "content": "Translate every message you receive to French."},
                {"role": "user", "content": "I like machine learning"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "Summarize the following document ...", "name": "jeff"},
            ]
        },
    ]
    dtype = _infer_colspec_type(data)
    assert dtype == Array(
        Object(
            [
                Property(
                    "messages",
                    Array(
                        Object(
                            [
                                Property("role", DataType.string),
                                Property("content", DataType.string),
                                Property("name", DataType.string, required=False),
                            ]
                        )
                    ),
                )
            ]
        )
    )


def test_infer_schema_on_objects_and_arrays_to_and_from_dict():
    data = [
        {
            "messages": [
                {"role": "system", "content": "Translate every message you receive to French."},
                {"role": "user", "content": "I like machine learning"},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "Summarize the following document ...", "name": "jeff"},
            ]
        },
    ]
    schema = _infer_schema(data)
    assert schema == Schema(
        [
            ColSpec(
                Array(
                    Object(
                        [
                            Property("role", DataType.string),
                            Property("content", DataType.string),
                            Property("name", DataType.string, required=False),
                        ]
                    )
                ),
                name="messages",
            )
        ]
    )
    assert schema.to_dict() == [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "required": True},
                    "name": {"type": "string", "required": False},
                    "role": {"type": "string", "required": True},
                },
            },
            "name": "messages",
            "required": True,
        }
    ]
    assert Schema.from_json(json.dumps(schema.to_dict())) == schema

    data = [
        {
            "messages": [
                {"role": "system", "content": "Translate every message you receive to French."},
                {"role": "user", "content": "I like machine learning"},
            ],
            "another_message": "hello world",
        },
        {
            "messages": [
                {"role": "user", "content": "Summarize the following document ...", "name": "jeff"},
            ]
        },
    ]
    schema = _infer_schema(data)
    assert schema == Schema(
        [
            ColSpec(
                Array(
                    Object(
                        [
                            Property("role", DataType.string),
                            Property("content", DataType.string),
                            Property("name", DataType.string, required=False),
                        ]
                    )
                ),
                name="messages",
            ),
            ColSpec(DataType.string, name="another_message", required=False),
        ]
    )
    assert schema.to_dict() == [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "required": True},
                    "name": {"type": "string", "required": False},
                    "role": {"type": "string", "required": True},
                },
            },
            "name": "messages",
            "required": True,
        },
        {
            "type": "string",
            "name": "another_message",
            "required": False,
        },
    ]
    assert Schema.from_json(json.dumps(schema.to_dict())) == schema


@pytest.mark.parametrize(
    ("data", "data_type"),
    [
        ("string", DataType.string),
        (np.int32(1), DataType.integer),
        (True, DataType.boolean),
        (1.0, DataType.double),
        (np.float32(0.1), DataType.float),
        (np.int64(100), DataType.long),
        (np.datetime64("2023-10-13 00:00:00"), DataType.datetime),
    ],
)
def test_schema_inference_on_datatypes(data, data_type):
    assert _infer_schema(data) == Schema([ColSpec(data_type)])


@pytest.mark.parametrize(
    ("data", "data_type"),
    [
        (["a", "b", "c"], DataType.string),
        ([np.int32(1), np.int32(2)], DataType.integer),
        ([True, False], DataType.boolean),
        ([1.0, 2.0], DataType.double),
        ([np.float32(0.1), np.float32(0.2)], DataType.float),
        ([np.int64(100), np.int64(200)], DataType.long),
        (
            [np.datetime64("2023-10-13 00:00:00"), np.datetime64("2023-10-14 00:00:00")],
            DataType.datetime,
        ),
    ],
)
def test_schema_inference_on_dictionaries(data, data_type):
    test_data = {"data": data}
    assert _infer_schema(test_data) == Schema([ColSpec(Array(data_type), name="data")])

    test_data = {"data": {"dict": data}}
    inferred_schema = _infer_schema(test_data)
    expected_schema = Schema([ColSpec(Object([Property("dict", Array(data_type))]), name="data")])
    assert inferred_schema == expected_schema


@pytest.mark.parametrize(
    ("data", "data_type"),
    [
        (["a", "b", "c"], DataType.string),
        ([np.int32(1), np.int32(2)], DataType.integer),
        ([True, False], DataType.boolean),
        ([1.0, 2.0], DataType.double),
        ([np.float32(0.1), np.float32(0.2)], DataType.float),
        ([np.int64(100), np.int64(200)], DataType.long),
        (
            [np.datetime64("2023-10-13 00:00:00"), np.datetime64("2023-10-14 00:00:00")],
            DataType.datetime,
        ),
    ],
)
def test_schema_inference_on_lists(data, data_type):
    assert _infer_schema(data) == Schema([ColSpec(data_type)])

    test_data = [{"data": data}]
    assert _infer_schema(test_data) == Schema([ColSpec(Array(data_type), name="data")])

    test_data = [{"data": [data]}]
    assert _infer_schema(test_data) == Schema([ColSpec(Array(Array(data_type)), name="data")])

    test_data = [{"data": {"dict": data}}, {"data": {"dict": data, "string": "a"}}]
    inferred_schema = _infer_schema(test_data)
    expected_schema = Schema(
        [
            ColSpec(
                Object(
                    [
                        Property("dict", Array(data_type)),
                        Property("string", DataType.string, required=False),
                    ]
                ),
                name="data",
            )
        ]
    )
    assert inferred_schema == expected_schema

    test_data = [{"data": {"dict": data}}, {"data": {"dict": data}, "string": "a"}]
    inferred_schema = _infer_schema(test_data)
    expected_schema = Schema(
        [
            ColSpec(Object([Property("dict", Array(data_type))]), name="data"),
            ColSpec(DataType.string, name="string", required=False),
        ]
    )
    assert inferred_schema == expected_schema


def test_schema_inference_with_empty_lists():
    # If (non-nested) empty list is passed, should be inferred as string for backward compatibility
    # with version 2.7.1: https://github.com/mlflow/mlflow/blob/v2.7.1/mlflow/types/utils.py#L150
    data = []
    assert _infer_schema(data) == Schema([ColSpec(DataType.string)])

    data = [None, "a"]
    assert _infer_schema(data) == Schema([ColSpec(DataType.string)])

    data = [["a", "b", "c"], ["a", "b", "c", "d"], None]
    assert _infer_schema(data) == Schema([ColSpec(Array(DataType.string))])

    data = [[None, np.nan]]
    assert _infer_schema(data) == Schema([ColSpec(Array(AnyType()))])

    assert _infer_schema([[]]) == Schema([ColSpec(AnyType())])

    # Empty numpy array is not allowed.
    with pytest.raises(
        MlflowException,
        match=r"Numpy array must include at least one non-empty item",
    ):
        _infer_schema([np.array([])])

    # If at least one of sublists is not empty, we can assume other empty lists have the same type.
    data = [
        {
            "data": [
                ["a", "b", "c"],
                [],
                ["d", "e"],
                [],
            ]
        }
    ]
    inferred_schema = _infer_schema(data)
    expected_schema = Schema([ColSpec(Array(Array(DataType.string)), name="data")])
    assert inferred_schema == expected_schema

    # Complex case for deeply nested array
    data = [
        {
            "data": [
                [["a", "b", "c"], [], ["d"]],
                [[], ["e"]],
                [
                    [],
                    [],
                    [],
                ],
            ]
        }
    ]
    inferred_schema = _infer_schema(data)
    expected_schema = Schema([ColSpec(Array(Array(Array(DataType.string))), name="data")])
    assert inferred_schema == expected_schema

    data = [{"data": {"key": []}}]
    assert _infer_schema(data) == Schema(
        [ColSpec(Object([Property("key", AnyType(), required=False)]), name="data")]
    )


def test_repr_of_objects():
    obj = Object(
        properties=[
            Property(name="a", dtype=DataType.string),
            Property(name="b", dtype=DataType.double, required=False),
            Property(name="c", dtype=Array(DataType.long)),
            Property(name="d", dtype=Object([Property("d1", DataType.string)])),
        ]
    )
    obj_repr = (
        "{a: string (required), b: double (optional), c: Array(long) "
        "(required), d: {d1: string (required)} (required)}"
    )
    assert repr(obj) == obj_repr

    arr = Array(obj)
    assert repr(arr) == f"Array({obj_repr})"


def test_convert_dataclass_to_schema_for_rag():
    schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionRequest())
    schema_dict = schema.to_dict()
    assert schema_dict == [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "required": True},
                    "role": {"type": "string", "required": True},
                },
            },
            "name": "messages",
            "required": True,
        }
    ]

    schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionResponse())
    schema_dict = schema.to_dict()
    assert schema_dict == [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finish_reason": {"type": "string", "required": True},
                    "index": {"type": "long", "required": True},
                    "message": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "required": True},
                            "role": {"type": "string", "required": True},
                        },
                        "required": True,
                    },
                },
            },
            "name": "choices",
            "required": True,
        },
        {
            "name": "object",
            "required": True,
            "type": "string",
        },
    ]


def test_convert_dataclass_to_schema_complex():
    @dataclass
    class Settings:
        baz: Optional[bool] = True  # noqa: UP045
        qux: bool | None = None

    @dataclass
    class Config:
        bar: int = 2
        config_settings: Settings = field(default_factory=Settings)

    @dataclass
    class MainTask:
        foo: str = "1"
        process_configs: list[Config] = field(default_factory=lambda: [Config()])

    schema = convert_dataclass_to_schema(MainTask)
    schema_dict = schema.to_dict()
    assert schema_dict == [
        {"type": "string", "name": "foo", "required": True},
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bar": {"type": "long", "required": True},
                    "config_settings": {
                        "type": "object",
                        "properties": {
                            "baz": {"type": "boolean", "required": False},
                            "qux": {"type": "boolean", "required": False},
                        },
                        "required": True,
                    },
                },
            },
            "name": "process_configs",
            "required": True,
        },
    ]


def test_convert_dataclass_to_schema_invalid():
    # Invalid dataclass with Union
    @dataclass
    class InvalidDataclassWithUnion:
        foo: str | int = "1"

    with pytest.raises(
        MlflowException,
        match=re.escape(r"Only Optional[...] is supported as a Union type in dataclass fields"),
    ):
        convert_dataclass_to_schema(InvalidDataclassWithUnion)

    # Invalid dataclass with Dict
    @dataclass
    class InvalidDataclassWithDict:
        foo: dict[str, int] = field(default_factory=dict)

    with pytest.raises(
        MlflowException,
        match=re.escape(r"Unsupported field type dict[str, int] in dataclass InvalidDataclass"),
    ):
        convert_dataclass_to_schema(InvalidDataclassWithDict)


@pytest.mark.parametrize(
    ("data", "expected_schema"),
    [
        # Pure empty list is inferred as string type for backwards compatibility
        ([], Schema([ColSpec(type=DataType.string)])),
        # None type is inferred as AnyType
        (None, Schema([ColSpec(AnyType())])),
        ({"a": None}, Schema([ColSpec(type=AnyType(), name="a", required=False)])),
        # Empty list in dictionary is inferred as AnyType
        ({"a": []}, Schema([ColSpec(type=AnyType(), name="a")])),
        (
            [{"a": []}],
            Schema([ColSpec(type=AnyType(), name="a")]),
        ),
        (
            [{"a": []}, {"a": None}],
            Schema([ColSpec(type=AnyType(), name="a", required=False)]),
        ),
        ({"a": [None]}, Schema([ColSpec(type=Array(AnyType()), name="a", required=False)])),
        ({"a": [[], None]}, Schema([ColSpec(type=Array(AnyType()), name="a", required=False)])),
        (
            {"a": [None, "string"]},
            Schema([ColSpec(type=Array(DataType.string), name="a", required=False)]),
        ),
        (
            {"a": {"x": None}},
            Schema([ColSpec(type=Object([Property("x", AnyType(), required=False)]), name="a")]),
        ),
        (
            {"a": {"x": pd.NA}},
            Schema([ColSpec(type=Object([Property("x", AnyType(), required=False)]), name="a")]),
        ),
        (
            pd.DataFrame({"a": [True, None]}),
            Schema([ColSpec(type=DataType.boolean, name="a", required=False)]),
        ),
        (
            pd.DataFrame({"a": [True, None]}).astype("boolean"),
            Schema([ColSpec(type=DataType.boolean, name="a", required=False)]),
        ),
        (
            [
                {
                    "id": None,
                    "object": "chat.completion",
                    "created": 1731491873,
                    "model": None,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "MLflow",
                            },
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    },
                }
            ],
            Schema(
                [
                    ColSpec(type=AnyType(), name="id", required=False),
                    ColSpec(type=DataType.string, name="object"),
                    ColSpec(type=DataType.long, name="created"),
                    ColSpec(type=AnyType(), name="model", required=False),
                    ColSpec(
                        type=Array(
                            Object(
                                properties=[
                                    Property("index", dtype=DataType.long),
                                    Property(
                                        "message",
                                        dtype=Object(
                                            [
                                                Property("role", DataType.string),
                                                Property("content", DataType.string),
                                            ]
                                        ),
                                    ),
                                    Property("finish_reason", dtype=AnyType(), required=False),
                                ]
                            )
                        ),
                        name="choices",
                    ),
                    ColSpec(
                        type=Object(
                            [
                                Property("prompt_tokens", AnyType(), required=False),
                                Property("completion_tokens", AnyType(), required=False),
                                Property("total_tokens", AnyType(), required=False),
                            ]
                        ),
                        name="usage",
                    ),
                ]
            ),
        ),
        (
            [
                {
                    "messages": [
                        {
                            "content": "You are a helpful assistant.",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "system",
                            "name": None,
                            "id": None,
                        },
                        {
                            "content": "What would you like to ask?",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "ai",
                            "name": None,
                            "id": None,
                            "example": False,
                            "tool_calls": [],
                            "invalid_tool_calls": [],
                            "usage_metadata": None,
                        },
                        {
                            "content": "Who owns MLflow?",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "type": "human",
                            "name": None,
                            "id": None,
                            "example": False,
                        },
                    ],
                    "text": "Hello?",
                }
            ],
            Schema(
                [
                    ColSpec(
                        Array(
                            Object(
                                properties=[
                                    Property("content", DataType.string),
                                    Property("additional_kwargs", AnyType(), required=False),
                                    Property("response_metadata", AnyType(), required=False),
                                    Property("type", DataType.string),
                                    Property("name", AnyType(), required=False),
                                    Property("id", AnyType(), required=False),
                                    Property("example", DataType.boolean, required=False),
                                    Property("tool_calls", AnyType(), required=False),
                                    Property("invalid_tool_calls", AnyType(), required=False),
                                    Property("usage_metadata", AnyType(), required=False),
                                ]
                            )
                        ),
                        name="messages",
                    ),
                    ColSpec(DataType.string, name="text"),
                ]
            ),
        ),
    ],
)
def test_infer_schema_with_anytype(data, expected_schema):
    inferred_schema = _infer_schema(data)
    assert inferred_schema == expected_schema

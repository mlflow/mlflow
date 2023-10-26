import datetime
import json
import math
import re

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.types as T
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _enforce_tensor_spec
from mlflow.pyfunc import _parse_spark_datatype
from mlflow.types import DataType
from mlflow.types.schema import (
    Array,
    ColSpec,
    Object,
    ParamSchema,
    ParamSpec,
    Property,
    Schema,
    TensorSpec,
)
from mlflow.types.utils import (
    _get_tensor_shape,
    _infer_colspec_type,
    _infer_param_schema,
    _infer_schema,
    _validate_input_dictionary_contains_only_strings_and_lists_of_strings,
)


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
def test_schema_inference_on_scalar(data, data_type):
    assert _infer_schema(data) == Schema([ColSpec(data_type)])


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


def test_schema_inference_validating_dictionary_keys():
    for data in [{1.7: "a", "b": "c"}, {12.4: "c", "d": "e"}]:
        with pytest.raises(MlflowException, match="The dictionary keys are not all strings."):
            _infer_schema(data)


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
def test_schema_inference_on_dictionary_of_objects(data, data_type):
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


def test_spark_schema_inference_complex_lists_and_objects(spark):
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


@pytest.fixture
def dict_of_ndarrays():
    return {
        "1D": np.arange(0, 12, 0.5),
        "2D": np.arange(0, 12, 0.5).reshape(3, 8),
        "3D": np.arange(0, 12, 0.5).reshape(2, 3, 4),
        "4D": np.arange(0, 12, 0.5).reshape(3, 2, 2, 2),
    }


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


def test_schema_inference_on_dictionary_of_ndarrays(dict_of_ndarrays):
    schema = _infer_schema(dict_of_ndarrays)
    assert ["float64"] * 4 == schema.numpy_types()
    with pytest.raises(MlflowException, match="TensorSpec only supports numpy types"):
        schema.pandas_types()
    with pytest.raises(MlflowException, match="TensorSpec cannot be converted to spark dataframe"):
        schema.as_spark_schema()

    assert schema == Schema(
        [
            TensorSpec(tensor.dtype, _get_tensor_shape(tensor), name)
            for name, tensor in dict_of_ndarrays.items()
        ]
    )


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


def test_schema_inference_on_pandas_dataframe(pandas_df_with_all_types):
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


def test_schema_inference_on_basic_numpy(pandas_df_with_all_types):
    for col in pandas_df_with_all_types:
        data = pandas_df_with_all_types[col].to_numpy()
        schema = _infer_schema(data)
        assert schema == Schema([TensorSpec(type=data.dtype, shape=(-1,))])


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


@pytest.fixture(scope="module")
def spark():
    with pyspark.sql.SparkSession.builder.getOrCreate() as s:
        yield s


def test_schema_inference_on_spark_dataframe(pandas_df_with_all_types):
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
                    T.StructField("datetime", _parse_spark_datatype("timestamp"), True)
                )
            else:
                struct_fields.append(T.StructField(t.name, _parse_spark_datatype(t.name), True))
        spark_schema = T.StructType(struct_fields)
        sparkdf = spark.createDataFrame(pandas_df_with_all_types, schema=spark_schema)
        schema = _infer_schema(sparkdf)
        assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])


def test_spark_type_mapping(pandas_df_with_all_types):
    assert isinstance(DataType.boolean.to_spark(), T.BooleanType)
    assert isinstance(DataType.integer.to_spark(), T.IntegerType)
    assert isinstance(DataType.long.to_spark(), T.LongType)
    assert isinstance(DataType.float.to_spark(), T.FloatType)
    assert isinstance(DataType.double.to_spark(), T.DoubleType)
    assert isinstance(DataType.string.to_spark(), T.StringType)
    assert isinstance(DataType.binary.to_spark(), T.BinaryType)
    assert isinstance(DataType.datetime.to_spark(), T.TimestampType)
    pandas_df_with_all_types = pandas_df_with_all_types.drop(
        columns=["boolean_ext", "integer_ext", "string_ext"]
    )
    schema = _infer_schema(pandas_df_with_all_types)
    expected_spark_schema = T.StructType(
        [T.StructField(t.name, t.to_spark(), False) for t in schema.input_types()]
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
    expected_spark_schema = T.StructType(
        [T.StructField(str(i), t.to_spark(), False) for i, t in enumerate(schema.input_types())]
    )
    actual_spark_schema = schema.as_spark_schema()
    assert expected_spark_schema.jsonValue() == actual_spark_schema.jsonValue()

    # test single unnamed column is mapped to just a single spark type
    schema = Schema([ColSpec(DataType.integer)])
    spark_type = schema.as_spark_schema()
    assert isinstance(spark_type, T.IntegerType)


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

    # Raise error for invalid parameters types - tuple, 2D array, dictionary
    test_parameters = {
        "a": "str_a",
        "b": (1, 2, 3),
        "c": True,
        "d": [[1, 2], [3, 4]],
        "e": {"a": 1, "b": 2},
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
            "('e', {'a': 1, 'b': 2}, MlflowException('Expected parameters "
            "to be 1D array or scalar, got dict'))"
        )
    )


def test_infer_param_schema_with_errors():
    with pytest.raises(
        MlflowException, match=r"Expected parameters to be 1D array or scalar, got Series"
    ):
        _infer_param_schema({"a": pd.Series([1, 2, 3])})


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

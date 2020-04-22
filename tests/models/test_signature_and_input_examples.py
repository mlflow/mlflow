import json
import math
import numpy as np
import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.models.utils import save_example, TensorsNotSupportedException
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema
from mlflow.types.utils import infer_schema

from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dataframe_from_json


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


def test_model_signature():
    signature1 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.double),
                        ColSpec(name=None, type=DataType.double)]))
    signature2 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.double),
                        ColSpec(name=None, type=DataType.double)]))
    assert signature1 == signature2
    signature3 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.float),
                        ColSpec(name=None, type=DataType.double)]))
    assert signature3 != signature1
    as_json = json.dumps(signature1.to_dict())
    signature4 = ModelSignature.from_dict(json.loads(as_json))
    assert signature1 == signature4
    signature5 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=None)
    as_json = json.dumps(signature5.to_dict())
    signature6 = ModelSignature.from_dict(json.loads(as_json))
    assert signature5 == signature6


@pytest.fixture
def pandas_df_with_all_types():
    return pd.DataFrame({
        "boolean": [True, False, True],
        "integer": np.array([1, 2, 3], np.int32),
        "long": np.array([1, 2, 3], np.int64),
        "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
        "double": [math.pi, 2 * math.pi, 3 * math.pi],
        "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9])],
        "string": ["a", "b", 'c'],
    })


def test_signature_inference_infers_input_and_output_as_expected():
    sig0 = infer_signature(np.array([1]))
    assert sig0.inputs is not None
    assert sig0.outputs is None
    sig1 = infer_signature(np.array([1]), np.array([1]))
    assert sig1.inputs == sig0.inputs
    assert sig1.outputs == sig0.inputs


def test_schema_inference_on_dataframe(pandas_df_with_all_types):
    sig = infer_signature(pandas_df_with_all_types)
    assert sig.inputs == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])


def test_schema_inference_on_dictionary(pandas_df_with_all_types):
    # test dictionary
    d = {c: pandas_df_with_all_types[c].values for c in pandas_df_with_all_types.columns}
    sig = infer_signature(d)
    assert dict(zip(sig.inputs.column_names(), sig.inputs.column_types())) == \
        {c: DataType[c] for c in pandas_df_with_all_types.columns}
    # test exception is raised if non-numpy data in dictionary
    with pytest.raises(TypeError):
        infer_signature({"x": 1})
    with pytest.raises(TypeError):
        infer_signature({"x": [1]})


def test_schema_inference_on_numpy_array(pandas_df_with_all_types):
    # drop int and float as we lose type size information when storing as objects and defaults are
    # 64b.
    pandas_df_with_all_types = pandas_df_with_all_types.drop(columns=["integer", "float"])
    schema = infer_schema(pandas_df_with_all_types.values)
    assert schema == Schema([ColSpec(x) for x in pandas_df_with_all_types.columns])

    # test objects
    schema = infer_schema(np.array(["a"], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.string)])
    schema = infer_schema(np.array([bytes([1])], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = infer_schema(np.array([bytearray([1]), None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.binary)])
    schema = infer_schema(np.array([True, None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.boolean)])
    schema = infer_schema(np.array([1.1, None], dtype=np.object))
    assert schema == Schema([ColSpec(DataType.double)])

    # test bytes
    schema = infer_schema(np.array([bytes([1])], dtype=np.bytes_))

    assert schema == Schema([ColSpec(DataType.binary)])
    schema = infer_schema(np.array([bytearray([1])], dtype=np.bytes_))
    assert schema == Schema([ColSpec(DataType.binary)])

    # test string
    schema = infer_schema(np.array(["a"], dtype=np.str))
    assert schema == Schema([ColSpec(DataType.string)])

    # test boolean
    schema = infer_schema(np.array([True], dtype=np.bool))
    assert schema == Schema([ColSpec(DataType.boolean)])

    # test ints
    for t in [np.uint8, np.uint16, np.int8, np.int16, np.int32]:
        schema = infer_schema(np.array([1, 2, 3], dtype=t))
        assert schema == Schema([ColSpec("integer")])

    # test longs
    for t in [np.uint32, np.int64]:
        schema = infer_schema(np.array([1, 2, 3], dtype=t))
        assert schema == Schema([ColSpec("long")])

    # unsigned long is unsupported
    with pytest.raises(MlflowException):
        infer_signature(np.array([1, 2, 3], dtype=np.uint64))

    # test floats
    for t in [np.float16, np.float32]:
        schema = infer_schema(np.array([1.1, 2.2, 3.3], dtype=t))
        assert schema == Schema([ColSpec("float")])

    # test doubles
    schema = infer_schema(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    assert schema == Schema([ColSpec("double")])

    # unsupported
    with pytest.raises(MlflowException):
        infer_signature(np.array([1, 2, 3], dtype=np.float128))


def test_that_schema_inference_with_tensors_raises_exception():
    with pytest.raises(MlflowException):
        infer_signature(np.array([[[1, 2, 3]]], dtype=np.int64))
    with pytest.raises(MlflowException):
        infer_signature(pd.DataFrame({"x": [np.array([[1, 2, 3]], dtype=np.int64)]}))
    with pytest.raises(MlflowException):
        infer_signature({"x": np.array([[1, 2, 3]], dtype=np.int64)})


def test_spark_schema_inference(pandas_df_with_all_types):
    try:
        import pyspark
        from pyspark.sql.types import _parse_datatype_string, StructField, StructType
        schema = infer_schema(pandas_df_with_all_types)
        assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
        spark_session = pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate())
        spark_schema = StructType(
            [StructField(t.name, _parse_datatype_string(t.name), True)
             for t in schema.column_types()])
        sparkdf = spark_session.createDataFrame(pandas_df_with_all_types, schema=spark_schema)
        schema = infer_schema(sparkdf)
        assert schema == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
    except ImportError:
        pass


def test_input_examples(pandas_df_with_all_types):
    sig = infer_signature(pandas_df_with_all_types)
    # test setting example with data frame with all supported data types
    with TempDir() as tmp:
        filename = save_example(tmp.path(), pandas_df_with_all_types)
        with open(tmp.path(filename), "r") as f:
            data = json.load(f)
            assert set(data.keys()) == set(("columns", "data"))
        parsed_df = dataframe_from_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()
        # the frame read without schema should match except for the binary values
        assert (parsed_df.drop(columns=["binary"]) == dataframe_from_json(tmp.path(filename))
                .drop(columns=["binary"])).all().all()

    # pass the input as dictionary instead
    with TempDir() as tmp:
        d = {name: pandas_df_with_all_types[name].values
             for name in pandas_df_with_all_types.columns}
        filename = save_example(tmp.path(), d)
        parsed_df = dataframe_from_json(tmp.path(filename), sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()

    # input passed as numpy array
    sig = infer_signature(pandas_df_with_all_types.values)
    with TempDir() as tmp:
        filename = save_example(tmp.path(), pandas_df_with_all_types.values)
        with open(tmp.path(filename), "r") as f:
            data = json.load(f)
            assert set(data.keys()) == set(("data",))
        parsed_ary = dataframe_from_json(tmp.path(filename), schema=sig.inputs).values
        assert (pandas_df_with_all_types.values == parsed_ary).all().all()

    # pass multidimensional array
    with TempDir() as tmp:
        example = np.array([[[1, 2, 3]]])
        with pytest.raises(TensorsNotSupportedException):
            filename = save_example(tmp.path(), example)

    # pass multidimensional array
    with TempDir() as tmp:
        example = np.array([[1, 2, 3]])
        with pytest.raises(TensorsNotSupportedException):
            filename = save_example(tmp.path(), {"x": example, "y": example})

    # pass dict with scalars
    with TempDir() as tmp:
        example = {"a": 1, "b": "abc"}
        filename = save_example(tmp.path(), example)
        parsed_df = dataframe_from_json(tmp.path(filename))
        assert example == parsed_df.to_dict(orient="records")[0]

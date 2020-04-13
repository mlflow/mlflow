import json
import math
import numpy as np
import pandas as pd
import pytest
import pyspark

from mlflow.exceptions import MlflowException
from mlflow.models.signature import ColSpec, DataType, ModelSignature, ModelInputExample, \
    infer_signature, save_example, from_json, Schema
from mlflow.utils.file_utils import TempDir


def test_col_spec():
    a1 = ColSpec("a", "string")
    a2 = ColSpec("a", DataType.string)
    a3 = ColSpec("a", DataType.integer)
    assert a1 != a3
    b1 = ColSpec("b", DataType.string)
    assert b1 != a1
    assert a1 == a2
    with pytest.raises(MlflowException) as ex:
        ColSpec("a", "unsupported")
    assert "Unsupported type 'unsupported'" in ex.value.message
    a4 = ColSpec(**a1.to_dict())
    assert a4 == a1
    assert ColSpec(**json.loads(json.dumps(a1.to_dict()))) == a1
    a5 = ColSpec(None, "string")
    a6 = ColSpec(None, "string")
    assert a5 == a6
    assert ColSpec(**json.loads(json.dumps(a5.to_dict()))) == a5


def test_model_signature():
    signature1 = ModelSignature(
        inputs=Schema([ColSpec("a", DataType.boolean), ColSpec("b", DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.double),
                        ColSpec(name=None, type=DataType.double)]))
    signature2 = ModelSignature(
        inputs=Schema([ColSpec("a", DataType.boolean), ColSpec("b", DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.double),
                        ColSpec(name=None, type=DataType.double)]))
    assert signature1 == signature2
    signature3 = ModelSignature(
        inputs=Schema([ColSpec("a", DataType.boolean), ColSpec("b", DataType.binary)]),
        outputs=Schema([ColSpec(name=None, type=DataType.float),
                        ColSpec(name=None, type=DataType.double)]))
    assert signature3 != signature1
    as_json = json.dumps(signature1.to_dict())
    signature4 = ModelSignature.from_dict(json.loads(as_json))
    assert signature1 == signature4
    signature5 = ModelSignature(
        inputs=Schema([ColSpec("a", DataType.boolean), ColSpec("b", DataType.binary)]),
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
        "string": ["a", "b", 'c'],
        "binary": [bytearray([1, 2, 3]), bytearray([4, 5, 6]), bytearray([7, 8, 9])],
    })


def test_schema_inference(pandas_df_with_all_types):
    sig = infer_signature(pandas_df_with_all_types)
    assert sig.inputs == Schema([ColSpec(x, x) for x in pandas_df_with_all_types.columns])
    assert sig.outputs is None
    df = pandas_df_with_all_types.drop(columns=["integer", "float"])
    spark_session = pyspark.sql.SparkSession(pyspark.SparkContext.getOrCreate())
    sparkdf = spark_session.createDataFrame(df)
    sig = infer_signature(sparkdf)
    assert sig.inputs == Schema([ColSpec(x, x) for x in df.columns])
    sig = infer_signature(df.values)
    assert sig.inputs == Schema([ColSpec(None, x) for x in df.columns])
    sig = infer_signature(df, np.array([1, 2, 3], dtype=np.int32))
    assert sig.outputs == Schema([ColSpec(None, "integer")])
    # test dictionary
    sig = infer_signature({"x": np.array([1, 2, 3], dtype=np.int32), "y": np.array([4, 5, 6],
                                                                                   dtype=np.int32)})
    assert sig.inputs == Schema([ColSpec("x", "integer"), ColSpec("y", "integer")])

    for t in [np.uint8, np.uint16, np.int8, np.int16, np.int32]:
        sig = infer_signature(np.array([1, 2, 3], dtype=t))
        assert sig.inputs == Schema([ColSpec(None, "integer")])
    for t in [np.uint32, np.int64]:
        sig = infer_signature(np.array([1, 2, 3], dtype=t))
        assert sig.inputs == Schema([ColSpec(None, "long")])
    # test negative cases
    # unsupported data type
    with pytest.raises(MlflowException):
        infer_signature(np.array([1, 2, 3], dtype=np.uint64))
    # tensors
    with pytest.raises(MlflowException):
        infer_signature(np.array([[[1, 2, 3]]], dtype=np.int64))
    with pytest.raises(MlflowException):
        infer_signature(pd.DataFrame({"x": [np.array([[1, 2, 3]], dtype=np.int64)]}))
    with pytest.raises(MlflowException):
        infer_signature({"x": np.array([[1, 2, 3]], dtype=np.int64)})


def test_input_examples(pandas_df_with_all_types):
    sig = infer_signature(pandas_df_with_all_types)
    # test setting example with data frame with all supported data types
    with TempDir() as tmp:
        filename = save_example(tmp.path(), pandas_df_with_all_types, schema=sig.inputs)
        parsed_df = from_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()
        # the frame read without schema should match except for the binary values
        assert (parsed_df.drop(columns=["binary"]) == from_json(tmp.path(filename))
                .drop(columns=["binary"])).all().all()

    # pass the input as dictionary instead
    with TempDir() as tmp:
        sig = infer_signature(
            {name: pandas_df_with_all_types[name].values
             for name in pandas_df_with_all_types.columns})
        filename = save_example(tmp.path(), pandas_df_with_all_types, schema=sig.inputs)
        parsed_df = from_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types == parsed_df).all().all()

    # saving example with binary data and no schema should fail
    with TempDir() as tmp:
        with pytest.raises(TypeError):
            save_example(tmp.path(), pandas_df_with_all_types)
        df_without_binary = pandas_df_with_all_types.drop(columns=["binary"])
        filename = save_example(tmp.path(), df_without_binary)
        parsed_df = from_json(tmp.path(filename))
        sig = infer_signature(df_without_binary)
        for col, type in zip(sig.inputs.column_names(), sig.inputs.numpy_types()):
            parsed_df[col] = parsed_df[col].astype(type)
        assert (df_without_binary == parsed_df).all().all()

    # input passed as numpy array
    sig = infer_signature(pandas_df_with_all_types)
    with TempDir() as tmp:
        filename = save_example(tmp.path(), pandas_df_with_all_types.values,
                                schema=sig.inputs)
        parsed_df = from_json(tmp.path(filename), schema=sig.inputs)
        assert (pandas_df_with_all_types.values == parsed_df.values).all().all()

    # pass multidimensional array
    with TempDir() as tmp:
        example = np.array([[[1, 2, 3]]])
        filename = save_example(tmp.path(), example)
        parsed_df = from_json(tmp.path(filename))
        assert (example == np.array(parsed_df.values.tolist())).all()
    # pass multidimensional array
    with TempDir() as tmp:
        example = np.array([[[1, 2, 3]]])
        filename = save_example(tmp.path(), {"x": example, "y": example})
        parsed_df = from_json(tmp.path(filename))
        assert (example == np.array(parsed_df["x"].values.tolist())).all().all()
        assert (example == np.array(parsed_df["y"].values.tolist())).all().all()

    # pass dict with scalars
    with TempDir() as tmp:
        example = {"a": 1, "b": "abc"}
        filename = save_example(tmp.path(), example)
        parsed_df = from_json(tmp.path(filename))
        assert example == parsed_df.to_dict(orient="records")[0]

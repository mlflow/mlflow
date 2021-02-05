import json
import math

import numpy as np
import pandas as pd
import pytest

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType
from mlflow.types.schema import Schema, ColSpec


@pytest.fixture
def pandas_df_with_all_types():
    df = pd.DataFrame(
        {
            "boolean": [True, False, True],
            "integer": np.array([1, 2, 3], np.int32),
            "long": np.array([1, 2, 3], np.int64),
            "float": np.array([math.pi, 2 * math.pi, 3 * math.pi], np.float32),
            "double": [math.pi, 2 * math.pi, 3 * math.pi],
            "binary": [bytes([1, 2, 3]), bytes([4, 5, 6]), bytes([7, 8, 9])],
            "string": ["a", "b", "c"],
            "boolean_ext": [True, False, pd.NA],
            "integer_ext": [1, 2, pd.NA],
            "string_ext": ["a", "b", "c"],
        }
    )
    df["boolean_ext"] = df["boolean_ext"].astype("boolean")
    df["integer_ext"] = df["integer_ext"].astype("Int64")
    df["string_ext"] = df["string_ext"].astype("string")
    return df


def test_pandas_signature_inference(pandas_df_with_all_types):
    signature1 = ModelSignature(
        inputs=Schema(
            [
                ColSpec(DataType.boolean, "boolean"),
                ColSpec(DataType.integer, "integer"),
                ColSpec(DataType.long, "long"),
                ColSpec(DataType.float, "float"),
                ColSpec(DataType.double, "double"),
                ColSpec(DataType.string, "binary"),
                ColSpec(DataType.string, "string"),
                ColSpec(DataType.boolean, "boolean_ext"),
                ColSpec(DataType.long, "integer_ext"),
                ColSpec(DataType.string, "string_ext"),
            ]
        )
    )
    signature2 = infer_signature(pandas_df_with_all_types)
    assert signature1 == signature2


def test_model_signature():
    signature1 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema(
            [ColSpec(name=None, type=DataType.double), ColSpec(name=None, type=DataType.double)]
        ),
    )
    signature2 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema(
            [ColSpec(name=None, type=DataType.double), ColSpec(name=None, type=DataType.double)]
        ),
    )
    assert signature1 == signature2
    signature3 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]),
        outputs=Schema(
            [ColSpec(name=None, type=DataType.float), ColSpec(name=None, type=DataType.double)]
        ),
    )
    assert signature3 != signature1
    as_json = json.dumps(signature1.to_dict())
    signature4 = ModelSignature.from_dict(json.loads(as_json))
    assert signature1 == signature4
    signature5 = ModelSignature(
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]), outputs=None,
    )
    as_json = json.dumps(signature5.to_dict())
    signature6 = ModelSignature.from_dict(json.loads(as_json))
    assert signature5 == signature6


def test_signature_inference_infers_input_and_output_as_expected():
    sig0 = infer_signature(np.array([1]))
    assert sig0.inputs is not None
    assert sig0.outputs is None
    sig1 = infer_signature(np.array([1]), np.array([1]))
    assert sig1.inputs == sig0.inputs
    assert sig1.outputs == sig0.inputs

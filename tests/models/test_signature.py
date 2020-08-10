import json
import numpy as np

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType
from mlflow.types.schema import Schema, ColSpec


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
        inputs=Schema([ColSpec(DataType.boolean), ColSpec(DataType.binary)]), outputs=None
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

import json
import numpy as np
import pandas as pd
import pyspark

from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types import DataType
from mlflow.types.schema import Schema, ColSpec, TensorSpec


def test_model_signature_with_colspec():
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


def test_model_signature_with_tensorspec():
    signature1 = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 10))]),
    )
    signature2 = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 10))]),
    )
    # Single type mismatch
    assert signature1 == signature2
    signature3 = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]),
        outputs=Schema([TensorSpec(np.dtype("int"), (-1, 10))]),
    )
    assert signature3 != signature1
    # Name mismatch
    signature4 = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 10), "misMatch")]),
    )
    assert signature3 != signature4
    as_json = json.dumps(signature1.to_dict())
    signature5 = ModelSignature.from_dict(json.loads(as_json))
    assert signature1 == signature5

    # Test with name
    signature6 = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype("float"), (-1, 28, 28), name="image"),
                TensorSpec(np.dtype("int"), (-1, 10), name="metadata"),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 10), name="outputs")]),
    )
    signature7 = ModelSignature(
        inputs=Schema(
            [
                TensorSpec(np.dtype("float"), (-1, 28, 28), name="image"),
                TensorSpec(np.dtype("int"), (-1, 10), name="metadata"),
            ]
        ),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 10), name="outputs")]),
    )
    assert signature6 == signature7
    assert signature1 != signature6

    # Test w/o output
    signature8 = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]), outputs=None
    )
    as_json = json.dumps(signature8.to_dict())
    signature9 = ModelSignature.from_dict(json.loads(as_json))
    assert signature8 == signature9


def test_model_signature_with_colspec_and_tensorspec():
    signature1 = ModelSignature(inputs=Schema([ColSpec(DataType.double)]))
    signature2 = ModelSignature(inputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]))
    assert signature1 != signature2
    assert signature2 != signature1

    signature3 = ModelSignature(
        inputs=Schema([ColSpec(DataType.double)]),
        outputs=Schema([TensorSpec(np.dtype("float"), (-1, 28, 28))]),
    )
    signature4 = ModelSignature(
        inputs=Schema([ColSpec(DataType.double)]),
        outputs=Schema([ColSpec(DataType.double)]),
    )
    assert signature3 != signature4
    assert signature4 != signature3


def test_signature_inference_infers_input_and_output_as_expected():
    sig0 = infer_signature(np.array([1]))
    assert sig0.inputs is not None
    assert sig0.outputs is None
    sig1 = infer_signature(np.array([1]), np.array([1]))
    assert sig1.inputs == sig0.inputs
    assert sig1.outputs == sig0.inputs


def test_signature_inference_infers_datime_types_as_expected():
    col_name = "datetime_col"
    test_datetime = np.datetime64("2021-01-01")
    test_series = pd.Series(pd.to_datetime([test_datetime]))
    test_df = test_series.to_frame(col_name)

    signature = infer_signature(test_series)
    assert signature.inputs == Schema([ColSpec(DataType.datetime)])

    signature = infer_signature(test_df)
    assert signature.inputs == Schema([ColSpec(DataType.datetime, name=col_name)])

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_df = spark.range(1).selectExpr(
        "current_timestamp() as timestamp", "current_date() as date"
    )
    signature = infer_signature(spark_df)
    assert signature.inputs == Schema(
        [ColSpec(DataType.datetime, name="timestamp"), ColSpec(DataType.datetime, name="date")]
    )

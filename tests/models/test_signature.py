import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pyspark
import pytest
from sklearn.ensemble import RandomForestRegressor

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature, infer_signature, rag_signatures, set_signature
from mlflow.models.model import get_model_info
from mlflow.types import DataType
from mlflow.types.schema import (
    Array,
    ColSpec,
    ParamSchema,
    ParamSpec,
    Schema,
    TensorSpec,
    convert_dataclass_to_schema,
)


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


def test_infer_signature_on_nested_array():
    signature = infer_signature(
        model_input=[{"queries": [["a", "b", "c"], ["d", "e"], []]}],
        model_output=[{"answers": [["f", "g"], ["h"]]}],
    )
    assert signature.inputs == Schema([ColSpec(Array(Array(DataType.string)), name="queries")])
    assert signature.outputs == Schema([ColSpec(Array(Array(DataType.string)), name="answers")])

    signature = infer_signature(
        model_input=[
            {
                "inputs": [
                    np.array([["a", "b"], ["c", "d"]]),
                    np.array([["e", "f"], ["g", "h"]]),
                ]
            }
        ],
        model_output=[{"outputs": [np.int32(5), np.int32(6)]}],
    )
    assert signature.inputs == Schema(
        [ColSpec(Array(Array(Array(DataType.string))), name="inputs")]
    )
    assert signature.outputs == Schema([ColSpec(Array(DataType.integer), name="outputs")])


def test_infer_signature_on_list_of_dictionaries():
    signature = infer_signature(
        model_input=[{"query": "test query"}],
        model_output=[
            {
                "output": "Output from the LLM",
                "candidate_ids": ["412", "1233"],
                "candidate_sources": ["file1.md", "file201.md"],
            }
        ],
    )
    assert signature.inputs == Schema([ColSpec(DataType.string, name="query")])
    assert signature.outputs == Schema(
        [
            ColSpec(DataType.string, name="output"),
            ColSpec(Array(DataType.string), name="candidate_ids"),
            ColSpec(Array(DataType.string), name="candidate_sources"),
        ]
    )


def test_signature_inference_infers_datime_types_as_expected():
    col_name = "datetime_col"
    test_datetime = np.datetime64("2021-01-01")
    test_series = pd.Series(pd.to_datetime([test_datetime]))
    test_df = test_series.to_frame(col_name)

    signature = infer_signature(test_series)
    assert signature.inputs == Schema([ColSpec(DataType.datetime)])

    signature = infer_signature(test_df)
    assert signature.inputs == Schema([ColSpec(DataType.datetime, name=col_name)])

    with pyspark.sql.SparkSession.builder.getOrCreate() as spark:
        spark_df = spark.range(1).selectExpr(
            "current_timestamp() as timestamp", "current_date() as date"
        )
        signature = infer_signature(spark_df)
        assert signature.inputs == Schema(
            [ColSpec(DataType.datetime, name="timestamp"), ColSpec(DataType.datetime, name="date")]
        )


def test_set_signature_to_logged_model():
    artifact_path = "regr-model"
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(sk_model=RandomForestRegressor(), artifact_path=artifact_path)
    signature = infer_signature(np.array([1]))
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    set_signature(model_uri, signature)
    model_info = get_model_info(model_uri)
    assert model_info.signature == signature


def test_set_signature_to_saved_model(tmp_path):
    model_path = str(tmp_path)
    mlflow.sklearn.save_model(
        RandomForestRegressor(),
        model_path,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    signature = infer_signature(np.array([1]))
    set_signature(model_path, signature)
    assert Model.load(model_path).signature == signature


def test_set_signature_overwrite():
    artifact_path = "regr-model"
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=RandomForestRegressor(),
            artifact_path=artifact_path,
            signature=infer_signature(np.array([1])),
        )
    new_signature = infer_signature(np.array([1]), np.array([1]))
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    set_signature(model_uri, new_signature)
    model_info = get_model_info(model_uri)
    assert model_info.signature == new_signature


def test_cannot_set_signature_on_models_scheme_uris():
    signature = infer_signature(np.array([1]))
    with pytest.raises(
        MlflowException, match="Model URIs with the `models:/` scheme are not supported."
    ):
        set_signature("models:/dummy_model@champion", signature)


def test_signature_construction():
    signature = ModelSignature(inputs=Schema([ColSpec(DataType.binary)]))
    assert signature.to_dict() == {
        "inputs": '[{"type": "binary", "required": true}]',
        "outputs": None,
        "params": None,
    }

    signature = ModelSignature(outputs=Schema([ColSpec(DataType.double)]))
    assert signature.to_dict() == {
        "inputs": None,
        "outputs": '[{"type": "double", "required": true}]',
        "params": None,
    }

    signature = ModelSignature(params=ParamSchema([ParamSpec("param1", DataType.string, "test")]))
    assert signature.to_dict() == {
        "inputs": None,
        "outputs": None,
        "params": '[{"name": "param1", "type": "string", "default": "test", "shape": null}]',
    }


def test_signature_with_errors():
    with pytest.raises(
        TypeError,
        match=r"inputs must be either None, mlflow.models.signature.Schema, or a dataclass",
    ):
        ModelSignature(inputs=1)

    with pytest.raises(
        ValueError, match=r"At least one of inputs, outputs or params must be provided"
    ):
        ModelSignature()


def test_signature_for_rag():
    signature = ModelSignature(
        inputs=rag_signatures.ChatCompletionRequest(),
        outputs=rag_signatures.ChatCompletionResponse(),
    )
    signature_dict = signature.to_dict()
    assert signature_dict == {
        "inputs": (
            '[{"type": "array", "items": {"type": "object", "properties": '
            '{"content": {"type": "string", "required": true}, '
            '"role": {"type": "string", "required": true}}}, '
            '"name": "messages", "required": true}]'
        ),
        "outputs": (
            '[{"type": "array", "items": {"type": "object", "properties": '
            '{"finish_reason": {"type": "string", "required": true}, '
            '"index": {"type": "long", "required": true}, '
            '"message": {"type": "object", "properties": '
            '{"content": {"type": "string", "required": true}, '
            '"role": {"type": "string", "required": true}}, '
            '"required": true}}}, "name": "choices", "required": true}, '
            '{"type": "string", "name": "object", "required": true}]'
        ),
        "params": None,
    }


def test_infer_signature_and_convert_dataclass_to_schema_for_rag():
    inferred_signature = infer_signature(
        asdict(rag_signatures.ChatCompletionRequest()),
        asdict(rag_signatures.ChatCompletionResponse()),
    )
    input_schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionRequest())
    output_schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionResponse())
    assert inferred_signature.inputs == input_schema
    assert inferred_signature.outputs == output_schema


def test_infer_signature_with_dataclass():
    inferred_signature = infer_signature(
        rag_signatures.ChatCompletionRequest(),
        rag_signatures.ChatCompletionResponse(),
    )
    input_schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionRequest())
    output_schema = convert_dataclass_to_schema(rag_signatures.ChatCompletionResponse())
    assert inferred_signature.inputs == input_schema
    assert inferred_signature.outputs == output_schema


@dataclass
class CustomInput:
    id: int = 0


@dataclass
class CustomOutput:
    id: int = 0


@dataclass
class FlexibleChatCompletionRequest(rag_signatures.ChatCompletionRequest):
    custom_input: Optional[CustomInput] = None


@dataclass
class FlexibleChatCompletionResponse(rag_signatures.ChatCompletionResponse):
    custom_output: Optional[CustomOutput] = None


def test_infer_signature_with_optional_and_child_dataclass():
    inferred_signature = infer_signature(
        FlexibleChatCompletionRequest(),
        FlexibleChatCompletionResponse(),
    )
    custom_input_schema = next(
        schema for schema in inferred_signature.inputs.to_dict() if schema["name"] == "custom_input"
    )
    assert custom_input_schema["required"] is False
    assert "id" in custom_input_schema["properties"]
    assert any(
        schema for schema in inferred_signature.inputs.to_dict() if schema["name"] == "messages"
    )

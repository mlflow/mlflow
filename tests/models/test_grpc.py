import pytest
import numpy as np

import mlflow.models.grpc as grpc
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec, DataType, TensorSpec
from mlflow.types.utils import TensorsNotSupportedException


@pytest.fixture()
def sample_schema():
    return Schema(
        [ColSpec(name="c1", type=DataType.boolean), ColSpec(name="c2", type=DataType.integer)]
    )


@pytest.fixture()
def sample_schema_with_datetime():
    return Schema(
        [ColSpec(name="c1", type=DataType.boolean), ColSpec(name="c2", type=DataType.datetime)]
    )


@pytest.fixture
def schema_with_all_types_dynamic():
    s = Schema([])
    for t in DataType:
        s.inputs.append(ColSpec(name="col_{}".format(t.name), type=t))
    return s


def _clean_string(s: str):
    return s.replace(" ", "").replace("\n", "")


def test_grpc_grammar_generation(sample_schema):
    signature = ModelSignature(inputs=sample_schema, outputs=sample_schema)
    grpc_grammar = grpc.generate_grammar(signature)

    expected_grammar = """
        syntax = "proto3";
        
        package org.mlflow.models.grpc;
        
        message ModelInput {
                bool c1 = 1;
                int32 c2 = 2;
        }
        
        message ModelOutput {
                bool c1 = 1;
                int32 c2 = 2;
        }
        
        service ModelService {
          rpc predict (ModelInput) returns (ModelOutput) {}
        }
    """

    assert _clean_string(expected_grammar) == _clean_string(grpc_grammar)


def test_grpc_grammar_generation_with_datetime(sample_schema_with_datetime):
    signature = ModelSignature(
        inputs=sample_schema_with_datetime, outputs=sample_schema_with_datetime
    )
    grpc_grammar = grpc.generate_grammar(signature)

    expected_grammar = """
        syntax = "proto3";
        
        package org.mlflow.models.grpc;
        
        import "google/protobuf/timestamp.proto";
        
        message ModelInput {
                bool c1 = 1;
                google.protobuf.Timestamp c2 = 2;
        }
        
        message ModelOutput {
                bool c1 = 1;
                google.protobuf.Timestamp c2 = 2;
        }
        
        service ModelService {
          rpc predict (ModelInput) returns (ModelOutput) {}
        }
    """

    assert _clean_string(expected_grammar) == _clean_string(grpc_grammar)


def test_grpc_type_mapping():
    schema_with_all_types = Schema(
        [
            ColSpec(name="feature1", type=DataType.boolean),
            ColSpec(name="feature2", type=DataType.integer),
            ColSpec(name="feature3", type=DataType.long),
            ColSpec(name="feature4", type=DataType.float),
            ColSpec(name="feature5", type=DataType.double),
            ColSpec(name="feature6", type=DataType.string),
            ColSpec(name="feature7", type=DataType.binary),
            ColSpec(name="feature8", type=DataType.datetime),
        ]
    )
    expected_mappings = [
        ("bool", "feature1"),
        ("int32", "feature2"),
        ("int64", "feature3"),
        ("float", "feature4"),
        ("double", "feature5"),
        ("string", "feature6"),
        ("bytes", "feature7"),
        ("google.protobuf.Timestamp", "feature8"),
    ]
    mappings = grpc._map_to_proto_types(schema_with_all_types)

    assert sorted(expected_mappings) == sorted(mappings)


def test_all_data_types_are_supported(schema_with_all_types_dynamic, sample_schema):
    try:
        signature_with_all_types = ModelSignature(
            inputs=schema_with_all_types_dynamic, outputs=sample_schema
        )
        _ = grpc.generate_grammar(signature_with_all_types)
    except MlflowException:
        assert False, "gRPC grammar generation raised an exception: unsupported type"


def test_exception_for_missing_signature_output(sample_schema):
    signature_without_outputs = ModelSignature(inputs=sample_schema, outputs=None)

    with pytest.raises(MlflowException):
        _ = grpc.generate_grammar(signature_without_outputs)


def test_exception_for_signature_with_tensors(sample_schema):
    schema_with_tensor = Schema([TensorSpec(np.dtype("str"), [-1], "a")])

    signature_with_input_tensor = ModelSignature(inputs=schema_with_tensor, outputs=sample_schema)
    with pytest.raises(TensorsNotSupportedException):
        _ = grpc.generate_grammar(signature_with_input_tensor)

    signature_with_output_tensor = ModelSignature(inputs=sample_schema, outputs=schema_with_tensor)
    with pytest.raises(TensorsNotSupportedException):
        _ = grpc.generate_grammar(signature_with_output_tensor)


def test_exception_for_missing_feature_name(sample_schema):
    schema_with_no_feature_name = Schema([ColSpec(type=DataType.boolean)])
    signature_with_no_feature_name = ModelSignature(
        inputs=schema_with_no_feature_name, outputs=sample_schema
    )

    with pytest.raises(MlflowException):
        _ = grpc.generate_grammar(signature_with_no_feature_name)

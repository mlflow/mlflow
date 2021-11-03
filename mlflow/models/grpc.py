"""
The :py:mod:`mlflow.models.grpc` module provides an API to generate the model gRPC grammar.

The grammar is generated using the Model Signature. See :py:class:`mlflow.models.signature.ModelSignature`
for more details on the signature.
"""
from typing import List, Tuple

from jinja2 import Template

from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec, DataType
from mlflow.types.utils import TensorsNotSupportedException

_TEMPLATE = """
syntax = "proto3";

package {{ package }};

{% if importTimestamp %}
import "google/protobuf/timestamp.proto";
{% endif %}
        
message ModelInput {
    {% for input in inputs %}
        {{ input[0] }} {{ input[1] }} = {{loop.index}};
    {% endfor %}
}

message ModelOutput {
    {% for output in outputs %}
        {{ output[0] }} {{ output[1] }} = {{loop.index}};
    {% endfor %}
}

service ModelService {
  rpc predict (ModelInput) returns (ModelOutput) {}
}
"""

_DEFAULT_PACKAGE = "org.mlflow.models.grpc"

_NUMPY_TO_PROTO_TYPE = {
    DataType.boolean.name: "bool",
    DataType.integer.name: "int32",
    DataType.long.name: "int64",
    DataType.float.name: "float",
    DataType.double.name: "double",
    DataType.string.name: "string",
    DataType.binary.name: "bytes",
    DataType.datetime.name: "google.protobuf.Timestamp",
}


def generate_grammar(signature: ModelSignature, package: str = _DEFAULT_PACKAGE) -> str:
    """
    Generates the gRPC grammar based on the Model Signature.
    The schema types are converted into protobuf types.

    This method will raise an exception if any of the following conditions is matched:
    - the signature does not have the output schema
    - the signature contains tensors or columns with unsupported types
    - the signature contains columns without name

    :param signature: the Model Signature.
    :param package: an optional string to override the default package `org.mlflow.models.grpc` in the gRPC grammar.

    :return: The gRPC grammar as string.
    """
    if signature.outputs is None:
        raise MlflowException(
            "Output signature is not present for signature {}".format(signature.to_dict())
        )

    input_signature: List[Tuple[str, str]] = _map_to_proto_types(signature.inputs)
    output_signature: List[Tuple[str, str]] = _map_to_proto_types(signature.outputs)

    has_datetime = any(
        map(
            lambda t: t[0] == _NUMPY_TO_PROTO_TYPE[DataType.datetime.name],
            input_signature + output_signature,
        )
    )

    template = Template(_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(
        package=package,
        importTimestamp=has_datetime,
        inputs=input_signature,
        outputs=output_signature,
    )


def _map_to_proto_types(schema: Schema) -> List[Tuple[str, str]]:
    """
    Transforms the schema into a list of tuples (protobuf-type, name).

    This method will raise an exception if the schema is a tensor.

    :param schema: the Schema.

    :return: A list of tuples containing the column protobuf type and the column name.
    """
    if schema.is_tensor_spec():
        raise TensorsNotSupportedException("Invalid schema '{}'".format(schema.to_json()))

    return list(map(_map_col_spec, schema.inputs))


def _map_col_spec(input: ColSpec) -> Tuple[str, str]:
    """
    Transforms a :py:class:`mlflow.types.ColSpec` into a tuple (protobuf-type, name).

    This method will raise an exception if either the column type can't be mapped to one
    of :py:class:`mlflow.types.DataType` or the column name is not present.

    :param input: the column to transform.

    :return: A tuple containing the protobuf type and the name.
    """
    if input.type.name not in _NUMPY_TO_PROTO_TYPE:
        raise MlflowException("Invalid type for ColSpec {}".format(input.to_dict()))

    if input.name is None:
        raise MlflowException("Missing name for ColSpec {}".format(input.to_dict()))

    return _NUMPY_TO_PROTO_TYPE.get(input.type.name), input.name

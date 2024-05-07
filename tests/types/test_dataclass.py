
from mlflow.models import rag_signatures
from mlflow.types.dataclass import deserialize_dataclass, serialize_dataclass


def serialize_and_deserialize(dataclass_instance):
    ser = serialize_dataclass(dataclass_instance)
    de = deserialize_dataclass(ser, type(dataclass_instance))
    assert de == dataclass_instance

def test_serialize_dataclass():
    serialize_and_deserialize(rag_signatures.Message())
    serialize_and_deserialize(rag_signatures.ChatCompletionRequest())
    serialize_and_deserialize(rag_signatures.MultiturnChatRequest())
    serialize_and_deserialize(rag_signatures.MultiturnChatRequest(query="foo", history=[rag_signatures.Message(role="system", content="bar")]))

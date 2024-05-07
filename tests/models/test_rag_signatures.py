
from mlflow.models import rag_signatures


def serialize_and_deserialize(dataclass_instance):
    ser = serialize_dataclass(dataclass_instance)
    print(ser)
    de = deserialize_dataclass(ser, type(dataclass_instance))
    print(de)
    assert de == dataclass_instance

def test_serialize_dataclass():
    serialize_and_deserialize(rag_signatures.Message())
    serialize_and_deserialize(rag_signatures.ChatCompletionRequest())
    serialize_and_deserialize(rag_signatures.MultiturnChatRequest())
    serialize_and_deserialize(rag_signatures.MultiturnChatRequest(query="foo", history=[rag_signatures.Message(role="system", content="bar")]))

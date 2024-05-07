import dataclasses
import json

from mlflow.models import rag_signatures


# Serialization function
def serialize_dataclass(instance):
    def to_serializable(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_serializable(item) for item in obj]
        else:
            return obj
    return json.dumps(to_serializable(instance))

# Deserialization function
def deserialize_dataclass(json_data, cls):
    data = json.loads(json_data)

    def from_serializable(data, cls):
        if dataclasses.is_dataclass(cls):
            field_types = {f.name: f.type for f in dataclasses.fields(cls)}
            return cls(**{f: from_serializable(data.get(f), field_types[f]) for f in field_types})
        elif hasattr(cls, '__origin__') and cls.__origin__ is list:  # Check for List
            item_type = cls.__args__[0]
            return [from_serializable(item, item_type) for item in data]
        else:
            return data

    return from_serializable(data, cls)


def test_serialize_dataclass():
    ser = serialize_dataclass(rag_signatures.Message())
    print(ser)
    de = deserialize_dataclass(ser, rag_signatures.Message)
    print(de)
    assert de == rag_signatures.Message()

    ser = serialize_dataclass(rag_signatures.ChatCompletionRequest())
    print(ser)
    de = deserialize_dataclass(ser, rag_signatures.ChatCompletionRequest)
    print(de)
    assert de == rag_signatures.ChatCompletionRequest()

    ser = serialize_dataclass(rag_signatures.MultiturnChatRequest())
    print(ser)
    de = deserialize_dataclass(ser, rag_signatures.MultiturnChatRequest)
    print(de)
    assert de == rag_signatures.MultiturnChatRequest()

    ser = serialize_dataclass(rag_signatures.MultiturnChatRequest(query="foo", history=[rag_signatures.Message(role="system", content="bar")]))
    print(ser)
    de = deserialize_dataclass(ser, rag_signatures.MultiturnChatRequest)
    print(de)
    assert de == rag_signatures.MultiturnChatRequest(query="foo", history=[rag_signatures.Message(role="system", content="bar")])

import dataclasses
import json
from typing import Union

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
        elif hasattr(cls, '__origin__') and cls.__origin__ is Union: # Check for Union
            for item_type in cls.__args__:
                try:
                    return from_serializable(data, item_type)
                except:
                    pass
            raise ValueError(f"Could not deserialize {data} as {cls}")
        else:
            return data

    return from_serializable(data, cls)


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

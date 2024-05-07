    # Serialization function
import dataclasses


def serialize_dataclass(instance):
    def to_serializable(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_serializable(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_serializable(item) for item in obj]
        else:
            return obj
    return to_serializable(instance)

# Deserialization function
def deserialize_dataclass(data, cls):
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

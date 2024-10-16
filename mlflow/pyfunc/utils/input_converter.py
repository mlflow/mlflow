from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, Union

from mlflow.utils.annotations import experimental


def _is_optional_dataclass(field_type) -> bool:
    """Check if the field type is an Optional containing a dataclass."""
    if get_origin(field_type) is Union:
        inner_types = get_args(field_type)
        # Check if it's a Union[Dataclass, NoneType] (i.e., Optional[Dataclass])
        if len(inner_types) == 2 and inner_types[1] == type(None):
            return is_dataclass(inner_types[0])
    return False


@experimental
def _hydrate_dataclass(dataclass_type, data):
    """Recursively create an instance of the dataclass_type from data."""
    if not (is_dataclass(dataclass_type) or _is_optional_dataclass(dataclass_type)):
        raise ValueError(f"{dataclass_type.__name__} is not a dataclass")

    field_names = {f.name: f.type for f in fields(dataclass_type)}
    kwargs = {}
    for key, field_type in field_names.items():
        if key in data:
            value = data[key]
            if is_dataclass(field_type):
                kwargs[key] = _hydrate_dataclass(field_type, value)
            elif _is_optional_dataclass(field_type):
                item_type = get_args(field_type)[0]
                kwargs[key] = _hydrate_dataclass(item_type, value)
            elif get_origin(field_type) == list:
                item_type = get_args(field_type)[0]
                if is_dataclass(item_type):
                    kwargs[key] = [_hydrate_dataclass(item_type, item) for item in value]
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value
    return dataclass_type(**kwargs)

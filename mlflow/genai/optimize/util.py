from typing import Any, Union

from pydantic import BaseModel, create_model


def infer_type_from_value(value: Any, model_name: str = "Output") -> type:
    """
    Infer the type from the value.
    Only supports primitive types, lists, and dict and Pydantic models.
    """
    if value is None:
        return type(None)
    elif isinstance(value, (bool, int, float, str)):
        return type(value)
    elif isinstance(value, list):
        if not value:
            return list[Any]
        element_types = set()
        for item in value:
            element_types.add(infer_type_from_value(item))
        if len(element_types) == 1:
            return list[element_types.pop()]
        else:
            return list[Union[tuple(element_types)]]
    elif isinstance(value, dict):
        fields = {k: (infer_type_from_value(v, model_name=k), ...) for k, v in value.items()}
        return create_model(model_name, **fields)
    elif isinstance(value, BaseModel):
        return type(value)
    return Any

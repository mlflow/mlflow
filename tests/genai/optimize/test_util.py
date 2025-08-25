from typing import Any, Union

import pytest
from pydantic import BaseModel

from mlflow.genai.optimize.util import infer_type_from_value


@pytest.mark.parametrize(
    ("input_value", "expected_type"),
    [
        (None, type(None)),
        (True, bool),
        (42, int),
        (3.14, float),
        ("hello", str),
    ],
)
def test_infer_primitive_types(input_value, expected_type):
    assert infer_type_from_value(input_value) == expected_type


@pytest.mark.parametrize(
    ("input_list", "expected_type"),
    [
        ([], list[Any]),
        ([1, 2, 3], list[int]),
        (["a", "b", "c"], list[str]),
        ([1.0, 2.0, 3.0], list[float]),
        ([True, False, True], list[bool]),
        ([1, "hello", True], list[Union[int, str, bool]]),  # noqa: UP007
        ([1, "hello", True], list[int | str | bool]),
        ([1, 2.0], list[int | float]),
        ([[1, 2], [3, 4]], list[list[int]]),
        ([["a"], ["b", "c"]], list[list[str]]),
    ],
)
def test_infer_list_types(input_list, expected_type):
    assert infer_type_from_value(input_list) == expected_type


@pytest.mark.parametrize(
    ("input_dict", "expected_fields"),
    [
        ({"name": "John", "age": 30, "active": True}, {"name": str, "age": int, "active": bool}),
        ({"score": 95.5, "passed": True}, {"score": float, "passed": bool}),
    ],
)
def test_infer_simple_dict(input_dict, expected_fields):
    result = infer_type_from_value(input_dict)

    assert isinstance(result, type)
    assert issubclass(result, BaseModel)

    for field_name, expected_type in expected_fields.items():
        assert result.__annotations__[field_name] == expected_type


def test_infer_nested_dict():
    data = {
        "user": {"name": "John", "scores": [85, 90, 95]},
        "settings": {"enabled": True, "theme": "dark"},
    }
    result = infer_type_from_value(data)

    assert isinstance(result, type)
    assert issubclass(result, BaseModel)

    # Check nested model types
    user_model = result.__annotations__["user"]
    settings_model = result.__annotations__["settings"]

    assert issubclass(user_model, BaseModel)
    assert issubclass(settings_model, BaseModel)

    # Check nested field types
    assert user_model.__annotations__["name"] == str
    assert user_model.__annotations__["scores"] == list[int]
    assert settings_model.__annotations__["enabled"] == bool
    assert settings_model.__annotations__["theme"] == str


@pytest.mark.parametrize(
    ("model_class", "model_data"),
    [
        (
            type("UserModel", (BaseModel,), {"__annotations__": {"name": str, "age": int}}),
            {"name": "John", "age": 30},
        ),
        (
            type("ProductModel", (BaseModel,), {"__annotations__": {"id": int, "price": float}}),
            {"id": 1, "price": 99.99},
        ),
    ],
)
def test_infer_pydantic_model(model_class, model_data):
    model = model_class(**model_data)
    result = infer_type_from_value(model)
    assert result == model_class


@pytest.mark.parametrize(
    "type_to_infer",
    [
        type("CustomClass", (), {}),
        type("AnotherClass", (), {"custom_attr": 42}),
    ],
)
def test_infer_unsupported_type(type_to_infer):
    obj = type_to_infer()
    assert infer_type_from_value(obj) == Any


@pytest.mark.parametrize(
    ("input_dict", "model_name"),
    [
        ({"name": "John", "age": 30}, "UserData"),
        ({"id": 1, "value": "test"}, "TestModel"),
    ],
)
def test_model_name_parameter(input_dict, model_name):
    result = infer_type_from_value(input_dict, model_name=model_name)
    assert result.__name__ == model_name

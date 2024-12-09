import datetime
from typing import Any, Dict, List, Optional, Union

import pydantic
import pytest

from mlflow.exceptions import MlflowException
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema
from mlflow.types.type_hints import (
    PYDANTIC_V1_OR_OLDER,
    _infer_schema_from_type_hint,
    _validate_example_against_type_hint,
)


class CustomModel(pydantic.BaseModel):
    long_field: int
    str_field: str
    bool_field: bool
    double_field: float
    binary_field: bytes
    datetime_field: datetime.datetime
    any_field: Any
    optional_str: Optional[str] = None


class Message(pydantic.BaseModel):
    role: str
    content: str


class CustomModel2(pydantic.BaseModel):
    custom_field: dict[str, Any]
    messages: list[Message]
    optional_int: Optional[int] = None


@pytest.mark.parametrize(
    ("type_hint", "expected_schema"),
    [
        (
            CustomModel,
            Schema(
                [
                    ColSpec(type=DataType.long, name="long_field"),
                    ColSpec(type=DataType.string, name="str_field"),
                    ColSpec(type=DataType.boolean, name="bool_field"),
                    ColSpec(type=DataType.double, name="double_field"),
                    ColSpec(type=DataType.binary, name="binary_field"),
                    ColSpec(type=DataType.datetime, name="datetime_field"),
                    ColSpec(type=AnyType(), name="any_field"),
                    ColSpec(type=DataType.string, name="optional_str", required=False),
                ]
            ),
        ),
        (
            list[CustomModel],
            Schema(
                [
                    ColSpec(
                        type=Array(
                            Object(
                                [
                                    Property(name="long_field", dtype=DataType.long),
                                    Property(name="str_field", dtype=DataType.string),
                                    Property(name="bool_field", dtype=DataType.boolean),
                                    Property(name="double_field", dtype=DataType.double),
                                    Property(name="binary_field", dtype=DataType.binary),
                                    Property(name="datetime_field", dtype=DataType.datetime),
                                    Property(name="any_field", dtype=AnyType()),
                                    Property(
                                        name="optional_str", dtype=DataType.string, required=False
                                    ),
                                ]
                            )
                        )
                    )
                ]
            ),
        ),
        (
            CustomModel2,
            Schema(
                [
                    ColSpec(type=Map(AnyType()), name="custom_field"),
                    ColSpec(
                        type=Array(
                            Object(
                                [
                                    Property(name="role", dtype=DataType.string),
                                    Property(name="content", dtype=DataType.string),
                                ]
                            )
                        ),
                        name="messages",
                    ),
                    ColSpec(type=DataType.long, name="optional_int", required=False),
                ]
            ),
        ),
    ],
)
def test_infer_schema_from_pydantic_model(type_hint, expected_schema):
    schema = _infer_schema_from_type_hint(type_hint)
    assert schema == expected_schema


@pytest.mark.parametrize(
    ("type_hint", "expected_schema"),
    [
        # scalars
        (int, Schema([ColSpec(type=DataType.long)])),
        (str, Schema([ColSpec(type=DataType.string)])),
        (bool, Schema([ColSpec(type=DataType.boolean)])),
        (float, Schema([ColSpec(type=DataType.double)])),
        (bytes, Schema([ColSpec(type=DataType.binary)])),
        (datetime.datetime, Schema([ColSpec(type=DataType.datetime)])),
        # lists
        (list[str], Schema([ColSpec(type=Array(DataType.string))])),
        (List[str], Schema([ColSpec(type=Array(DataType.string))])),  # noqa: UP006
        (list[list[str]], Schema([ColSpec(type=Array(Array(DataType.string)))])),
        (List[List[str]], Schema([ColSpec(type=Array(Array(DataType.string)))])),  # noqa: UP006
        # dictionaries
        (dict[str, int], Schema([ColSpec(type=Map(DataType.long))])),
        (Dict[str, int], Schema([ColSpec(type=Map(DataType.long))])),  # noqa: UP006
        (dict[str, list[str]], Schema([ColSpec(type=Map(Array(DataType.string)))])),
        (Dict[str, List[str]], Schema([ColSpec(type=Map(Array(DataType.string)))])),  # noqa: UP006
        # Union
        (Union[int, str], Schema([ColSpec(type=AnyType())])),
        # Any
        (list[Any], Schema([ColSpec(type=Array(AnyType()))])),
    ],
)
def test_infer_schema_from_python_type_hints(type_hint, expected_schema):
    schema = _infer_schema_from_type_hint(type_hint)
    assert schema == expected_schema


def test_infer_schema_from_type_hints_errors():
    class InvalidModel(pydantic.BaseModel):
        bool_field: Optional[bool]

    if not PYDANTIC_V1_OR_OLDER:
        message = (
            r"Optional field `bool_field` in Pydantic model `InvalidModel` "
            r"doesn't have a default value. Please set default value to None for this field."
        )
        with pytest.raises(
            MlflowException,
            match=message,
        ):
            _infer_schema_from_type_hint(InvalidModel)

        with pytest.raises(MlflowException, match=message):
            _infer_schema_from_type_hint(list[InvalidModel])

    message = r"If you would like to use Optional types, use a Pydantic-based type hint definition."
    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(Optional[str])

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(Union[str, int, type(None)])

    with pytest.raises(
        MlflowException, match=r"List type hint must contain only one internal type"
    ):
        _infer_schema_from_type_hint(list[str, int])

    with pytest.raises(MlflowException, match=r"Dictionary key type must be str"):
        _infer_schema_from_type_hint(dict[int, int])

    with pytest.raises(
        MlflowException, match=r"Dictionary type hint must contain two internal types"
    ):
        _infer_schema_from_type_hint(dict[int])

    message = r"it must include a valid internal type"
    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(Union)

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(Optional)

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(list)

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_type_hint(dict)

    with pytest.raises(MlflowException, match=r"Unsupported type hint"):
        _infer_schema_from_type_hint(object)


@pytest.mark.parametrize(
    ("type_hint", "example"),
    [
        (
            CustomModel,
            {
                "long_field": 1,
                "str_field": "a",
                "bool_field": False,
                "double_field": 1.0,
                "binary_field": b"abc",
                "datetime_field": datetime.datetime.now(),
                "any_field": "a",
                "optional_str": "b",
            },
        ),
        (
            CustomModel,
            {
                "long_field": 1,
                "str_field": "a",
                "bool_field": True,
                "double_field": 1.0,
                "binary_field": b"abc",
                "datetime_field": datetime.datetime.now(),
                "any_field": "a",
            },
        ),
        (
            list[CustomModel],
            [
                {
                    "long_field": 1,
                    "str_field": "a",
                    "bool_field": True,
                    "double_field": 1.0,
                    "binary_field": b"abc",
                    "datetime_field": datetime.datetime.now(),
                    "any_field": "a",
                    "optional_str": "b",
                },
                {
                    "long_field": 2,
                    "str_field": "b",
                    "bool_field": False,
                    "double_field": 2.0,
                    "binary_field": b"def",
                    "datetime_field": datetime.datetime.now(),
                    "any_field": "b",
                },
            ],
        ),
        (
            CustomModel2,
            {
                "custom_field": {"a": 1},
                "messages": [{"role": "a", "content": "b"}],
                "optional_int": 1,
            },
        ),
        (
            CustomModel2,
            {
                "custom_field": {"a": "abc"},
                "messages": [{"role": "a", "content": "b"}, {"role": "c", "content": "d"}],
            },
        ),
    ],
)
def test_pydantic_model_validation(type_hint, example):
    _validate_example_against_type_hint(example=example, type_hint=type_hint)


@pytest.mark.parametrize(
    ("type_hint", "example"),
    [
        (int, 1),
        (str, "a"),
        (bool, True),
        (float, 1.0),
        (bytes, b"abc"),
        (datetime.datetime, datetime.datetime.now()),
        (Any, "a"),
        (Any, ["a", 1]),
        (list[str], ["a", "b"]),
        (list[list[str]], [["a", "b"], ["c", "d"]]),
        (dict[str, int], {"a": 1, "b": 2}),
        (dict[str, list[str]], {"a": ["a", "b"], "b": ["c", "d"]}),
        (Union[int, str], 1),
        (Union[int, str], "a"),
        # Union type is inferred as AnyType, so it accepts double here as well
        (Union[int, str], 1.2),
        (list[Any], [1, "a"]),
    ],
)
def test_python_type_hints_validation(type_hint, example):
    _validate_example_against_type_hint(example=example, type_hint=type_hint)


def test_type_hints_validation_errors():
    with pytest.raises(
        MlflowException, match=r"Input example is not valid for Pydantic model `CustomModel`"
    ):
        _validate_example_against_type_hint({"long_field": 1, "str_field": "a"}, CustomModel)

    with pytest.raises(MlflowException, match=r"Expected type <class 'int'>, but got str"):
        _validate_example_against_type_hint("a", int)

    with pytest.raises(MlflowException, match=r"Expected list, but got str"):
        _validate_example_against_type_hint("a", list[str])

    with pytest.raises(
        MlflowException,
        match=r'Invalid elements in list: {\'1\': "Expected type <class \'str\'>, but got int"}',
    ):
        _validate_example_against_type_hint(["a", 1], list[str])

    with pytest.raises(
        MlflowException,
        match=r"Expected dict, but got list",
    ):
        _validate_example_against_type_hint(["a", 1], dict[str, int])

    with pytest.raises(
        MlflowException,
        match=r"Invalid elements in dict: {'1': 'Key must be a string, got int', "
        r"'a': 'Expected list, but got int'}",
    ):
        _validate_example_against_type_hint({1: ["a", "b"], "a": 1}, dict[str, list[str]])

    with pytest.raises(
        MlflowException,
        match=r"Expected type <class 'int'>, but got str",
    ):
        _validate_example_against_type_hint("a", Optional[int])

    with pytest.raises(
        MlflowException,
        match=r"Unsupported type hint `<class 'list'>`, it must include a valid internal type.",
    ):
        _validate_example_against_type_hint(["a"], list)

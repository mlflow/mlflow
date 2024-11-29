import datetime
from typing import Any, Optional, Union

import pydantic
import pytest

from mlflow.exceptions import MlflowException
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema
from mlflow.types.type_hints import _infer_schema_from_type_hint


class CustomModel(pydantic.BaseModel):
    long_field: int
    str_field: str
    bool_field: Optional[bool] = None
    double_field: float
    binary_field: bytes
    datetime_field: datetime.datetime
    any_field: Any


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
                    ColSpec(type=DataType.boolean, name="bool_field", required=False),
                    ColSpec(type=DataType.double, name="double_field"),
                    ColSpec(type=DataType.binary, name="binary_field"),
                    ColSpec(type=DataType.datetime, name="datetime_field"),
                    ColSpec(type=AnyType(), name="any_field"),
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
                                    Property(
                                        name="bool_field", dtype=DataType.boolean, required=False
                                    ),
                                    Property(name="double_field", dtype=DataType.double),
                                    Property(name="binary_field", dtype=DataType.binary),
                                    Property(name="datetime_field", dtype=DataType.datetime),
                                    Property(name="any_field", dtype=AnyType()),
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
        (list[list[str]], Schema([ColSpec(type=Array(Array(DataType.string)))])),
        # dictionaries
        (dict[str, int], Schema([ColSpec(type=Map(DataType.long))])),
        (dict[str, list[str]], Schema([ColSpec(type=Map(Array(DataType.string)))])),
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

    message = r"Optional type hint is not supported"
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

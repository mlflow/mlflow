import datetime
from typing import Any, Dict, List, Optional, Union, get_args
from unittest import mock

import numpy as np
import pandas as pd
import pydantic
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _enforce_schema
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema
from mlflow.types.type_hints import (
    InvalidTypeHintException,
    UnsupportedTypeHintException,
    _convert_data_to_type_hint,
    _convert_dataframe_to_example_format,
    _infer_schema_from_list_type_hint,
    _is_example_valid_for_type_from_example,
    _signature_cannot_be_inferred_from_type_hint,
    _validate_data_against_type_hint,
)
from mlflow.types.utils import _infer_schema
from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER


class CustomModel(pydantic.BaseModel):
    long_field: int
    str_field: str
    bool_field: bool
    double_field: float
    binary_field: bytes
    datetime_field: datetime.datetime
    any_field: Any
    optional_str: str | None = None


class Message(pydantic.BaseModel):
    role: str
    content: str


class CustomModel2(pydantic.BaseModel):
    custom_field: dict[str, Any]
    messages: list[Message]
    optional_int: int | None = None


@pytest.mark.parametrize(
    ("type_hint", "expected_schema"),
    [
        (
            list[CustomModel],
            Schema(
                [
                    ColSpec(
                        type=Object(
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
                ]
            ),
        ),
        (
            list[list[CustomModel]],
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
            list[CustomModel2],
            Schema(
                [
                    ColSpec(
                        type=Object(
                            [
                                Property(name="custom_field", dtype=Map(AnyType())),
                                Property(
                                    name="messages",
                                    dtype=Array(
                                        Object(
                                            [
                                                Property(name="role", dtype=DataType.string),
                                                Property(name="content", dtype=DataType.string),
                                            ]
                                        )
                                    ),
                                ),
                                Property(name="optional_int", dtype=DataType.long, required=False),
                            ]
                        )
                    )
                ]
            ),
        ),
    ],
)
def test_infer_schema_from_pydantic_model(type_hint, expected_schema):
    schema = _infer_schema_from_list_type_hint(type_hint)
    assert schema == expected_schema


@pytest.mark.parametrize(
    ("type_hint", "expected_schema"),
    [
        # scalars
        (list[int], Schema([ColSpec(type=DataType.long)])),
        (list[str], Schema([ColSpec(type=DataType.string)])),
        (list[bool], Schema([ColSpec(type=DataType.boolean)])),
        (list[float], Schema([ColSpec(type=DataType.double)])),
        (list[bytes], Schema([ColSpec(type=DataType.binary)])),
        (list[datetime.datetime], Schema([ColSpec(type=DataType.datetime)])),
        # lists
        (list[list[str]], Schema([ColSpec(type=Array(DataType.string))])),
        (List[List[str]], Schema([ColSpec(type=Array(DataType.string))])),  # noqa: UP006
        (list[list[list[str]]], Schema([ColSpec(type=Array(Array(DataType.string)))])),
        (List[List[List[str]]], Schema([ColSpec(type=Array(Array(DataType.string)))])),  # noqa: UP006
        # dictionaries
        (list[dict[str, str]], Schema([ColSpec(type=Map(DataType.string))])),
        (list[dict[str, int]], Schema([ColSpec(type=Map(DataType.long))])),
        (list[Dict[str, int]], Schema([ColSpec(type=Map(DataType.long))])),  # noqa: UP006
        (list[dict[str, list[str]]], Schema([ColSpec(type=Map(Array(DataType.string)))])),
        (list[Dict[str, List[str]]], Schema([ColSpec(type=Map(Array(DataType.string)))])),  # noqa: UP006
        # Union
        (list[Union[int, str]], Schema([ColSpec(type=AnyType())])),  # noqa: UP007
        (list[int | str], Schema([ColSpec(type=AnyType())])),
        (list[list[int | str]], Schema([ColSpec(type=Array(AnyType()))])),
        # Any
        (list[Any], Schema([ColSpec(type=AnyType())])),
        (list[list[Any]], Schema([ColSpec(type=Array(AnyType()))])),
    ],
)
def test_infer_schema_from_python_type_hints(type_hint, expected_schema):
    schema = _infer_schema_from_list_type_hint(type_hint)
    assert schema == expected_schema


@pytest.mark.parametrize(
    "type_hint",
    [
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        csc_matrix,
        csr_matrix,
    ],
)
def test_type_hints_needs_signature(type_hint):
    assert _signature_cannot_be_inferred_from_type_hint(type_hint) is True


def test_infer_schema_from_type_hints_errors():
    with pytest.raises(MlflowException, match=r"Type hints must be wrapped in list\[...\]"):
        _infer_schema_from_list_type_hint(str)

    with pytest.raises(
        MlflowException, match=r"Type hint `list` doesn't contain a collection element type"
    ):
        _infer_schema_from_list_type_hint(list)

    class InvalidModel(pydantic.BaseModel):
        bool_field: bool | None

    if IS_PYDANTIC_V2_OR_NEWER:
        message = (
            r"Optional field `bool_field` in Pydantic model `InvalidModel` "
            r"doesn't have a default value. Please set default value to None for this field."
        )
        with pytest.raises(
            MlflowException,
            match=message,
        ):
            _infer_schema_from_list_type_hint(list[InvalidModel])

        with pytest.raises(MlflowException, match=message):
            _infer_schema_from_list_type_hint(list[list[InvalidModel]])

    message = r"Input cannot be Optional type"
    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(Optional[list[str]])  # noqa: UP045

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[str] | None)

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[Optional[str]])  # noqa: UP045

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[str | None])

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[Union[str, int, type(None)]])  # noqa: UP007

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[str | int | type(None)])

    with pytest.raises(
        MlflowException, match=r"Collections must have only a single type definition"
    ):
        _infer_schema_from_list_type_hint(list[str, int])

    with pytest.raises(MlflowException, match=r"Dictionary key type must be str"):
        _infer_schema_from_list_type_hint(list[dict[int, int]])

    with pytest.raises(
        MlflowException, match=r"Dictionary type hint must contain two element types"
    ):
        _infer_schema_from_list_type_hint(list[dict[int]])

    message = r"it must include a valid element type"
    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[Union])

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[Optional])

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[list])

    with pytest.raises(MlflowException, match=message):
        _infer_schema_from_list_type_hint(list[dict])

    with pytest.raises(UnsupportedTypeHintException, match=r"Unsupported type hint"):
        _infer_schema_from_list_type_hint(list[object])


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
            CustomModel,
            CustomModel(
                long_field=1,
                str_field="a",
                bool_field=True,
                double_field=1.0,
                datetime_field=datetime.datetime.now(),
                binary_field=b"abc",
                any_field="a",
            ),
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
    if isinstance(example, dict):
        assert _validate_data_against_type_hint(data=example, type_hint=type_hint) == type_hint(
            **example
        )
    elif isinstance(example, list):
        assert _validate_data_against_type_hint(data=example, type_hint=type_hint) == [
            get_args(type_hint)[0](**item) for item in example
        ]
    else:
        assert _validate_data_against_type_hint(data=example.dict(), type_hint=type_hint) == example


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
        (Union[int, str], 1),  # noqa: UP007
        (Union[int, str], "a"),  # noqa: UP007
        (int | str, 1),
        (int | str, "a"),
        # Union type is inferred as AnyType, so it accepts double here as well
        (Union[int, str], 1.2),  # noqa: UP007
        (int | str, 1.2),
        (list[Any], [1, "a"]),
    ],
)
def test_python_type_hints_validation(type_hint, example):
    assert _validate_data_against_type_hint(data=example, type_hint=type_hint) == example


def test_type_hints_validation_errors():
    with pytest.raises(MlflowException, match=r"Data doesn't match type hint"):
        _validate_data_against_type_hint({"long_field": 1, "str_field": "a"}, CustomModel)

    with pytest.raises(MlflowException, match=r"Expected type int, but got str"):
        _validate_data_against_type_hint("a", int)

    with pytest.raises(MlflowException, match=r"Expected list, but got str"):
        _validate_data_against_type_hint("a", list[str])

    with pytest.raises(
        MlflowException,
        match=r"Failed to validate data against type hint `list\[str\]`",
    ):
        _validate_data_against_type_hint(["a", 1], list[str])

    with pytest.raises(
        MlflowException,
        match=r"Expected dict, but got list",
    ):
        _validate_data_against_type_hint(["a", 1], dict[str, int])

    with pytest.raises(
        MlflowException,
        match=r"Failed to validate data against type hint `dict\[str, list\[str\]\]`",
    ):
        _validate_data_against_type_hint({1: ["a", "b"], "a": 1}, dict[str, list[str]])

    with pytest.raises(
        MlflowException,
        match=r"Expected type int, but got str",
    ):
        _validate_data_against_type_hint("a", int | None)

    with pytest.raises(
        InvalidTypeHintException,
        match=r"Invalid type hint `list`, it must include a valid element type.",
    ):
        _validate_data_against_type_hint(["a"], list)


@pytest.mark.parametrize(
    ("data", "type_hint", "expected_data"),
    [
        ("a", str, "a"),
        (["a", "b"], list[str], ["a", "b"]),
        ({"a": 1, "b": 2}, dict[str, int], {"a": 1, "b": 2}),
        (1, Optional[int], 1),  # noqa: UP045
        (1, int | None, 1),
        (None, Optional[int], None),  # noqa: UP045
        (None, int | None, None),
        (pd.DataFrame({"a": ["a", "b"]}), list[str], ["a", "b"]),
        (pd.DataFrame({"a": [{"x": "x"}]}), list[dict[str, str]], [{"x": "x"}]),
        # This is a temp workaround for evaluate
        (pd.DataFrame({"a": ["x", "y"], "b": ["c", "d"]}), list[str], ["x", "y"]),
    ],
)
def test_maybe_convert_data_for_type_hint(data, type_hint, expected_data):
    if isinstance(expected_data, pd.DataFrame):
        pd.testing.assert_frame_equal(_convert_data_to_type_hint(data, type_hint), expected_data)
    else:
        assert _convert_data_to_type_hint(data, type_hint) == expected_data


def test_maybe_convert_data_for_type_hint_errors():
    with mock.patch("mlflow.types.type_hints._logger.warning") as mock_warning:
        _convert_data_to_type_hint(pd.DataFrame({"a": ["x", "y"], "b": ["c", "d"]}), list[str])
        assert mock_warning.call_count == 1
        assert (
            "The data will be converted to a list of the first column."
            in mock_warning.call_args[0][0]
        )

    with pytest.raises(
        MlflowException,
        match=r"Only `list\[\.\.\.\]` type hint supports pandas DataFrame input",
    ):
        _convert_data_to_type_hint(pd.DataFrame([["a", "b"]]), str)


def test_is_example_valid_for_type_from_example():
    for data in [
        pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]}),
        pd.Series([1, 2, 3]),
        ["a", "b", "c"],
        [1, 2, 3],
    ]:
        assert _is_example_valid_for_type_from_example(data) is True

    for data in [
        "abc",
        123,
        None,
        {"a": 1},
        {"a": ["x", "y"]},
    ]:
        assert _is_example_valid_for_type_from_example(data) is False


@pytest.mark.parametrize(
    "data",
    [
        # list[scalar]
        ["x", "y", "z"],
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        [True, False, True],
        [b"Hello", b"World"],
        # list[dict]
        [{"a": 1, "b": 2}],
        [{"role": "user", "content": "hello"}, {"role": "admin", "content": "hi"}],
        # pd Series
        pd.Series([1, 2, 3]),
        # pd DataFrame
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
    ],
)
def test_convert_dataframe_to_example_format(data):
    schema = _infer_schema(data)
    df = _enforce_schema(data, schema)
    converted_data = _convert_dataframe_to_example_format(df, data)
    if isinstance(data, pd.Series):
        pd.testing.assert_series_equal(converted_data, data)
    elif isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(converted_data, data)
    else:
        assert converted_data == data

import datetime
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union, get_args
from unittest import mock

import numpy as np
import pandas as pd
import pydantic
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from mlflow.exceptions import MlflowException
from mlflow.models.signature import _infer_signature_from_type_hints, _TypeHints
from mlflow.models.utils import PyFuncOutput, _enforce_schema
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


def test_import_responses_suppresses_expected_schema_inference_warnings():
    code = """
import logging

logging.basicConfig(level=logging.WARNING)

import mlflow.types.responses
from mlflow.types.type_hints import _infer_schema_from_type_hint

_infer_schema_from_type_hint(list[int | str])
"""
    env = os.environ.copy()
    env.pop("MLFLOW_LOGGING_CONFIGURE_LOGGING", None)
    env["MLFLOW_CONFIGURE_LOGGING"] = "0"

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    warning = (
        "Union type hint is inferred as AnyType, and MLflow doesn't validate the data against "
        "its element types."
    )
    assert result.stderr.count(warning) == 1
    assert "Union type hint with multiple non-None types" not in result.stderr


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
            Schema([
                ColSpec(
                    type=Object([
                        Property(name="long_field", dtype=DataType.long),
                        Property(name="str_field", dtype=DataType.string),
                        Property(name="bool_field", dtype=DataType.boolean),
                        Property(name="double_field", dtype=DataType.double),
                        Property(name="binary_field", dtype=DataType.binary),
                        Property(name="datetime_field", dtype=DataType.datetime),
                        Property(name="any_field", dtype=AnyType()),
                        Property(name="optional_str", dtype=DataType.string, required=False),
                    ])
                )
            ]),
        ),
        (
            list[list[CustomModel]],
            Schema([
                ColSpec(
                    type=Array(
                        Object([
                            Property(name="long_field", dtype=DataType.long),
                            Property(name="str_field", dtype=DataType.string),
                            Property(name="bool_field", dtype=DataType.boolean),
                            Property(name="double_field", dtype=DataType.double),
                            Property(name="binary_field", dtype=DataType.binary),
                            Property(name="datetime_field", dtype=DataType.datetime),
                            Property(name="any_field", dtype=AnyType()),
                            Property(name="optional_str", dtype=DataType.string, required=False),
                        ])
                    )
                )
            ]),
        ),
        (
            list[CustomModel2],
            Schema([
                ColSpec(
                    type=Object([
                        Property(name="custom_field", dtype=Map(AnyType())),
                        Property(
                            name="messages",
                            dtype=Array(
                                Object([
                                    Property(name="role", dtype=DataType.string),
                                    Property(name="content", dtype=DataType.string),
                                ])
                            ),
                        ),
                        Property(name="optional_int", dtype=DataType.long, required=False),
                    ])
                )
            ]),
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
        assert (
            _validate_data_against_type_hint(data=example.model_dump(), type_hint=type_hint)
            == example
        )


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


def _assert_no_numpy_scalars(value):
    if isinstance(value, list):
        for item in value:
            _assert_no_numpy_scalars(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_no_numpy_scalars(item)
    else:
        assert not isinstance(value, np.generic)


@pytest.mark.parametrize(
    ("type_hint", "data", "expected"),
    [
        (int, np.int64(1), 1),
        (int, np.uint8(2), 2),
        (float, np.float32(1.5), 1.5),
        (float, np.float16(0.5), 0.5),
        (bool, np.bool_(True), True),
        (int | None, np.int32(3), 3),
        (list[int], [np.int64(1), 2], [1, 2]),
        (list[float], [np.float32(0.5), 1.5], [0.5, 1.5]),
        (dict[str, int], {"a": np.int64(1)}, {"a": 1}),
        (list[dict[str, float]], [{"a": np.float32(0.5)}], [{"a": 0.5}]),
    ],
)
def test_type_hints_validation_accepts_numpy_scalars(type_hint, data, expected):
    result = _validate_data_against_type_hint(data=data, type_hint=type_hint)
    assert result == expected
    _assert_no_numpy_scalars(result)


def test_type_hints_validation_preserves_numpy_float64():
    # np.float64 subclasses float, so it already passes the isinstance check and
    # must keep flowing through unchanged rather than being re-wrapped
    data = np.float64(2.5)
    assert _validate_data_against_type_hint(data=data, type_hint=float) is data


def test_type_hints_validation_rejects_mismatched_numpy_scalars():
    with pytest.raises(MlflowException, match=r"Expected type int, but got float32"):
        _validate_data_against_type_hint(np.float32(1.5), int)

    with pytest.raises(MlflowException, match=r"Expected type bool, but got int64"):
        _validate_data_against_type_hint(np.int64(1), bool)

    with pytest.raises(MlflowException, match=r"Expected type str, but got int64"):
        _validate_data_against_type_hint(np.int64(1), str)

    with pytest.raises(
        MlflowException,
        match=r"Failed to validate data against type hint `list\[int\]`, "
        r"invalid elements: \[\('a', 'Expected type int, but got str'\)\]",
    ):
        # np.int64(1) is normalized and passes validation; only "a" is reported
        _validate_data_against_type_hint([np.int64(1), "a"], list[int])


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
        (
            pd.DataFrame({"a": [1, 3], "b": [2, 4]}),
            list[dict[str, int]],
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        ),
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


def test_dict_in_pyfunc_output():
    """
    Ensure dict is in PyFuncOutput union.

    ResponsesAgent, ChatAgent, and ChatModel all return dict[str, Any]
    at runtime via their pyfunc wrappers. PyFuncOutput must include dict
    to accurately reflect this behavior.
    """
    output_types = get_args(PyFuncOutput)
    # Check if dict is in the union (it could be dict or dict[str, Any])
    has_dict = any(
        t is dict or (hasattr(t, "__origin__") and t.__origin__ is dict) for t in output_types
    )
    assert has_dict, (
        f"dict must be in PyFuncOutput for ResponsesAgent/ChatAgent/ChatModel. "
        f"Current types: {output_types}"
    )


def test_infer_signature_from_type_hints_reports_the_underlying_error():
    with pytest.raises(
        MlflowException,
        match=r"Error: Type hint `list` doesn.t contain a collection element type",
    ):
        _infer_signature_from_type_hints(
            python_model=lambda x: x, context=None, type_hints=_TypeHints(input_=list)
        )

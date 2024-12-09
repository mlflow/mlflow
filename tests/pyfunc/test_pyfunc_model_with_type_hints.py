import datetime
import sys
from typing import Any, Dict, List, Optional, Union
from unittest import mock

import pydantic
import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.signature import _extract_type_hints
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema


class CustomExample(pydantic.BaseModel):
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


class CustomExample2(pydantic.BaseModel):
    custom_field: dict[str, Any]
    messages: list[Message]
    optional_int: Optional[int] = None


@pytest.mark.parametrize(
    ("type_hint", "expected_schema", "input_example"),
    [
        # scalars
        (int, Schema([ColSpec(type=DataType.long)]), 123),
        (str, Schema([ColSpec(type=DataType.string)]), "string"),
        (bool, Schema([ColSpec(type=DataType.boolean)]), True),
        (float, Schema([ColSpec(type=DataType.double)]), 1.23),
        (bytes, Schema([ColSpec(type=DataType.binary)]), b"bytes"),
        (datetime.datetime, Schema([ColSpec(type=DataType.datetime)]), datetime.datetime.now()),
        # lists
        (list[str], Schema([ColSpec(type=Array(DataType.string))]), ["a", "b"]),
        (List[str], Schema([ColSpec(type=Array(DataType.string))]), ["a"]),  # noqa: UP006
        (
            list[list[str]],
            Schema([ColSpec(type=Array(Array(DataType.string)))]),
            [["a", "b"], ["c"]],
        ),
        (List[List[str]], Schema([ColSpec(type=Array(Array(DataType.string)))]), [["a"], ["b"]]),  # noqa: UP006
        # dictionaries
        (dict[str, int], Schema([ColSpec(type=Map(DataType.long))]), {"a": 1}),
        (Dict[str, int], Schema([ColSpec(type=Map(DataType.long))]), {"a": 1, "b": 2}),  # noqa: UP006
        (dict[str, list[str]], Schema([ColSpec(type=Map(Array(DataType.string)))]), {"a": ["b"]}),
        (
            Dict[str, List[str]],  # noqa: UP006
            Schema([ColSpec(type=Map(Array(DataType.string)))]),
            {"a": ["a", "b"]},
        ),
        # Union
        (Union[int, str], Schema([ColSpec(type=AnyType())]), [1, "a", 234]),
        # Any
        (list[Any], Schema([ColSpec(type=Array(AnyType()))]), [True, "abc", 123]),
        # Pydantic Models
        (
            CustomExample,
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
            {
                "long_field": 123,
                "str_field": "abc",
                "bool_field": True,
                "double_field": 1.23,
                "binary_field": b"bytes",
                "datetime_field": datetime.datetime.now(),
                "any_field": ["any", 123],
            },
        ),
        (
            list[CustomExample],
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
            [
                {
                    "long_field": 123,
                    "str_field": "abc",
                    "bool_field": True,
                    "double_field": 1.23,
                    "binary_field": b"bytes",
                    "datetime_field": datetime.datetime.now(),
                    "any_field": ["any", 123],
                },
                {
                    "long_field": 123,
                    "str_field": "abc",
                    "bool_field": False,
                    "double_field": 1.23,
                    "binary_field": b"bytes",
                    "datetime_field": datetime.datetime.now(),
                    "any_field": 123456,
                    "optional_str": "optional",
                },
            ],
        ),
        (
            CustomExample2,
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
            {
                "custom_field": {"a": 1},
                "messages": [{"role": "admin", "content": "hello"}],
                "optional_int": 123,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("model_type", "has_input_example"),
    # if python_model is callable, input_example should be provided
    [("callable", True), ("python_model", True), ("python_model", False)],
)
def test_pyfunc_model_infer_signature_from_type_hints(
    type_hint, expected_schema, input_example, has_input_example, model_type
):
    kwargs = {}
    if model_type == "callable":

        def predict(model_input: type_hint) -> type_hint:
            return model_input

        kwargs["python_model"] = predict
    elif model_type == "python_model":

        class TestModel(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input: type_hint, params=None) -> type_hint:
                return model_input

        kwargs["python_model"] = TestModel()

    if has_input_example:
        kwargs["input_example"] = input_example
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model("test_model", **kwargs)
    assert model_info.signature.inputs == expected_schema
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict(input_example) == input_example


def test_pyfunc_model_infer_signature_from_type_hints_errors():
    def predict(model_input: int) -> int:
        return model_input

    with pytest.raises(
        MlflowException,
        match=r"Input example is not compatible with the type hint of the `predict` function.",
    ):
        with mlflow.start_run():
            mlflow.pyfunc.log_model("test_model", python_model=predict, input_example="string")

    def predict(model_input: int) -> str:
        return model_input

    output_hints = _extract_type_hints(predict, 0).output
    with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                "test_model", python_model=predict, input_example=123
            )
        mock_warning.assert_called_once_with(
            f"Failed to validate output `123` against type hint `{output_hints}`. "
            "Set the logging level to DEBUG to see the full traceback.",
            exc_info=False,
        )
        assert model_info.signature.inputs == Schema([ColSpec(type=DataType.long)])
        assert model_info.signature.outputs is None


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10 or higher")
def test_pyfunc_model_infer_signature_from_type_hints_for_python_3_10():
    def predict(model_input: int | str) -> int | str:
        return model_input

    with mlflow.start_run():
        model_info1 = mlflow.pyfunc.log_model("test_model", python_model=predict, input_example=123)
        model_info2 = mlflow.pyfunc.log_model(
            "test_model", python_model=predict, input_example="string"
        )

    assert model_info1.signature.inputs == Schema([ColSpec(type=AnyType())])
    assert model_info2.signature.outputs == Schema([ColSpec(type=AnyType())])
    assert model_info1.signature == model_info2.signature

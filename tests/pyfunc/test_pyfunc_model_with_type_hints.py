import datetime
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Union
from unittest import mock

import pydantic
import pytest

import mlflow
from mlflow.models.signature import _extract_type_hints
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema
from mlflow.types.type_hints import PYDANTIC_V1_OR_OLDER, _is_pydantic_type_hint


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
        (list[dict[str, str]], Schema([ColSpec(type=Array(Map(DataType.string)))]), [{"a": "b"}]),
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
                    ),
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
                "optional_str": "optional",
            },
        ),
        (
            CustomExample2,
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
    [
        ("callable", True),
        ("python_model", True),
        ("python_model", False),
        ("python_model_no_context", True),
        ("python_model_no_context", False),
    ],
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
    elif model_type == "python_model_no_context":

        class TestModel(mlflow.pyfunc.PythonModel):
            def predict(self, model_input: type_hint, params=None) -> type_hint:
                return model_input

        kwargs["python_model"] = TestModel()

    if has_input_example:
        kwargs["input_example"] = input_example
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model("test_model", **kwargs)
    assert model_info.signature._is_signature_from_type_hint is True
    assert model_info.signature.inputs == expected_schema
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    if _is_pydantic_type_hint(type_hint):
        if PYDANTIC_V1_OR_OLDER:
            assert pyfunc_model.predict(input_example).dict() == input_example
        else:
            assert pyfunc_model.predict(input_example).model_dump() == input_example
    else:
        assert pyfunc_model.predict(input_example) == input_example


def test_pyfunc_model_infer_signature_from_type_hints_errors():
    def predict(model_input: int) -> int:
        return model_input

    with mlflow.start_run():
        with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model("test_model", python_model=predict, input_example="string")
        assert (
            "Input example is not compatible with the type hint of the `predict` function."
            in mock_warning.call_args[0][0]
        )

    def predict(model_input: int) -> str:
        return model_input

    output_hints = _extract_type_hints(predict, 0).output
    with mlflow.start_run():
        with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
            model_info = mlflow.pyfunc.log_model(
                "test_model", python_model=predict, input_example=123
            )
        assert (
            f"Failed to validate output `123` against type hint `{output_hints}`"
            in mock_warning.call_args[0][0]
        )
        assert model_info.signature.inputs == Schema([ColSpec(type=DataType.long)])
        assert model_info.signature.outputs == Schema([ColSpec(AnyType())])


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
    assert model_info1.signature._is_signature_from_type_hint is True
    assert model_info2.signature._is_signature_from_type_hint is True


def save_model_file_for_code_based_logging(type_hint, tmp_path, model_type, extra_def=""):
    if model_type == "callable":
        model_def = f"""
def predict(model_input: {type_hint}) -> {type_hint}:
    return model_input

set_model(predict)
"""
    elif model_type == "python_model":
        model_def = f"""
class TestModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: {type_hint}, params=None) -> {type_hint}:
        return model_input

set_model(TestModel())
"""
    file_content = f"""
import mlflow
from mlflow.models import set_model

import datetime
import pydantic
from typing import Any, Optional, Union

{extra_def}
{model_def}
"""
    model_path = tmp_path / "model.py"
    model_path.write_text(file_content)
    return {"python_model": model_path}


class TypeHintExample(NamedTuple):
    type_hint: str
    input_example: Any
    extra_def: str = ""


@pytest.mark.parametrize(
    "type_hint_example",
    [
        TypeHintExample("int", 123),
        TypeHintExample("str", "string"),
        TypeHintExample("bool", True),
        TypeHintExample("float", 1.23),
        TypeHintExample("bytes", b"bytes"),
        TypeHintExample("datetime.datetime", datetime.datetime.now()),
        TypeHintExample("Any", "any"),
        TypeHintExample("list[str]", ["a", "b"]),
        TypeHintExample("dict[str, int]", {"a": 1}),
        TypeHintExample("Union[int, str]", 123),
        TypeHintExample(
            "CustomExample2",
            CustomExample2(
                custom_field={"a": 1},
                messages=[Message(role="admin", content="hello")],
                optional_int=123,
            ),
            """
class Message(pydantic.BaseModel):
    role: str
    content: str


class CustomExample2(pydantic.BaseModel):
    custom_field: dict[str, Any]
    messages: list[Message]
    optional_int: Optional[int] = None
""",
        ),
    ],
)
@pytest.mark.parametrize(
    ("model_type", "has_input_example"),
    # if python_model is callable, input_example should be provided
    [("callable", True), ("python_model", True), ("python_model", False)],
)
def test_pyfunc_model_with_type_hints_code_based_logging(
    tmp_path, type_hint_example, model_type, has_input_example
):
    kwargs = save_model_file_for_code_based_logging(
        type_hint_example.type_hint,
        tmp_path,
        model_type,
        type_hint_example.extra_def,
    )
    input_example = type_hint_example.input_example
    if has_input_example:
        kwargs["input_example"] = input_example

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model("test_model", **kwargs)

    assert model_info.signature is not None
    assert model_info.signature._is_signature_from_type_hint is True
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict(input_example) == input_example


def test_functional_python_model_only_input_type_hints():
    def python_model(x: list[str]):
        return x

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=python_model, input_example=["a"]
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=Array(DataType.string))])
    assert model_info.signature.outputs is None


def test_functional_python_model_only_output_type_hints():
    def python_model(x) -> list[str]:
        return x

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=python_model, input_example=["a"]
        )
    assert model_info.signature is None


class CallableObject:
    def __call__(self, x: list[str]) -> list[str]:
        return x


def test_functional_python_model_callable_object():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            "model", python_model=CallableObject(), input_example=["a"]
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=Array(DataType.string))])
    assert model_info.signature.outputs == Schema([ColSpec(type=Array(DataType.string))])
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert loaded_model.predict(["a", "b"]) == ["a", "b"]


def test_invalid_type_hint_in_python_model():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[object], params=None) -> str:
            return model_input[0]

    with mlflow.start_run():
        with pytest.warns(UserWarning, match=r"Unsupported type hint"):
            mlflow.pyfunc.log_model("model", python_model=MyModel())

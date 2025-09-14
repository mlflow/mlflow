import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Union
from unittest import mock

import pandas as pd
import pydantic
import pytest
from packaging.version import Version
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    Row,
    StringType,
    StructField,
    StructType,
)

import mlflow
from mlflow.environment_variables import _MLFLOW_IS_IN_SERVING_ENVIRONMENT
from mlflow.exceptions import MlflowException
from mlflow.models import convert_input_example_to_serving_input
from mlflow.models.signature import _extract_type_hints, infer_signature
from mlflow.pyfunc.model import ChatAgent, ChatModel, _FunctionPythonModel
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON
from mlflow.pyfunc.utils import pyfunc
from mlflow.pyfunc.utils.environment import _simulate_serving_environment
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext
from mlflow.types.llm import ChatMessage, ChatParams
from mlflow.types.schema import AnyType, Array, ColSpec, DataType, Map, Object, Property, Schema
from mlflow.types.type_hints import TypeFromExample
from mlflow.utils.pydantic_utils import model_dump_compat

from tests.helper_functions import pyfunc_serve_and_score_model


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


class CustomExample(pydantic.BaseModel):
    long_field: int
    str_field: str
    bool_field: bool
    double_field: float
    any_field: Any
    optional_str: Optional[str] = None  # noqa: UP045
    str_or_none: str | None = None


class Message(pydantic.BaseModel):
    role: str
    content: str


class CustomExample2(pydantic.BaseModel):
    custom_field: dict[str, Any]
    messages: list[Message]
    optional_int: Optional[int] = None  # noqa: UP045
    int_or_none: int | None = None


@pytest.mark.parametrize(
    ("type_hint", "expected_schema", "input_example"),
    [
        # scalars
        (list[int], Schema([ColSpec(type=DataType.long)]), [123]),
        (list[str], Schema([ColSpec(type=DataType.string)]), ["string"]),
        (list[bool], Schema([ColSpec(type=DataType.boolean)]), [True]),
        (list[float], Schema([ColSpec(type=DataType.double)]), [1.23]),
        (list[bytes], Schema([ColSpec(type=DataType.binary)]), [b"bytes"]),
        (
            list[datetime.datetime],
            Schema([ColSpec(type=DataType.datetime)]),
            [datetime.datetime.now()],
        ),
        # lists
        (list[list[str]], Schema([ColSpec(type=Array(DataType.string))]), [["a", "b"]]),
        (List[List[str]], Schema([ColSpec(type=Array(DataType.string))]), [["a"], ["b"]]),  # noqa: UP006
        (
            list[list[list[str]]],
            Schema([ColSpec(type=Array(Array(DataType.string)))]),
            [[["a", "b"], ["c"]]],
        ),
        (
            List[List[List[str]]],  # noqa: UP006
            Schema([ColSpec(type=Array(Array(DataType.string)))]),
            [[["a"], ["b"]]],
        ),
        (
            list[list[dict[str, str]]],
            Schema([ColSpec(type=Array(Map(DataType.string)))]),
            [[{"a": "b"}]],
        ),
        # dictionaries
        (
            list[dict[str, str]],
            Schema([ColSpec(type=Map(DataType.string))]),
            [{"a": "b"}, {"c": "d"}],
        ),
        (list[dict[str, int]], Schema([ColSpec(type=Map(DataType.long))]), [{"a": 1}, {"a": 2}]),
        (list[Dict[str, int]], Schema([ColSpec(type=Map(DataType.long))]), [{"a": 1, "b": 2}]),  # noqa: UP006
        (
            list[dict[str, list[str]]],
            Schema([ColSpec(type=Map(Array(DataType.string)))]),
            [{"a": ["b"]}],
        ),
        (
            List[Dict[str, List[str]]],  # noqa: UP006
            Schema([ColSpec(type=Map(Array(DataType.string)))]),
            [{"a": ["a", "b"]}],
        ),
        # Union
        (list[Union[int, str]], Schema([ColSpec(type=AnyType())]), [1, "a", 234]),  # noqa: UP007
        (list[int | str], Schema([ColSpec(type=AnyType())]), [1, "a", 234]),
        # Any
        (list[Any], Schema([ColSpec(type=AnyType())]), [1, "a", 234]),
        (list[list[Any]], Schema([ColSpec(type=Array(AnyType()))]), [[True], ["abc"], [123]]),
        # Pydantic Models
        (
            list[CustomExample],
            Schema(
                [
                    ColSpec(
                        type=Object(
                            [
                                Property(name="long_field", dtype=DataType.long),
                                Property(name="str_field", dtype=DataType.string),
                                Property(name="bool_field", dtype=DataType.boolean),
                                Property(name="double_field", dtype=DataType.double),
                                Property(name="any_field", dtype=AnyType()),
                                Property(
                                    name="optional_str", dtype=DataType.string, required=False
                                ),
                                Property(name="str_or_none", dtype=DataType.string, required=False),
                            ]
                        )
                    ),
                ]
            ),
            [
                {
                    "long_field": 123,
                    "str_field": "abc",
                    "bool_field": True,
                    "double_field": 1.23,
                    "any_field": ["any", 123],
                    "optional_str": "optional",
                    "str_or_none": "str_or_none",
                }
            ],
        ),
        (
            list[CustomExample2],
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
                                Property(name="int_or_none", dtype=DataType.long, required=False),
                            ]
                        )
                    )
                ]
            ),
            [
                {
                    "custom_field": {"a": 1},
                    "messages": [{"role": "admin", "content": "hello"}],
                    "optional_int": 123,
                    "int_or_none": 456,
                }
            ],
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
        with mock.patch("mlflow.models.model._logger.warning") as mock_warning:
            model_info = mlflow.pyfunc.log_model(name="test_model", **kwargs)
        assert not any(
            "Failed to validate serving input example" in call[0][0]
            for call in mock_warning.call_args_list
        )
    assert model_info.signature._is_signature_from_type_hint is True
    assert model_info.signature.inputs == expected_schema
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = pyfunc_model.predict(input_example)
    if isinstance(result[0], pydantic.BaseModel):
        result = [model_dump_compat(r) for r in result]
    assert result == input_example

    # test serving
    payload = convert_input_example_to_serving_input(input_example)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=payload,
        content_type=CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200


class CustomExample3(pydantic.BaseModel):
    custom_field: dict[str, list[str]]
    messages: list[Message]
    optional_int: Optional[int] = None  # noqa: UP045
    int_or_none: int | None = None


@pytest.mark.parametrize(
    ("type_hint", "result_type", "input_example"),
    [
        # scalars
        # bytes and datetime are not supported in spark_udf
        (list[int], None, [1, 2, 3]),
        (list[str], None, ["a", "b", "c"]),
        (list[bool], None, [True, False, True]),
        (list[float], None, [1.23, 2.34, 3.45]),
        # lists
        (list[list[str]], ArrayType(StringType()), [["a", "b"], ["c", "d"]]),
        # dictionaries
        (
            list[dict[str, str]],
            MapType(StringType(), StringType()),
            [{"a": "b"}, {"c": "d"}],
        ),
        (
            list[dict[str, list[str]]],
            MapType(StringType(), ArrayType(StringType())),
            [{"a": ["b"]}, {"c": ["d"]}],
        ),
        # Union type is not supported because fields in the same column of spark DataFrame
        # must be of same type
        # Any type is not supported yet
        # (list[Any], Schema([ColSpec(type=AnyType())]), ["a", "b", "c"]),
        # Pydantic Models
        (
            list[CustomExample3],
            StructType(
                [
                    StructField("custom_field", MapType(StringType(), ArrayType(StringType()))),
                    StructField(
                        "messages",
                        ArrayType(
                            StructType(
                                [
                                    StructField("role", StringType(), False),
                                    StructField("content", StringType(), False),
                                ]
                            )
                        ),
                    ),
                    StructField("optional_int", IntegerType()),
                    StructField("int_or_none", IntegerType()),
                ]
            ),
            [
                {
                    "custom_field": {"a": ["a", "b", "c"]},
                    "messages": [
                        {"role": "admin", "content": "hello"},
                        {"role": "user", "content": "hi"},
                    ],
                    "optional_int": 123,
                    "int_or_none": 456,
                },
                {
                    "custom_field": {"a": ["a", "b", "c"]},
                    "messages": [
                        {"role": "admin", "content": "hello"},
                    ],
                    "optional_int": None,
                    "int_or_none": None,
                },
            ],
        ),
    ],
)
def test_spark_udf(spark, type_hint, result_type, input_example):
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: type_hint) -> type_hint:
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=Model(), input_example=input_example
        )

    # test spark_udf
    udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, result_type=result_type)
    # the spark dataframe must put the input data in a single column
    if result_type is None:
        # rely on spark to auto-infer schema
        df = spark.createDataFrame(pd.DataFrame({"input": input_example}))
    else:
        schema = StructType([StructField("input", result_type)])
        df = spark.createDataFrame(pd.DataFrame({"input": input_example}), schema=schema)
    df = df.withColumn("response", udf("input"))
    pdf = df.toPandas()
    assert [
        x.asDict(recursive=True) if isinstance(x, Row) else x for x in pdf["response"].tolist()
    ] == input_example


def test_pyfunc_model_with_no_op_type_hint_pass_signature_works():
    def predict(model_input: pd.DataFrame) -> pd.DataFrame:
        return model_input

    input_example = pd.DataFrame({"a": [1]})
    signature = infer_signature(input_example, predict(input_example))
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=predict,
            input_example=input_example,
            signature=signature,
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.long, name="a")])
    pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    pd.testing.assert_frame_equal(pyfunc.predict(input_example), input_example)

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="test_model",
            python_model=Model(),
            input_example=input_example,
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.long, name="a")])
    pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    pd.testing.assert_frame_equal(pyfunc.predict(input_example), input_example)


def test_pyfunc_model_infer_signature_from_type_hints_errors(recwarn):
    def predict(model_input: list[int]) -> int:
        return model_input

    with mlflow.start_run():
        with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(
                name="test_model", python_model=predict, input_example=["string"]
            )
        assert (
            "Input example is not compatible with the type hint of the `predict` function."
            in mock_warning.call_args[0][0]
        )

    def predict(model_input: list[int]) -> str:
        return model_input

    output_hints = _extract_type_hints(predict, 0).output
    with mlflow.start_run():
        with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
            model_info = mlflow.pyfunc.log_model(
                name="test_model", python_model=predict, input_example=[123]
            )
        assert (
            f"Failed to validate output `[123]` against type hint `{output_hints}`"
            in mock_warning.call_args[0][0]
        )
        assert model_info.signature.inputs == Schema([ColSpec(type=DataType.long)])
        assert model_info.signature.outputs == Schema([ColSpec(AnyType())])

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
            return model_input

    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(name="test_model", python_model=Model())
        assert (
            "cannot be used to infer model signature and input example is not provided, "
            "model signature cannot be inferred."
        ) in mock_warning.call_args[0][0]

    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(
                name="test_model", python_model=Model(), input_example=pd.DataFrame()
            )
        assert "Failed to infer model signature from input example" in mock_warning.call_args[0][0]


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
        TypeHintExample("list[int]", [123]),
        TypeHintExample("list[str]", ["string"]),
        TypeHintExample("list[bool]", [True]),
        TypeHintExample("list[float]", [1.23]),
        TypeHintExample("list[bytes]", [b"bytes"]),
        TypeHintExample("list[datetime.datetime]", [datetime.datetime.now()]),
        TypeHintExample("list[Any]", ["any"]),
        TypeHintExample("list[list[str]]", [["a"], ["b"]]),
        TypeHintExample("list[dict[str, int]]", [{"a": 1}]),
        TypeHintExample("list[Union[int, str]]", [123, "abc"]),
        TypeHintExample("list[int | str]", [123, "abc"]),
        TypeHintExample(
            "list[CustomExample2]",
            [
                CustomExample2(
                    custom_field={"a": 1},
                    messages=[Message(role="admin", content="hello")],
                    optional_int=123,
                )
            ],
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
        model_info = mlflow.pyfunc.log_model(name="test_model", **kwargs)

    assert model_info.signature is not None
    assert model_info.signature._is_signature_from_type_hint is True
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict(input_example) == input_example


def test_functional_python_model_only_input_type_hints():
    def python_model(x: list[str]):
        return x

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=python_model, input_example=["a"]
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.string)])
    assert model_info.signature.outputs == Schema([ColSpec(AnyType())])


def test_functional_python_model_only_output_type_hints():
    def python_model(x) -> list[str]:
        return x

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=python_model, input_example=["a"]
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.string)])
    assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string, name=0)])


class CallableObject:
    def __call__(self, x: list[str]) -> list[str]:
        return x


def test_functional_python_model_callable_object():
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=CallableObject(), input_example=["a"]
        )
    assert model_info.signature.inputs == Schema([ColSpec(type=DataType.string)])
    assert model_info.signature.outputs == Schema([ColSpec(type=DataType.string)])
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert loaded_model.predict(["a", "b"]) == ["a", "b"]


def test_python_model_local_testing():
    class ModelWOTypeHint(mlflow.pyfunc.PythonModel):
        def predict(self, model_input, params=None) -> list[str]:
            return model_input

    class ModelWithTypeHint(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[dict[str, str]], params=None) -> list[str]:
            return [m["x"] for m in model_input]

    model1 = ModelWOTypeHint()
    assert model1.predict("a") == "a"
    model2 = ModelWithTypeHint()
    assert model2.predict([{"x": "a"}, {"x": "b"}]) == ["a", "b"]
    with pytest.raises(MlflowException, match=r"Expected dict, but got str"):
        model2.predict(["a"])


def test_python_model_with_optional_input_local_testing():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[dict[str, str | None]], params=None) -> Any:
            return [x["key"] if x.get("key") else "default" for x in model_input]

    model = Model()
    assert model.predict([{"key": None}]) == ["default"]
    assert model.predict([{"key": "a"}]) == ["a"]
    with pytest.raises(MlflowException, match=r"Expected list, but got str"):
        model.predict("a")


def test_callable_local_testing():
    @pyfunc
    def predict(model_input: list[str]) -> list[str]:
        return model_input

    assert predict(["a"]) == ["a"]
    with pytest.raises(MlflowException, match=r"Expected list, but got str"):
        predict("a")

    @pyfunc
    def predict(messages: list[Message]) -> dict[str, str]:
        return {m.role: m.content for m in messages}

    assert predict([Message(role="admin", content="hello")]) == {"admin": "hello"}
    assert predict(
        [{"role": "admin", "content": "hello"}, {"role": "user", "content": "hello"}]
    ) == {"admin": "hello", "user": "hello"}
    pdf = pd.DataFrame([[{"role": "admin", "content": "hello"}]])
    assert predict(pdf) == {"admin": "hello"}

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=predict,
            input_example=[{"role": "admin", "content": "hello"}],
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict([Message(role="admin", content="hello")]) == {"admin": "hello"}
    assert pyfunc_model.predict(
        [{"role": "admin", "content": "hello"}, {"role": "user", "content": "hello"}]
    ) == {"admin": "hello", "user": "hello"}
    assert pyfunc_model.predict(pdf) == {"admin": "hello"}

    # without decorator
    def predict(messages: list[Message]) -> dict[str, str]:
        return {m.role: m.content for m in messages}

    with pytest.raises(AttributeError, match=r"'dict' object has no attribute 'role'"):
        predict([{"role": "admin", "content": "hello"}])

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=predict,
            input_example=[{"role": "admin", "content": "hello"}],
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict([{"role": "admin", "content": "hello"}]) == {"admin": "hello"}


def test_no_warning_for_unsupported_type_hint_with_decorator(recwarn):
    warn_msg = "Type hint used in the model's predict function is not supported"

    @pyfunc
    def predict(model_input: pd.DataFrame) -> pd.DataFrame:
        return model_input

    data = pd.DataFrame({"a": [1]})
    predict(data)
    assert not any(warn_msg in str(w.message) for w in recwarn)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=predict, input_example=data)
    assert not any(warn_msg in str(w.message) for w in recwarn)

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
            return model_input

    model = Model()
    model.predict(data)
    assert not any(warn_msg in str(w.message) for w in recwarn)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=model, input_example=data)
    assert not any(warn_msg in str(w.message) for w in recwarn)


def test_python_model_local_testing_data_validation():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[Message], params=None) -> dict[str, str]:
            return {m.role: m.content for m in model_input}

    model = Model()
    assert model.predict([Message(role="admin", content="hello")]) == {"admin": "hello"}
    assert model.predict(
        [{"role": "admin", "content": "hello"}, {"role": "user", "content": "hello"}]
    ) == {"admin": "hello", "user": "hello"}
    pdf = pd.DataFrame([[{"role": "admin", "content": "hello"}]])
    assert model.predict(pdf) == {"admin": "hello"}

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model", python_model=model, input_example=[{"role": "admin", "content": "hello"}]
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert pyfunc_model.predict([Message(role="admin", content="hello")]) == {"admin": "hello"}
    assert pyfunc_model.predict(
        [{"role": "admin", "content": "hello"}, {"role": "user", "content": "hello"}]
    ) == {"admin": "hello", "user": "hello"}
    assert pyfunc_model.predict(pdf) == {"admin": "hello"}


def test_python_model_local_testing_same_as_pyfunc_predict():
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: list[str], params=None) -> list[str]:
            return model_input

    model = MyModel()
    with pytest.raises(MlflowException, match=r"Expected list, but got str") as e_local:
        model.predict(None, "a")

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(name="model", python_model=model, input_example=["a"])
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with pytest.raises(MlflowException, match=r"Expected list, but got str") as e_pyfunc:
        pyfunc_model.predict("a")

    assert e_local.value.message == e_pyfunc.value.message


def test_unsupported_type_hint_in_python_model(recwarn):
    invalid_type_hint_msg = "Type hint used in the model's predict function is not supported"

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[object], params=None) -> str:
            if isinstance(model_input, list):
                return model_input[0]
            return "abc"

    assert any(invalid_type_hint_msg in str(w.message) for w in recwarn)
    recwarn.clear()

    model = MyModel()
    assert model.predict(["a"]) == "a"
    assert not any(invalid_type_hint_msg in str(w.message) for w in recwarn)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=MyModel())
    assert any("Unsupported type hint" in str(w.message) for w in recwarn)


def test_unsupported_type_hint_in_callable(recwarn):
    @pyfunc
    def predict(model_input: list[object]) -> str:
        if isinstance(model_input, list):
            return model_input[0]
        return "abc"

    invalid_type_hint_msg = "Type hint used in the model's predict function is not supported"
    assert any(invalid_type_hint_msg in str(w.message) for w in recwarn)
    recwarn.clear()
    # The warning should not be raised again when the function is called
    assert predict(["a"]) == "a"
    assert not any(invalid_type_hint_msg in str(w.message) for w in recwarn)

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=predict, input_example=["a"])
    assert any("Unsupported type hint" in str(w.message) for w in recwarn)
    recwarn.clear()

    # without decorator
    def predict(model_input: list[object]) -> str:
        if isinstance(model_input, list):
            return model_input[0]
        return "abc"

    assert predict(["a"]) == "a"
    assert not any(invalid_type_hint_msg in str(w.message) for w in recwarn)

    with mlflow.start_run():
        with pytest.warns(UserWarning, match=r"Unsupported type hint"):
            mlflow.pyfunc.log_model(name="model", python_model=predict, input_example=["a"])


def test_log_model_warn_only_if_model_with_valid_type_hint_not_decorated(recwarn):
    def predict(model_input: list[str]) -> list[str]:
        return model_input

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=predict, input_example=["a"])
        assert any("Decorate your function" in str(w.message) for w in recwarn)
        recwarn.clear()

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[str], params=None) -> list[str]:
            return model_input

    def predict_df(model_input: pd.DataFrame) -> pd.DataFrame:
        return model_input

    with mlflow.start_run():
        mlflow.pyfunc.log_model(name="model", python_model=Model(), input_example=["a"])
    assert not any("Decorate your function" in str(w.message) for w in recwarn)
    recwarn.clear()
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            name="model", python_model=predict_df, input_example=pd.DataFrame({"a": [1]})
        )
    assert not any("Decorate your function" in str(w.message) for w in recwarn)


def test_serving_environment(monkeypatch):
    with _simulate_serving_environment():
        assert os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name] == "true"
    assert os.environ.get(_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name) is None

    monkeypatch.setenv(_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name, "false")
    with _simulate_serving_environment():
        assert os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name] == "true"
    assert os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name] == "false"


def test_predict_model_with_type_hints():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[str]) -> list[str]:
            return model_input

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TestModel(),
        )

    mlflow.models.predict(
        model_uri=model_info.model_uri,
        input_data=["a", "b", "c"],
        env_manager="uv",
    )


def test_predict_with_wrong_signature_warns():
    message = r"Model's `predict` method contains invalid parameters"
    with pytest.warns(FutureWarning, match=message):

        class ModelWOTypeHint(mlflow.pyfunc.PythonModel):
            def predict(self, context, messages, params=None):
                return messages

    with pytest.warns(FutureWarning, match=message):

        class ModelWithTypeHint(mlflow.pyfunc.PythonModel):
            def predict(self, messages: list[str], params=None) -> list[str]:
                return messages

    # applying @pyfunc on the callable should trigger the warning
    with pytest.warns(FutureWarning, match=message):

        @pyfunc
        def predict(messages: list[str]) -> list[str]:
            return messages

    with pytest.warns(FutureWarning, match=message):

        @pyfunc
        def predict(messages):
            return message

    # no @pyfunc decorator, then logging it should trigger the warning
    def predict(messages) -> list[str]:
        return messages

    with mlflow.start_run():
        with pytest.warns(FutureWarning, match=message):
            mlflow.pyfunc.log_model(name="model", python_model=predict, input_example=["a"])


def test_model_with_wrong_predict_signature_works():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, messages: list[Message], params=None) -> list[str]:
            return [m.content for m in messages]

    model = Model()
    input_example = [{"role": "admin", "content": "hello"}]
    expected_response = ["hello"]
    assert model.predict(input_example) == expected_response
    assert model.predict(messages=input_example) == expected_response

    @pyfunc
    def predict(messages: list[Message]) -> list[str]:
        return [m.content for m in messages]

    assert predict(input_example) == expected_response
    assert predict(messages=input_example) == expected_response


def test_warning_message_when_logging_model():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            raise ValueError("test")

    # no type hint + invalid input example
    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(name="model", python_model=TestModel(), input_example="abc")
        assert "Failed to infer model signature from input example" in mock_warning.call_args[0][0]

    # invalid type hint + invalid input example
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: list[object], params=None) -> str:
            raise ValueError("test")

    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(name="model", python_model=TestModel(), input_example="abc")
        assert "Failed to infer model signature from input example" in mock_warning.call_args[0][0]

    # type hint that cannot be used to infer model signature + no input example
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: pd.DataFrame, params=None):
            return model_input

    model = TestModel()
    with mlflow.start_run():
        with mock.patch("mlflow.pyfunc._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(name="model", python_model=model)
        assert (
            "Failed to infer model signature: "
            f"Type hint {model.predict_type_hints} cannot be used to infer model signature and "
            "input example is not provided, model signature cannot be inferred."
        ) in mock_warning.call_args[0][0]


def assert_equal(data1, data2):
    if isinstance(data1, pd.DataFrame):
        pd.testing.assert_frame_equal(data1, data2)
    elif isinstance(data1, pd.Series):
        pd.testing.assert_series_equal(data1, data2)
    else:
        assert data1 == data2


def _type_from_example_models():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, model_input: TypeFromExample):
            return model_input

    def predict(model_input: TypeFromExample):
        return model_input

    return [Model(), predict]


@pytest.fixture(params=_type_from_example_models())
def type_from_example_model(request):
    return request.param


@pytest.mark.parametrize(
    "input_example",
    [
        # list[scalar]
        ["x", "y", "z"],
        [1, 2, 3],
        [1.0, 2.0, 3.0],
        [True, False, True],
        # list[dict]
        [{"x": True}],
        [{"a": 1, "b": 2}],
        [{"role": "user", "content": "hello"}, {"role": "admin", "content": "hi"}],
        # pd DataFrame
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
    ],
)
def test_type_hint_from_example(input_example, type_from_example_model):
    if callable(type_from_example_model):
        assert_equal(type_from_example_model(input_example), input_example)
    else:
        assert_equal(type_from_example_model.predict(input_example), input_example)

    with mlflow.start_run():
        with mock.patch("mlflow.models.model._logger.warning") as mock_warning:
            model_info = mlflow.pyfunc.log_model(
                name="model", python_model=type_from_example_model, input_example=input_example
            )
        assert not any(
            "Failed to validate serving input example" in call[0][0]
            for call in mock_warning.call_args_list
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    result = pyfunc_model.predict(input_example)
    assert_equal(result, input_example)

    # test serving
    payload = convert_input_example_to_serving_input(input_example)
    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_info.model_uri,
        data=payload,
        content_type=CONTENT_TYPE_JSON,
        extra_args=["--env-manager", "local"],
    )
    assert scoring_response.status_code == 200
    if isinstance(input_example, pd.DataFrame):
        assert_equal(
            json.loads(scoring_response.content)["predictions"], input_example.to_dict("records")
        )
    else:
        assert_equal(json.loads(scoring_response.content)["predictions"], input_example)


def test_type_hint_from_example_invalid_input(type_from_example_model):
    with mlflow.start_run():
        with mock.patch("mlflow.models.model._logger.warning") as mock_warning:
            model_info = mlflow.pyfunc.log_model(
                name="model", python_model=type_from_example_model, input_example=[1, 2, 3]
            )
        assert not any(
            "Failed to validate serving input example" in call[0][0]
            for call in mock_warning.call_args_list
        )
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    with pytest.raises(MlflowException, match="Failed to enforce schema of data"):
        pyfunc_model.predict(["1", "2", "3"])


@pytest.mark.skipif(
    Version(pydantic.VERSION).major <= 1,
    reason="pydantic v1 has default value None if the field is Optional",
)
def test_invalid_type_hint_raise_exception():
    class Message(pydantic.BaseModel):
        role: str
        # this doesn't include default value
        content: str | None

    with pytest.raises(MlflowException, match="To disable data validation, remove the type hint"):

        class TestModel(mlflow.pyfunc.PythonModel):
            def predict(self, model_input: list[Message], params=None):
                return model_input

    with pytest.raises(MlflowException, match="To disable data validation, remove the type hint"):

        @pyfunc
        def predict(model_input: list[Message]):
            return model_input


def test_python_model_without_type_hint_warning():
    msg = r"Add type hints to the `predict` method"
    with pytest.warns(UserWarning, match=msg):

        class PythonModelWithoutTypeHint(mlflow.pyfunc.PythonModel):
            def predict(self, model_input, params=None):
                return model_input

    with pytest.warns(UserWarning, match=msg):

        @pyfunc
        def predict(model_input):
            return model_input

    def predict(model_input):
        return model_input

    with mlflow.start_run():
        with pytest.warns(UserWarning, match=msg):
            mlflow.pyfunc.log_model(name="model", python_model=predict, input_example="abc")


@mock.patch("mlflow.pyfunc.utils.data_validation.color_warning")
def test_type_hint_warning_not_shown_for_builtin_subclasses(mock_warning):
    # Class outside "mlflow" module should warn
    class PythonModelWithoutTypeHint(mlflow.pyfunc.PythonModel):
        def predict(self, model_input, params=None):
            return model_input

    assert mock_warning.call_count == 1
    assert "Add type hints to the `predict` method" in mock_warning.call_args[0][0]
    mock_warning.reset_mock()

    # Class inside "mlflow" module should not warn
    ChatModel.__init_subclass__()
    assert mock_warning.call_count == 0

    _FunctionPythonModel.__init_subclass__()
    assert mock_warning.call_count == 0

    # Subclass of ChatModel should not warn (exception to the rule)
    class ChatModelSubclass(ChatModel):
        def predict(self, model_input: list[ChatMessage], params: ChatParams | None = None):
            return model_input

    assert mock_warning.call_count == 0

    # Subclass of ChatAgent should not warn as well (valid pydantic type hint)
    class SimpleChatAgent(ChatAgent):
        def predict(
            self,
            messages: list[ChatAgentMessage],
            context: ChatContext | None = None,
            custom_inputs: dict[str, Any] | None = None,
        ) -> ChatAgentResponse:
            pass

    assert mock_warning.call_count == 0

    # Check import does not trigger any warning (from builtin sub-classes)
    # Note: DO NOT USE importlib.reload as classes in the reloaded
    # module are different than original ones, which could cause unintended
    # side effects in other tests.
    subprocess.check_call(
        [
            sys.executable,
            "-W",
            "error::UserWarning:mlflow.pyfunc.model",
            "-c",
            "import mlflow.pyfunc.model",
        ]
    )


def test_load_context_type_hint():
    class MyModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.context_loaded = True

        def predict(self, model_input: list[str], params=None) -> list[str]:
            assert getattr(self, "context_loaded", False), "load_context was not executed"
            return model_input

    input_example = ["Hello", "World"]
    signature = infer_signature(input_example, input_example)

    with mlflow.start_run() as run:
        with mock.patch("mlflow.models.signature._logger.warning") as mock_warning:
            mlflow.pyfunc.log_model(
                name="model",
                python_model=MyModel(),
                input_example=input_example,
                signature=signature,
            )
        assert not any(
            "Failed to run the predict function on input example" in call[0][0]
            for call in mock_warning.call_args_list
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    underlying_model = pyfunc_model._model_impl.python_model
    assert getattr(underlying_model, "context_loaded", False), (
        "load_context was not called as expected."
    )

    new_data = ["New", "Data"]
    prediction = pyfunc_model.predict(new_data)
    assert prediction == new_data

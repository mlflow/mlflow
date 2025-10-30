from typing import Any, Union

import pytest
from pydantic import BaseModel

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges import CategoricalRating
from mlflow.genai.optimize.util import (
    create_metric_from_scorers,
    infer_type_from_value,
    validate_train_data,
)
from mlflow.genai.scorers import scorer


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


@pytest.mark.parametrize(
    ("categorical_value", "expected_score"),
    [
        (CategoricalRating.YES, 1.0),
        (CategoricalRating.NO, 0.0),
    ],
)
def test_create_metric_from_scorers_with_categorical_rating(categorical_value, expected_score):
    @scorer(name="test_scorer")
    def test_scorer(inputs, outputs):
        return Feedback(name="test_scorer", value=categorical_value)

    metric = create_metric_from_scorers([test_scorer])

    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == expected_score


def test_create_metric_from_scorers_with_multiple_categorical_ratings():
    @scorer(name="scorer1")
    def scorer1(inputs, outputs):
        return Feedback(name="scorer1", value=CategoricalRating.YES)

    @scorer(name="scorer2")
    def scorer2(inputs, outputs):
        return Feedback(name="scorer2", value=CategoricalRating.YES)

    metric = create_metric_from_scorers([scorer1, scorer2])

    # Should sum: 1.0 + 1.0 = 2.0
    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == 2.0


@pytest.mark.parametrize(
    ("primitive_value", "expected_score"),
    [
        (5, 5.0),
        (3.14, 3.14),
        (True, 1.0),
        (False, 0.0),
    ],
)
def test_create_metric_from_scorers_with_feedback_primitive_values(primitive_value, expected_score):
    @scorer(name="test_scorer")
    def test_scorer(inputs, outputs):
        return Feedback(name="test_scorer", value=primitive_value)

    metric = create_metric_from_scorers([test_scorer])

    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == expected_score


def test_create_metric_from_scorers_with_mixed_feedback_types():
    @scorer(name="scorer1")
    def scorer1(inputs, outputs):
        return Feedback(name="scorer1", value=10)

    @scorer(name="scorer2")
    def scorer2(inputs, outputs):
        return Feedback(name="scorer2", value=2.5)

    @scorer(name="scorer3")
    def scorer3(inputs, outputs):
        return Feedback(name="scorer3", value=True)

    metric = create_metric_from_scorers([scorer1, scorer2, scorer3])

    # Should sum: 10.0 + 2.5 + 1.0 = 13.5
    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == 13.5


def test_create_metric_from_scorers_with_feedback_and_direct_values():
    @scorer(name="scorer1")
    def scorer1(inputs, outputs):
        return Feedback(name="scorer1", value=5)

    @scorer(name="scorer2")
    def scorer2(inputs, outputs):
        return 3

    metric = create_metric_from_scorers([scorer1, scorer2])

    # Should sum: 5.0 + 3.0 = 8.0
    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == 8.0


def test_create_metric_from_scorers_with_feedback_and_categorical():
    @scorer(name="scorer1")
    def scorer1(inputs, outputs):
        return Feedback(name="scorer1", value=10)

    @scorer(name="scorer2")
    def scorer2(inputs, outputs):
        return Feedback(name="scorer2", value=CategoricalRating.YES)

    metric = create_metric_from_scorers([scorer1, scorer2])

    # Should sum: 10.0 + 1.0 = 11.0
    result = metric({"input": "test"}, {"output": "result"}, {})
    assert result == 11.0


@pytest.mark.parametrize(
    ("train_data", "expected_error"),
    [
        # Empty inputs
        (
            [{"inputs": {}, "outputs": "result"}],
            "Record 0 is missing required 'inputs' field or it is empty",
        ),
        # Missing inputs
        ([{"outputs": "result"}], "Record 0 is missing required 'inputs' field"),
        # Missing both outputs and expectations
        ([{"inputs": {"text": "hello"}}], "Record 0 must have at least one non-empty field"),
        # Both outputs and expectations are None
        (
            [{"inputs": {"text": "hello"}, "outputs": None, "expectations": None}],
            "Record 0 must have at least one non-empty field",
        ),
    ],
)
def test_validate_train_data_errors(train_data, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        validate_train_data(train_data)


@pytest.mark.parametrize(
    "train_data",
    [
        # Valid with outputs
        [{"inputs": {"text": "hello"}, "outputs": "result"}],
        # Valid with expectations
        [{"inputs": {"text": "hello"}, "expectations": {"expected": "result"}}],
        # Multiple valid records
        [
            {"inputs": {"text": "hello"}, "outputs": "result1"},
            {"inputs": {"text": "world"}, "expectations": {"expected": "result2"}},
        ],
        # Falsy but valid values: False as output
        [{"inputs": {"text": "hello"}, "outputs": False}],
    ],
)
def test_validate_train_data_success(train_data):
    validate_train_data(train_data)

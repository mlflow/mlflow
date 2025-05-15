import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.utils.data_validation import validate_inputs_column_format


def _extract_code_example(e: MlflowException) -> str:
    """Extract the code example from the exception message."""
    return e.message.split("```python")[1].split("```")[0]


def test_validate_inputs_column_format_dynamic_args():
    def fn(*args, key: str, **kwargs):
        return "response"

    inputs = pd.Series([{"key": "value"}])

    with pytest.raises(MlflowException, match=r"The `predict_fn` has dynamic") as e:
        validate_inputs_column_format(inputs, fn)

    expected_code_example = """
def predict_fn(param1, param2):
    # Invoke the original predict function with positional arguments
    return fn(param1, param2)

data = [
    {
        "inputs": {
            "param1": "value1",
            "param2": "value2",
        }
    }
]

mlflow.genai.evaluate(predict_fn=predict_fn, data=data, ...)
"""
    assert _extract_code_example(e.value) == expected_code_example


def test_validate_inputs_column_format_string_input():
    inputs = pd.Series(["What is the capital of France?", "What is the capital of Germany?"])

    with pytest.raises(MlflowException, match=r"The 'inputs' column must be a dictionary of") as e:
        validate_inputs_column_format(inputs, predict_fn=None)

    expected_code_example = """
data = [
    {
        "inputs": {
            "query": "What is the capital of France?",
        }
    }
]

mlflow.genai.evaluate(data=data, scorers=...)
"""
    assert _extract_code_example(e.value) == expected_code_example


def test_validate_inputs_column_format_string_input_with_predict_fn():
    def fn(question: str, context: str):
        return "response"

    inputs = pd.Series(["What is the capital of France?", "What is the capital of Germany?"])

    with pytest.raises(MlflowException, match=r"The 'inputs' column must be a dictionary") as e:
        validate_inputs_column_format(inputs, fn)

    expected_code_example = """
data = [
    {
        "inputs": {
            "question": "What is the capital of France?",
        }
    }
]

mlflow.genai.evaluate(predict_fn=fn, data=data, ...)
"""
    assert _extract_code_example(e.value) == expected_code_example


def test_validate_inputs_column_format_non_string_input():
    def fn(question: str):
        return "response"

    inputs = pd.Series([["a", "b", "c"], ["d", "e", "f"]])

    with pytest.raises(MlflowException, match=r"The 'inputs' column must be a dictionary") as e:
        validate_inputs_column_format(inputs, fn)

    expected_code_example = """
data = [
    {
        "inputs": {
            "question": "What is MLflow?",
        }
    }
]

mlflow.genai.evaluate(predict_fn=fn, data=data, ...)
"""
    assert _extract_code_example(e.value) == expected_code_example


def test_validate_inputs_column_format_predict_fn_with_no_arg():
    def fn():
        return "response"

    inputs = pd.Series(["What is the capital of France?", "What is the capital of Germany?"])

    with pytest.raises(MlflowException, match=r"`predict_fn` must accept at least"):
        validate_inputs_column_format(inputs, fn)

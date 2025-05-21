import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.data_validation import check_model_prediction

from tests.tracing.helper import get_traces


def _extract_code_example(e: MlflowException) -> str:
    """Extract the code example from the exception message."""
    return e.message.split("```python")[1].split("```")[0]


@pytest.mark.parametrize(
    ("predict_fn", "sample_input"),
    [
        # Single argument
        (lambda question: None, {"question": "What is the capital of France?"}),
        # Multiple arguments
        (
            lambda question, context: None,
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe.",
            },
        ),
        # Unnamed keyword arguments
        (lambda **kwargs: None, {"question": "What is the capital of France?"}),
        # Mix of named and unnamed keyword arguments
        (
            lambda question, **kwargs: None,
            {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe.",
            },
        ),
        # Non-string value
        (
            lambda messages: None,
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                ],
            },
        ),
    ],
)
def test_check_model_prediction(predict_fn, sample_input):
    check_model_prediction(predict_fn, sample_input)

    # No trace should be logged during the check
    assert len(get_traces()) == 0

    traced_predict_fn = mlflow.trace(predict_fn)
    check_model_prediction(traced_predict_fn, sample_input)

    # A trace should be logged during the check
    assert len(get_traces()) == 0

    # Running the traced function normally should pass and generate a trace
    traced_predict_fn(**sample_input)
    assert len(get_traces()) == 1


def test_check_model_prediction_class_methods():
    class MyClass:
        def predict(self, question: str, context: str):
            return "response"

        @classmethod
        def predict_cls(cls, question: str, context: str):
            return "response"

        @staticmethod
        def predict_static(question: str, context: str):
            return "response"

    sample_input = {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe.",
    }

    check_model_prediction(MyClass().predict, sample_input)
    check_model_prediction(MyClass.predict_cls, sample_input)
    check_model_prediction(MyClass.predict_static, sample_input)

    assert len(get_traces()) == 0

    # Validate traced version
    check_model_prediction(mlflow.trace(MyClass().predict), sample_input)
    check_model_prediction(mlflow.trace(MyClass.predict_cls), sample_input)
    check_model_prediction(mlflow.trace(MyClass.predict_static), sample_input)

    assert len(get_traces()) == 0


def test_check_model_prediction_no_args():
    def fn():
        return "response"

    with pytest.raises(MlflowException, match=r"`predict_fn` must accept at least one argument."):
        check_model_prediction(fn, {"question": "What is the capital of France?"})


def test_check_model_prediction_variable_args():
    """
    If the function has variable positional arguments (*args), it is not supported.
    """

    def fn(*args):
        return "response"

    with pytest.raises(MlflowException, match=r"The `predict_fn` has dynamic") as e:
        check_model_prediction(fn, {"question": "What is the capital of France?"})

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


def test_check_model_prediction_unmatched_keys():
    def fn(role: str, content: str):
        return "response"

    sample_input = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}

    with pytest.raises(
        MlflowException, match=r"The `inputs` column must be a dictionary with"
    ) as e:
        check_model_prediction(fn, sample_input)

    code_example = """
data = [
    {
        "inputs": {
            "role": "value1",
            "content": "value2",
        }
    }
]
"""
    assert _extract_code_example(e.value) == code_example


def test_check_model_prediction_unmatched_keys_with_many_args():
    def fn(param1, param2, param3, param4, param5):
        return "response"

    sample_input = {"question": "What is the capital of France?"}

    with pytest.raises(MlflowException, match=r"The `inputs` column must be a dictionary") as e:
        check_model_prediction(fn, sample_input)

    # The code snippet shouldn't show more than three parameters
    code_example = """
data = [
    {
        "inputs": {
            "param1": "value1",
            "param2": "value2",
            "param3": "value3",
            "...": "...",
        }
    }
]
"""
    assert _extract_code_example(e.value) == code_example


def test_check_model_prediction_unmatched_keys_with_variable_kwargs():
    def fn(question: str, **kwargs):
        return "response"

    sample_input = {"query": "What is the capital of France?"}
    with pytest.raises(MlflowException, match=r"Failed to run the prediction function"):
        check_model_prediction(fn, sample_input)


def test_check_model_prediction_unknown_error():
    def fn(question: str):
        raise ValueError("Unknown error")

    sample_input = {"question": "What is the capital of France?"}
    with pytest.raises(MlflowException, match=r"Failed to run the prediction function"):
        check_model_prediction(fn, sample_input)

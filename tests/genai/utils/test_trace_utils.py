import openai
import pytest

import mlflow
from mlflow.genai.utils.trace_utils import is_model_traced
from mlflow.models import Model
from mlflow.models.evaluation.base import _get_model_from_function
from mlflow.pyfunc import PyFuncModel


def get_openai_predict_fn(with_tracing=False):
    client = openai.OpenAI()

    if with_tracing:
        mlflow.openai.autolog()

    def predict_fn(request):
        response = client.chat.completions.create(
            message=request["messages"],
            model="gpt-4o-mini",
        )
        return response.choices[0].message.content

    return predict_fn


def get_dummy_predict_fn(with_tracing=False):
    def predict_fn(inp):
        return inp * 2

    if with_tracing:
        return mlflow.trace(predict_fn)

    return predict_fn


def get_pyfunc_model(predict_fn):
    return PyFuncModel(
        model_id=None,
        model_meta=Model(),
        model_impl=_get_model_from_function(predict_fn),
    )


@pytest.fixture
def mock_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_api_key")


@pytest.mark.usefixtures("mock_openai_env")
@pytest.mark.parametrize(
    ("test_case", "predict_fn_generator", "with_tracing", "expected_traced"),
    [
        ("dummy predict_fn without tracing", get_dummy_predict_fn, False, False),
        ("dummy predict_fn with tracing", get_dummy_predict_fn, True, True),
        ("openai predict_fn without tracing", get_openai_predict_fn, False, False),
        ("openai predict_fn with tracing", get_openai_predict_fn, True, True),
    ],
)
def test_is_traced(test_case, predict_fn_generator, with_tracing, expected_traced):
    predict_fn = predict_fn_generator(with_tracing=with_tracing)
    model = get_pyfunc_model(predict_fn)
    is_actually_traced = is_model_traced(model)
    assert is_actually_traced == expected_traced, f"Failed for {test_case}"

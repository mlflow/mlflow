from unittest import mock

import httpx
import openai
import pytest

import mlflow
from mlflow.genai.utils.trace_utils import convert_predict_fn

from tests.tracing.helper import get_traces, purge_traces


def httpx_send_patch(request, *args, **kwargs):
    return httpx.Response(
        status_code=200,
        request=request,
        json={
            "id": "chatcmpl-Ax4UAd5xf32KjgLkS1SEEY9oorI9m",
            "object": "chat.completion",
            "created": 1738641958,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "test",
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        },
    )


def get_openai_predict_fn(with_tracing=False):
    if with_tracing:
        mlflow.openai.autolog()

    def predict_fn(request):
        with mock.patch("httpx.Client.send", side_effect=httpx_send_patch):
            response = openai.OpenAI().chat.completions.create(
                messages=request["messages"],
                model="gpt-4o-mini",
            )
            return response.choices[0].message.content

    return predict_fn


def get_dummy_predict_fn(with_tracing=False):
    def predict_fn(request):
        return "test"

    if with_tracing:
        return mlflow.trace(predict_fn)

    return predict_fn


@pytest.fixture
def mock_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_api_key")


@pytest.mark.usefixtures("mock_openai_env")
@pytest.mark.parametrize(
    ("predict_fn_generator", "with_tracing"),
    [
        (get_dummy_predict_fn, False),
        (get_dummy_predict_fn, True),
        (get_openai_predict_fn, False),
        (get_openai_predict_fn, True),
    ],
    ids=[
        "dummy predict_fn without tracing",
        "dummy predict_fn with tracing",
        "openai predict_fn without tracing",
        "openai predict_fn with tracing",
    ],
)
def test_convert_predict_fn(predict_fn_generator, with_tracing):
    predict_fn = predict_fn_generator(with_tracing=with_tracing)
    sample_input = {"request": {"messages": [{"role": "user", "content": "test"}]}}

    # predict_fn is callable as is
    result = predict_fn(**sample_input)
    assert result == "test"
    assert len(get_traces()) == (1 if with_tracing else 0)
    purge_traces()

    converted_fn = convert_predict_fn(predict_fn, sample_input)

    # converted function takes a single 'request' argument
    result = converted_fn(request=sample_input)
    assert result == "test"

    # Trace should be generated
    assert len(get_traces()) == 1

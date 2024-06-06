import openai
import pytest

import mlflow
from mlflow.tracing.constant import SpanAttributeKey

from tests.openai.conftest import is_v1


@pytest.fixture
def client(mock_openai):
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_tracing_success(client, monkeypatch):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run():
        client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=0,
        )

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "Completions"
    assert span.attributes[SpanAttributeKey.INPUTS]["messages"][0]["content"] == "test"
    assert span.attributes[SpanAttributeKey.OUTPUTS]["id"] == "chatcmpl-123"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_tracing_error(client, monkeypatch):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_run(), pytest.raises(
        openai.BadRequestError, match="Temperature must be between 0.0 and 2.0"
    ):
        client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            temperature=5.0,
        )

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "Completions"
    assert span.attributes[SpanAttributeKey.INPUTS]["messages"][0]["content"] == "test"
    assert span.attributes.get(SpanAttributeKey.OUTPUTS) is None

    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "openai.BadRequestError"

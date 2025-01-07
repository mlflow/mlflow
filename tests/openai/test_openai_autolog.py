import json

import openai
import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

from tests.openai.conftest import is_v1
from tests.openai.mock_openai import EMPTY_CHOICES
from tests.tracing.helper import get_traces

MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


@pytest.fixture
def client(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )
    return openai.OpenAI(api_key="test", base_url=mock_openai)


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
@pytest.mark.parametrize("log_models", [True, False])
def test_chat_completions_autolog(client, log_models):
    mlflow.openai.autolog(log_models=log_models)

    messages = [{"role": "user", "content": "test"}]
    client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0,
    )

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.inputs == {"messages": messages, "model": "gpt-4o-mini", "temperature": 0}
    assert span.outputs["id"] == "chatcmpl-123"
    assert span.attributes["model"] == "gpt-4o-mini"
    assert span.attributes["temperature"] == 0

    if log_models:
        run_id = client.chat.completions._mlflow_run_id
        assert run_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
        loaded_model = mlflow.openai.load_model(f"runs:/{run_id}/model")
        assert loaded_model == {
            "model": "gpt-4o-mini",
            "task": "chat.completions",
        }
        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        assert pyfunc_model.predict("test") == [json.dumps(messages)]

    else:
        assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_under_current_active_span(client):
    # If a user have an active span, the autologging should create a child span under it.
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_span(name="parent"):
        client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0,
        )

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 2
    parent_span = trace.data.spans[0]
    assert parent_span.name == "parent"
    child_span = trace.data.spans[1]
    assert child_span.name == "Completions"
    assert child_span.inputs == {"messages": messages, "model": "gpt-4o-mini", "temperature": 0}
    assert child_span.outputs["id"] == "chatcmpl-123"
    assert child_span.parent_id == parent_span.span_id


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_streaming(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0,
        stream=True,
    )
    for _ in stream:
        pass

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.inputs == {
        "messages": messages,
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    assert span.outputs == "Hello world"  # aggregated string of streaming response

    stream_event_data = trace.data.spans[0].attributes["events"]

    assert stream_event_data[0]["id"] == "chatcmpl-123"
    assert stream_event_data[0]["choices"][0]["delta"]["content"] == "Hello"
    assert stream_event_data[1]["id"] == "chatcmpl-123"
    assert stream_event_data[1]["choices"][0]["delta"]["content"] == " world"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_tracing_error(client):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with pytest.raises(openai.BadRequestError, match="Temperature must be between 0.0 and 2.0"):
        client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=5.0,
        )

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "Completions"
    assert span.inputs["messages"][0]["content"] == "test"
    assert span.outputs is None

    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "openai.BadRequestError"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_chat_completions_autolog_tracing_error(client):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with pytest.raises(openai.BadRequestError, match="Temperature must be between 0.0 and 2.0"):
        client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=5.0,
        )

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "Completions"
    assert span.inputs["messages"][0]["content"] == "test"
    assert span.outputs is None

    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "BadRequestError"


def test_chat_completions_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": EMPTY_CHOICES}],
        model="gpt-4o-mini",
        stream=True,
    )

    # Ensure the stream has a chunk with empty choices
    first_chunk = next(stream)
    assert first_chunk.choices == []

    # Exhaust the stream
    for _ in stream:
        pass

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
@pytest.mark.parametrize("log_models", [True, False])
def test_completions_autolog(client, log_models):
    mlflow.openai.autolog(log_models=log_models)

    client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
    )

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.inputs == {"prompt": "test", "model": "gpt-4o-mini", "temperature": 0}
    assert span.outputs["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"

    if log_models:
        run_id = client.completions._mlflow_run_id
        assert run_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
        loaded_model = mlflow.openai.load_model(f"runs:/{run_id}/model")
        assert loaded_model == {
            "model": "gpt-4o-mini",
            "task": "completions",
        }
        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        assert pyfunc_model.predict("test") == ["test"]
    else:
        assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


def test_completions_autolog_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.completions.create(
        prompt=EMPTY_CHOICES,
        model="gpt-4o-mini",
        stream=True,
    )

    # Ensure the stream has a chunk with empty choices
    first_chunk = next(stream)
    assert first_chunk.choices == []

    # Exhaust the stream
    for _ in stream:
        pass

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_completions_autolog_streaming(client):
    mlflow.openai.autolog()

    stream = client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
        stream=True,
    )
    for _ in stream:
        pass

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.inputs == {
        "prompt": "test",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    assert span.outputs == "Hello world"  # aggregated string of streaming response

    stream_event_data = trace.data.spans[0].attributes["events"]

    assert stream_event_data[0]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert stream_event_data[0]["choices"][0]["text"] == "Hello"
    assert stream_event_data[1]["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert stream_event_data[1]["choices"][0]["text"] == " world"


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
@pytest.mark.parametrize("log_models", [True, False])
def test_embeddings_autolog(client, log_models):
    mlflow.openai.autolog(log_models=log_models)

    client.embeddings.create(
        input="test",
        model="text-embedding-ada-002",
    )

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.inputs == {"input": "test", "model": "text-embedding-ada-002"}
    assert span.outputs["data"][0]["embedding"] == list(range(1536))

    if log_models:
        run_id = client.embeddings._mlflow_run_id
        assert run_id is not None
        assert trace.info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_id
        loaded_model = mlflow.openai.load_model(f"runs:/{run_id}/model")
        assert loaded_model == {
            "model": "text-embedding-ada-002",
            "task": "embeddings",
        }
        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        output = pyfunc_model.predict("test")
        assert len(output) == 1
        assert len(output[0]) == 1536
    else:
        assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_autolog_with_registered_model_name(client):
    registered_model_name = "test_model"
    mlflow.openai.autolog(log_models=True, registered_model_name=registered_model_name)
    client.chat.completions.create(
        messages=[{"role": "user", "content": "test"}],
        model="gpt-4o-mini",
        temperature=0,
    )
    registered_model = MlflowClient().get_registered_model(registered_model_name)
    assert registered_model.name == registered_model_name


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
@pytest.mark.parametrize("log_models", [True, False])
def test_autolog_use_active_run_id(client, log_models):
    mlflow.openai.autolog(log_models=log_models)

    messages = [{"role": "user", "content": "test"}]

    with mlflow.start_run() as run_1:
        client.chat.completions.create(messages=messages, model="gpt-4o-mini")

    assert client.chat.completions._mlflow_run_id == run_1.info.run_id

    with mlflow.start_run() as run_2:
        client.chat.completions.create(messages=messages, model="gpt-4o-mini")
        client.chat.completions.create(messages=messages, model="gpt-4o-mini")

    assert client.chat.completions._mlflow_run_id == run_2.info.run_id

    with mlflow.start_run() as run_3:
        mlflow.openai.autolog(
            log_models=log_models,
            extra_tags={"foo": "bar"},
        )
        client.chat.completions.create(messages=messages, model="gpt-4o-mini")

    assert client.chat.completions._mlflow_run_id == run_3.info.run_id

    traces = get_traces()[::-1]  # reverse order to sort by timestamp in ascending order
    assert len(traces) == 4

    assert traces[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_1.info.run_id
    assert traces[1].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[2].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[3].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_3.info.run_id


def test_autolog_raw_response(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    with mlflow.start_run():
        resp = client.chat.completions.with_raw_response.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=MOCK_TOOLS,
        )
        resp = resp.parse()  # ensure the raw response is returned

    assert resp.choices[0].message.content == '[{"role": "user", "content": "test"}]'
    trace = mlflow.get_last_active_trace()
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert isinstance(span.outputs, dict)
    assert (
        span.outputs["choices"][0]["message"]["content"] == '[{"role": "user", "content": "test"}]'
    )
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == (
        messages + [{"role": "assistant", "content": '[{"role": "user", "content": "test"}]'}]
    )
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == MOCK_TOOLS


def test_autolog_raw_response_stream(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    with mlflow.start_run():
        resp = client.chat.completions.with_raw_response.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=MOCK_TOOLS,
            stream=True,
        )
        resp = resp.parse()  # ensure the raw response is returned

    assert [c.choices[0].delta.content for c in resp] == ["Hello", " world"]
    trace = mlflow.get_last_active_trace()
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.outputs == "Hello world"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == (
        messages + [{"role": "assistant", "content": "Hello world"}]
    )
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == MOCK_TOOLS

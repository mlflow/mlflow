import asyncio

import litellm
import pytest

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces


def _assert_usage(
    attributes,
    expected_prompt_tokens=None,
    expected_completion_tokens=None,
    expected_total_tokens=None,
):
    if "mlflow.chat.tokenUsage" in attributes:
        usage = attributes["mlflow.chat.tokenUsage"]
        assert usage["total_tokens"] == expected_total_tokens
        assert usage["input_tokens"] == expected_prompt_tokens
        assert usage["output_tokens"] == expected_completion_tokens
    else:
        usage = attributes["usage"]
        assert usage["total_tokens"] == expected_total_tokens
        assert usage["completion_tokens"] == expected_completion_tokens
        assert usage["prompt_tokens"] == expected_prompt_tokens


@pytest.fixture(autouse=True)
def cleanup_callbacks():
    yield
    litellm.success_callbacks = []
    litellm.failure_callbacks = []


def test_litellm_tracing_success():
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )
    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    spans = traces[0].data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs == response.model_dump()
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    assert spans[0].attributes["call_type"] == "completion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=9,
        expected_completion_tokens=12,
        expected_total_tokens=21,
    )


def test_litellm_tracing_failure():
    mlflow.litellm.autolog()

    with pytest.raises(litellm.exceptions.BadRequestError, match="LLM Provider"):
        litellm.completion(
            model="invalid-model",
            messages=[{"role": "system", "content": "Hello"}],
        )

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "ERROR"
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs is None
    assert spans[0].attributes["model"] == "invalid-model"
    assert spans[0].attributes["response_cost"] == 0
    assert len(spans[0].events) == 1
    assert spans[0].events[0].name == "exception"


def test_litellm_tracing_streaming():
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )

    chunks = [c.choices[0].delta.content for c in response]
    assert chunks == ["Hello", " world", None]

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-completion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].span_type == SpanType.LLM
    assert spans[0].inputs == {
        "messages": [{"role": "system", "content": "Hello"}],
        "stream": True,
    }
    assert spans[0].outputs["choices"][0]["message"]["content"] == "Hello world"
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=8,
        expected_completion_tokens=2,
        expected_total_tokens=10,
    )


@pytest.mark.asyncio
async def test_litellm_tracing_async():
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )
    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    # Adding a sleep here to ensure that trace is logged.
    await asyncio.sleep(0.1)

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs == response.model_dump()
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    assert spans[0].attributes["call_type"] == "acompletion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    _assert_usage(
        spans[0].attributes,
        expected_prompt_tokens=9,
        expected_completion_tokens=12,
        expected_total_tokens=21,
    )


@pytest.mark.asyncio
async def test_litellm_tracing_async_streaming():
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )
    chunks: list[str | None] = []
    async for c in response:
        chunks.append(c.choices[0].delta.content)
        # Adding a sleep here to ensure that `content` in the span outputs is
        # consistently 'Hello World', not 'Hello' or ''.
        await asyncio.sleep(0.1)

    assert chunks == ["Hello", " world", None]

    # Await the logger task to ensure that the trace is logged.
    logger_task = next(
        task
        for task in asyncio.all_tasks()
        if "async_success_handler" in getattr(task.get_coro(), "__name__", "")
    )
    await logger_task

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].outputs["choices"][0]["message"]["content"] == "Hello world"


def test_litellm_tracing_with_parent_span():
    mlflow.litellm.autolog()

    with mlflow.start_span(name="parent"):
        litellm.completion(model="gpt-4o-mini", messages=[{"role": "system", "content": "Hello"}])

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 2
    assert spans[0].name == "parent"
    assert spans[1].name == "litellm-completion"


def test_litellm_tracing_disable():
    mlflow.litellm.autolog()

    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    assert mlflow.get_trace(mlflow.get_last_active_trace_id()) is not None
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(disable=True)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    # no additional trace should be created
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(log_traces=False)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    # no additional trace should be created
    assert len(get_traces()) == 1

import asyncio
import time
from typing import Optional
from unittest import mock

import litellm
import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.utils.databricks_utils import is_in_databricks_runtime

from tests.tracing.helper import get_traces


# fixture to run test twice, one for in databricks and one for not in databricks
@pytest.fixture(params=[True, False])
def is_in_databricks(request):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_runtime", return_value=request.param
    ):
        yield request.param


def _wait_if_not_in_databricks():
    if not is_in_databricks_runtime():
        time.sleep(1)


@pytest.fixture(autouse=True)
def cleanup_callbacks():
    yield
    litellm.success_callbacks = []
    litellm.failure_callbacks = []


def test_litellm_tracing_success(is_in_databricks):
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )

    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    # Success logging is asynchronous by default, but in Databricks, we patch it to be synchronous
    # to ensure that the trace is rendered correctly in notebook.
    _wait_if_not_in_databricks()

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
    assert spans[0].attributes["api_base"].startswith("http://localhost")
    assert spans[0].attributes["call_type"] == "completion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    assert spans[0].attributes["usage"] is not None


def test_litellm_tracing_failure(is_in_databricks):
    mlflow.litellm.autolog()

    with pytest.raises(litellm.exceptions.BadRequestError, match="LLM Provider"):
        litellm.completion(
            model="invalid-model",
            messages=[{"role": "system", "content": "Hello"}],
        )

    trace = mlflow.get_last_active_trace()
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


def test_litellm_tracing_streaming(is_in_databricks):
    mlflow.litellm.autolog()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )

    chunks = [c.choices[0].delta.content for c in response]
    assert chunks == ["Hello", " world", None]

    _wait_if_not_in_databricks()

    trace = mlflow.get_last_active_trace()
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


@pytest.mark.asyncio
async def test_litellm_tracing_async(is_in_databricks):
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
    )
    assert response.choices[0].message.content == '[{"role": "system", "content": "Hello"}]'

    # Await the logger task to ensure that the trace is logged.
    logger_task = next(
        t for t in asyncio.all_tasks() if "async_success_handler" in t.get_coro().__name__
    )
    await logger_task

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"messages": [{"role": "system", "content": "Hello"}]}
    assert spans[0].outputs == response.model_dump()
    assert spans[0].attributes["model"] == "gpt-4o-mini"
    assert spans[0].attributes["api_base"].startswith("http://localhost")
    assert spans[0].attributes["call_type"] == "acompletion"
    assert spans[0].attributes["cache_hit"] is None
    assert spans[0].attributes["response_cost"] > 0
    assert spans[0].attributes["usage"] is not None


@pytest.mark.asyncio
async def test_litellm_tracing_async_streaming(is_in_databricks):
    mlflow.litellm.autolog()

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Hello"}],
        stream=True,
    )
    chunks: list[Optional[str]] = []
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

    trace = mlflow.get_last_active_trace()
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "litellm-acompletion"
    assert spans[0].status.status_code == "OK"
    assert spans[0].outputs["choices"][0]["message"]["content"] == "Hello world"


def test_litellm_tracing_disable(is_in_databricks):
    mlflow.litellm.autolog()

    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    _wait_if_not_in_databricks()
    assert mlflow.get_last_active_trace() is not None
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(disable=True)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    _wait_if_not_in_databricks()
    # no additional trace should be created
    assert len(get_traces()) == 1

    mlflow.litellm.autolog(log_traces=False)
    litellm.completion("gpt-4o-mini", [{"role": "system", "content": "Hello"}])
    _wait_if_not_in_databricks()
    # no additional trace should be created
    assert len(get_traces()) == 1

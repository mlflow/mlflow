import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    TypeSafeBadRequestError,
    TypeSafeClient,
)

import mlflow.typesafe
from mlflow.entities import SpanLogLevel, SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

_RESULT = {
    "model": "jev-resolved",
    "usage": {"input_tokens": 12, "output_tokens": 3},
    "answers": {
        "relevant": {"type": "noul", "noul": 0.98},
        "tone": {
            "type": "choice",
            "choice": "friendly",
            "confidence": 0.9,
            "probabilities": {"friendly": 0.9, "hostile": 0.1},
        },
    },
}

_QUESTIONS = {
    "relevant": Noul(instructions="Is this relevant?"),
    "tone": Choice(
        instructions="What is the tone?",
        criteria={"friendly": None, "hostile": None},
    ),
}

_SERIALIZED_QUESTIONS = {
    "relevant": {"type": "noul", "instructions": "Is this relevant?"},
    "tone": {
        "type": "choice",
        "instructions": "What is the tone?",
        "criteria": {"friendly": None, "hostile": None},
    },
}


@pytest.fixture(autouse=True)
def disable_typesafe_autolog_after_test():
    yield
    mlflow.typesafe.autolog(disable=True)


def _system_one(
    async_mode: bool,
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    client_kwargs: dict[str, Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
):
    client_kwargs = client_kwargs or {}
    call_kwargs = call_kwargs or {}
    transport = httpx2.MockTransport(handler)

    if async_mode:

        async def run():
            async with AsyncTypeSafeClient(
                api_key="typesafe-test-key",
                transport=transport,
                **client_kwargs,
            ) as client:
                return await client.system_one(**call_kwargs)

        return asyncio.run(run())

    with TypeSafeClient(
        api_key="typesafe-test-key",
        transport=transport,
        **client_kwargs,
    ) as client:
        return client.system_one(**call_kwargs)


def _list_models(
    async_mode: bool,
    handler: Callable[[httpx2.Request], httpx2.Response],
):
    transport = httpx2.MockTransport(handler)
    if async_mode:

        async def run():
            async with AsyncTypeSafeClient(
                api_key="typesafe-test-key",
                transport=transport,
            ) as client:
                return await client.models.list()

        return asyncio.run(run())

    with TypeSafeClient(api_key="typesafe-test-key", transport=transport) as client:
        return client.models.list()


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_system_one_autolog(async_mode):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json=_RESULT,
            headers={"x-typesafe-request-id": "req_success"},
        )

    mlflow.typesafe.autolog()
    result = _system_one(
        async_mode,
        handler,
        client_kwargs={"model": "jev-alias", "retry": RetryPolicy(max_retries=0)},
        call_kwargs={
            "state": {"document": "Hello"},
            "questions": _QUESTIONS,
            "extra_body": {"feature": "test"},
        },
    )

    assert result.model == "jev-resolved"
    assert len(requests) == 1
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1

    span = traces[0].data.spans[0]
    assert span.name == "typesafe.system_one"
    assert span.span_type == SpanType.LLM
    assert span.inputs == {
        "state": {"document": "Hello"},
        "model": "jev-alias",
        "questions": _SERIALIZED_QUESTIONS,
        "feature": "test",
    }
    assert span.outputs == _RESULT
    assert span.model_name == "jev-resolved"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "typesafe"
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 12,
        TokenUsageKey.OUTPUT_TOKENS: 3,
        TokenUsageKey.TOTAL_TOKENS: 15,
    }
    assert span.get_attribute("typesafe.request_id") == "req_success"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 12,
        TokenUsageKey.OUTPUT_TOKENS: 3,
        TokenUsageKey.TOTAL_TOKENS: 15,
    }


def test_extra_body_overrides_effective_inputs():
    mlflow.typesafe.autolog()
    _system_one(
        False,
        lambda request: httpx2.Response(200, json=_RESULT),
        client_kwargs={"model": "client-model", "retry": RetryPolicy(max_retries=0)},
        call_kwargs={
            "state": "original state",
            "questions": {"original": Noul(instructions="Original?")},
            "model": "call-model",
            "extra_body": {
                "state": {"overridden": True},
                "model": "body-model",
                "questions": {"body_question": {"type": "noul", "instructions": "From body?"}},
            },
        },
    )

    span = get_traces()[0].data.spans[0]
    assert span.inputs == {
        "state": {"overridden": True},
        "model": "body-model",
        "questions": {"body_question": {"type": "noul", "instructions": "From body?"}},
    }


def test_arbitrary_sequence_input_matches_wire_body():
    wire_bodies = []

    def handler(request):
        wire_bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_RESULT)

    mlflow.typesafe.autolog()
    _system_one(
        False,
        handler,
        client_kwargs={"retry": RetryPolicy(max_retries=0)},
        call_kwargs={
            "state": range(3),
            "questions": {"relevant": Noul(instructions="Relevant?")},
        },
    )

    assert len(wire_bodies) == 1
    assert wire_bodies[0]["state"] == [0, 1, 2]
    assert get_traces()[0].data.spans[0].inputs["state"] == wire_bodies[0]["state"]


class _CustomResponse(BaseModel):
    decision: str


def test_custom_response_model_uses_requested_model_as_fallback():
    mlflow.typesafe.autolog()
    _system_one(
        False,
        lambda request: httpx2.Response(200, json={"decision": "allow"}),
        client_kwargs={"model": "client-model", "retry": RetryPolicy(max_retries=0)},
        call_kwargs={
            "state": "hello",
            "questions": {"relevant": Noul(instructions="Relevant?")},
            "response_model": _CustomResponse,
        },
    )

    span = get_traces()[0].data.spans[0]
    assert span.outputs == {"decision": "allow"}
    assert span.model_name == "client-model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "typesafe"
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) is None


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_system_one_error_is_traced_and_reraised(async_mode):
    error_body = {"message": "invalid request"}

    def handler(request):
        return httpx2.Response(
            400,
            json=error_body,
            headers={"x-typesafe-request-id": "req_error"},
        )

    mlflow.typesafe.autolog()
    with pytest.raises(TypeSafeBadRequestError, match="invalid request"):
        _system_one(
            async_mode,
            handler,
            client_kwargs={"retry": RetryPolicy(max_retries=0)},
            call_kwargs={
                "state": "hello",
                "questions": {"relevant": Noul(instructions="Relevant?")},
            },
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert "TypeSafeBadRequestError" in span.status.description
    assert "invalid request" in span.status.description
    assert span.get_attribute("typesafe.request_id") == "req_error"


def test_async_cancellation_is_traced_as_error():
    async def run():
        request_started = asyncio.Event()
        block_request = asyncio.Event()

        async def handler(request):
            request_started.set()
            await block_request.wait()
            return httpx2.Response(200, json=_RESULT)

        mlflow.typesafe.autolog()
        async with AsyncTypeSafeClient(
            api_key="typesafe-test-key",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(handler),
        ) as client:
            task = asyncio.create_task(
                client.system_one(
                    state="hello",
                    questions={"relevant": Noul(instructions="Relevant?")},
                )
            )
            await request_started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("The System One task was not cancelled")

    asyncio.run(run())

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert "CancelledError" in span.status.description
    assert span.log_level == SpanLogLevel.ERROR


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_retries_produce_one_span(async_mode):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx2.Response(
                429,
                json={"message": "retry"},
                headers={"retry-after-ms": "0"},
            )
        return httpx2.Response(200, json=_RESULT)

    mlflow.typesafe.autolog()
    _system_one(
        async_mode,
        handler,
        client_kwargs={
            "retry": RetryPolicy(
                max_retries=2,
                backoff_initial=0.001,
                backoff_max=0.001,
            )
        },
        call_kwargs={
            "state": "hello",
            "questions": {"relevant": Noul(instructions="Relevant?")},
        },
    )

    assert attempts == 3
    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1
    assert traces[0].info.status == "OK"


def test_sensitive_client_and_call_options_are_not_traced():
    api_key = "secret-api-key"
    default_header_secret = "secret-default-header"
    call_header_secret = "secret-call-header"

    def handler(request):
        assert request.headers["authorization"] == f"Bearer {api_key}"
        assert request.headers["x-default-secret"] == default_header_secret
        assert request.headers["x-call-secret"] == call_header_secret
        return httpx2.Response(200, json=_RESULT)

    mlflow.typesafe.autolog()
    with TypeSafeClient(
        api_key=api_key,
        headers={"x-default-secret": default_header_secret},
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    ) as client:
        client.system_one(
            "hello",
            {"relevant": Noul(instructions="Relevant?")},
            timeout=1.25,
            extra_headers={"x-call-secret": call_header_secret},
        )

    trace = get_traces()[0]
    span = trace.data.spans[0]
    assert set(span.inputs) == {"state", "model", "questions"}
    serialized_trace = json.dumps(trace.to_dict(), default=str)
    for secret in (api_key, default_header_secret, call_header_secret):
        assert secret not in serialized_trace


def test_disable_and_log_traces_false():
    def handler(request):
        return httpx2.Response(200, json=_RESULT)

    call_kwargs = {
        "state": "hello",
        "questions": {"relevant": Noul(instructions="Relevant?")},
    }
    client_kwargs = {"retry": RetryPolicy(max_retries=0)}

    mlflow.typesafe.autolog()
    _system_one(False, handler, client_kwargs=client_kwargs, call_kwargs=call_kwargs)
    assert len(get_traces()) == 1

    mlflow.typesafe.autolog(disable=True)
    _system_one(False, handler, client_kwargs=client_kwargs, call_kwargs=call_kwargs)
    assert len(get_traces()) == 1

    mlflow.typesafe.autolog(log_traces=False)
    _system_one(False, handler, client_kwargs=client_kwargs, call_kwargs=call_kwargs)
    assert len(get_traces()) == 1

    mlflow.typesafe.autolog()
    _system_one(False, handler, client_kwargs=client_kwargs, call_kwargs=call_kwargs)
    assert len(get_traces()) == 2


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_models_list_is_not_traced(async_mode):
    mlflow.typesafe.autolog()
    response = _list_models(
        async_mode,
        lambda request: httpx2.Response(200, json={"models": []}),
    )

    assert response.models == ()
    assert get_traces() == []

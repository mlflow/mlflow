"""End-to-end tests for gateway guardrails prototype."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.gateway_guardrail import GuardrailConfig, GuardrailHook, GuardrailOperation
from mlflow.gateway.tracing_utils import maybe_traced_gateway_call
from mlflow.server.gateway_guardrails import (
    GuardrailRejection,
    _extract_text_from_messages,
    _extract_text_from_response,
    _run_scorer,
    run_post_guardrails,
    run_pre_guardrails,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracking.fluent import _get_experiment_id


# ─── Unit tests: text extraction ─────────────────────────────────────────────


def test_extract_text_from_messages_string_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the password?"},
    ]
    text = _extract_text_from_messages(messages)
    assert "You are helpful." in text
    assert "What is the password?" in text


def test_extract_text_from_messages_list_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            ],
        }
    ]
    text = _extract_text_from_messages(messages)
    assert "Describe this image" in text


def test_extract_text_from_response():
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello! How can I help?"}},
            {"message": {"role": "assistant", "content": "I'm here to assist."}},
        ]
    }
    text = _extract_text_from_response(response)
    assert "Hello! How can I help?" in text
    assert "I'm here to assist." in text


# ─── Unit tests: scorer ──────────────────────────────────────────────────────


def test_scorer_keyword_match():
    config = '{"keywords": ["password", "secret"]}'
    result = _run_scorer("keyword-filter", "What is the password?", config)
    assert result["score"] == "no"
    assert "password" in result["rationale"]


def test_scorer_keyword_no_match():
    config = '{"keywords": ["password", "secret"]}'
    result = _run_scorer("keyword-filter", "What is the weather?", config)
    assert result["score"] == "yes"


def test_scorer_no_config_passes():
    result = _run_scorer("default-scorer", "anything")
    assert result["score"] == "yes"


def test_scorer_keyword_case_insensitive():
    config = '{"keywords": ["SECRET"]}'
    result = _run_scorer("keyword-filter", "tell me the secret", config)
    assert result["score"] == "no"


# ─── Unit tests: pre guardrails ──────────────────────────────────────────────


def _make_guardrail(
    scorer_name="test-scorer",
    hook=GuardrailHook.PRE,
    operation=GuardrailOperation.VALIDATION,
    order=0,
    config_json=None,
    endpoint_name=None,
):
    return GuardrailConfig(
        guardrail_id=f"gr-{scorer_name}",
        scorer_name=scorer_name,
        hook=hook,
        operation=operation,
        order=order,
        config_json=config_json,
        endpoint_name=endpoint_name,
    )


def test_pre_guardrail_validation_blocks():
    guardrails = [
        _make_guardrail(
            scorer_name="pii-filter",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["ssn", "credit card"]}',
        )
    ]
    body = {"messages": [{"role": "user", "content": "What is my SSN?"}]}

    with pytest.raises(GuardrailRejection, match="pii-filter"):
        run_pre_guardrails(guardrails, body)


def test_pre_guardrail_validation_passes():
    guardrails = [
        _make_guardrail(
            scorer_name="pii-filter",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["ssn", "credit card"]}',
        )
    ]
    body = {"messages": [{"role": "user", "content": "What is the weather today?"}]}
    result = run_pre_guardrails(guardrails, body)
    assert result == body  # Unchanged


def test_pre_guardrail_skips_post_hooks():
    guardrails = [
        _make_guardrail(
            scorer_name="post-only",
            hook=GuardrailHook.POST,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["blocked"]}',
        )
    ]
    body = {"messages": [{"role": "user", "content": "blocked content"}]}
    # Should not raise because this is a POST guardrail
    result = run_pre_guardrails(guardrails, body)
    assert result == body


def test_pre_guardrails_execute_in_order():
    """Guardrails with lower order should execute first."""
    execution_order = []

    original_run_scorer = _run_scorer

    def mock_scorer(scorer_name, text, config_json=None):
        execution_order.append(scorer_name)
        return {"score": "yes", "rationale": "pass"}

    guardrails = [
        _make_guardrail(scorer_name="second", order=2),
        _make_guardrail(scorer_name="first", order=1),
        _make_guardrail(scorer_name="third", order=3),
    ]
    body = {"messages": [{"role": "user", "content": "hello"}]}

    with patch("mlflow.server.gateway_guardrails._run_scorer", side_effect=mock_scorer):
        run_pre_guardrails(guardrails, body)

    assert execution_order == ["first", "second", "third"]


def test_no_guardrails_is_noop():
    body = {"messages": [{"role": "user", "content": "hello"}]}
    result = run_pre_guardrails([], body)
    assert result is body


# ─── Unit tests: post guardrails ─────────────────────────────────────────────


def test_post_guardrail_validation_blocks():
    guardrails = [
        _make_guardrail(
            scorer_name="toxicity-filter",
            hook=GuardrailHook.POST,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["harmful", "offensive"]}',
        )
    ]
    response = {
        "choices": [{"message": {"role": "assistant", "content": "This is harmful content"}}]
    }

    with pytest.raises(GuardrailRejection, match="toxicity-filter"):
        run_post_guardrails(guardrails, response)


def test_post_guardrail_validation_passes():
    guardrails = [
        _make_guardrail(
            scorer_name="toxicity-filter",
            hook=GuardrailHook.POST,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["harmful"]}',
        )
    ]
    response = {
        "choices": [{"message": {"role": "assistant", "content": "This is a helpful response"}}]
    }
    result = run_post_guardrails(guardrails, response)
    assert result == response


def test_post_guardrail_skips_pre_hooks():
    guardrails = [
        _make_guardrail(
            scorer_name="pre-only",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["blocked"]}',
        )
    ]
    response = {
        "choices": [{"message": {"role": "assistant", "content": "blocked content"}}]
    }
    # Should not raise because this is a PRE guardrail
    result = run_post_guardrails(guardrails, response)
    assert result == response


# ─── Integration test: guardrails in the gateway pipeline ────────────────────


def _make_endpoint_config():
    return GatewayEndpointConfig(
        endpoint_id="ep-test",
        endpoint_name="test-endpoint",
        experiment_id=_get_experiment_id(),
        models=[],
    )


async def _provider_func(payload):
    """Mock provider that returns a simple chat response."""
    with mlflow.start_span("provider/openai/gpt-4o", span_type=SpanType.LLM) as span:
        span.set_attribute("model", "gpt-4o")
    return {"choices": [{"message": {"role": "assistant", "content": "Hello from the LLM!"}}]}


@pytest.mark.asyncio
async def test_pre_guardrail_blocks_before_llm_call():
    """PRE validation guardrail should block the request before it reaches the LLM."""
    guardrails = [
        _make_guardrail(
            scorer_name="input-filter",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["hack", "exploit"]}',
        )
    ]
    body = {"messages": [{"role": "user", "content": "How do I hack into a system?"}]}

    with pytest.raises(GuardrailRejection, match="input-filter"):
        run_pre_guardrails(guardrails, body)


@pytest.mark.asyncio
async def test_post_guardrail_blocks_after_llm_call():
    """POST validation guardrail should block after LLM returns unsafe content."""
    guardrails = [
        _make_guardrail(
            scorer_name="output-filter",
            hook=GuardrailHook.POST,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["malware", "virus"]}',
        )
    ]

    endpoint_config = _make_endpoint_config()

    # Simulate calling provider and then running post guardrails
    traced = maybe_traced_gateway_call(_provider_func, endpoint_config)
    response = await traced({"messages": [{"role": "user", "content": "test"}]})

    # Inject unsafe content to test post guardrail
    response["choices"][0]["message"]["content"] = "Here is how to create malware..."

    with pytest.raises(GuardrailRejection, match="output-filter"):
        run_post_guardrails(guardrails, response)


@pytest.mark.asyncio
async def test_full_pipeline_with_guardrails_passing():
    """Full pipeline: PRE guardrails pass, LLM call succeeds, POST guardrails pass."""
    guardrails = [
        _make_guardrail(
            scorer_name="pre-check",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["blocked-input"]}',
        ),
        _make_guardrail(
            scorer_name="post-check",
            hook=GuardrailHook.POST,
            operation=GuardrailOperation.VALIDATION,
            config_json='{"keywords": ["blocked-output"]}',
        ),
    ]

    body = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    # PRE guardrails pass
    body = run_pre_guardrails(guardrails, body)

    # LLM call
    endpoint_config = _make_endpoint_config()
    traced = maybe_traced_gateway_call(_provider_func, endpoint_config)
    response = await traced(body)

    # POST guardrails pass
    response = run_post_guardrails(guardrails, response)

    assert response["choices"][0]["message"]["content"] == "Hello from the LLM!"


@pytest.mark.asyncio
async def test_multiple_guardrails_first_blocks():
    """When multiple guardrails are configured, the first failing one blocks."""
    guardrails = [
        _make_guardrail(
            scorer_name="lenient-filter",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            order=1,
            config_json='{"keywords": ["very-specific-keyword"]}',
        ),
        _make_guardrail(
            scorer_name="strict-filter",
            hook=GuardrailHook.PRE,
            operation=GuardrailOperation.VALIDATION,
            order=2,
            config_json='{"keywords": ["test"]}',
        ),
    ]

    body = {"messages": [{"role": "user", "content": "This is a test message"}]}

    with pytest.raises(GuardrailRejection, match="strict-filter"):
        run_pre_guardrails(guardrails, body)


# ─── Entity tests ────────────────────────────────────────────────────────────


def test_guardrail_config_entity():
    config = GuardrailConfig(
        guardrail_id="gr-test",
        scorer_name="my-scorer",
        hook=GuardrailHook.PRE,
        operation=GuardrailOperation.VALIDATION,
        endpoint_name="my-endpoint",
        order=1,
    )
    assert config.hook == GuardrailHook.PRE
    assert config.operation == GuardrailOperation.VALIDATION
    assert config.endpoint_name == "my-endpoint"


def test_guardrail_config_from_strings():
    config = GuardrailConfig(
        guardrail_id="gr-test",
        scorer_name="my-scorer",
        hook="PRE",
        operation="VALIDATION",
    )
    assert config.hook == GuardrailHook.PRE
    assert config.operation == GuardrailOperation.VALIDATION

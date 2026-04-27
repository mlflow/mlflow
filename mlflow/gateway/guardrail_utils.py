from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from fastapi import Request

from mlflow.entities.gateway_guardrail import GuardrailStage
from mlflow.gateway.guardrails import JudgeGuardrail
from mlflow.gateway.schemas import chat
from mlflow.types.chat import ChatCompletionResponse

if TYPE_CHECKING:
    from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

_logger = logging.getLogger(__name__)


def load_guardrails(
    store: SqlAlchemyStore,
    endpoint_config: GatewayEndpointConfig,
    request: Request,
) -> list[JudgeGuardrail]:
    """Load guardrails for an endpoint and convert to callable JudgeGuardrail instances."""
    # Configs are returned ordered by execution_order ASC (nulls last), then guardrail_id.
    configs = store.list_endpoint_guardrail_configs(endpoint_config.endpoint_id)
    if not configs:
        return []

    server_url = str(request.base_url).rstrip("/")
    guardrails = []
    for config in configs:
        if config.guardrail is None:
            continue
        try:
            resolved_scorer = store.resolve_endpoint_in_scorer(config.guardrail.scorer)
            guardrail = dataclasses.replace(config.guardrail, scorer=resolved_scorer)
            guardrails.append(JudgeGuardrail.from_entity(guardrail, server_url))
        except Exception:
            _logger.warning(
                "Failed to load guardrail %s, skipping", config.guardrail_id, exc_info=True
            )
    return guardrails


def extract_auth_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return only the Authorization header for internal guardrail calls."""
    auth = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
    return {"authorization": auth} if auth else {}


async def run_pre_llm_guardrails(
    guardrails: list[JudgeGuardrail],
    payload_dict: dict[str, Any],
    auth_headers: dict[str, str] | None = None,
    usage_tracking: bool = False,
    payload_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run pre-LLM guardrails on the request payload. Returns the (possibly modified) dict."""
    for guardrail in guardrails:
        if guardrail.stage == GuardrailStage.BEFORE:
            payload_dict = await guardrail.process_request(
                payload_dict,
                auth_headers=auth_headers,
                usage_tracking=usage_tracking,
                payload_schema=payload_schema,
            )
    return payload_dict


async def run_post_llm_guardrails(
    guardrails: list[JudgeGuardrail],
    request_payload: dict[str, Any],
    response: chat.ResponsePayload,
    auth_headers: dict[str, str] | None = None,
    usage_tracking: bool = False,
) -> chat.ResponsePayload:
    """Run post-LLM guardrails on the response. Returns the (possibly modified) response.

    Note: post-LLM guardrails are skipped for streaming responses. Configure guardrails
    that must run on all responses to use the pre-LLM stage, or disable streaming on the endpoint.
    """
    post_llm_guardrails = [g for g in guardrails if g.stage == GuardrailStage.AFTER]
    if not post_llm_guardrails:
        return response

    response_dict = response.model_dump()
    schema = ChatCompletionResponse.model_json_schema()
    for guardrail in post_llm_guardrails:
        response_dict = await guardrail.process_response(
            request_payload,
            response_dict,
            auth_headers=auth_headers,
            usage_tracking=usage_tracking,
            payload_schema=schema,
        )
    return chat.ResponsePayload(**response_dict)


async def run_post_llm_guardrails_passthrough(
    guardrails: list[JudgeGuardrail],
    request_payload: dict[str, Any],
    response: dict[str, Any],
    auth_headers: dict[str, str] | None = None,
    usage_tracking: bool = False,
) -> dict[str, Any]:
    """Run post-LLM guardrails for passthrough endpoints.

    Like ``run_post_llm_guardrails`` but accepts and returns a plain ``dict``
    instead of a ``chat.ResponsePayload``. No ``response_format`` schema constraint
    is applied since passthrough responses are provider-specific formats.

    Note: post-LLM guardrails are skipped for streaming responses. Configure
    guardrails that must run on all responses to use the pre-LLM stage, or
    disable streaming on the endpoint.
    """
    for guardrail in guardrails:
        if guardrail.stage != GuardrailStage.AFTER:
            continue
        response = await guardrail.process_response(
            request_payload, response, auth_headers=auth_headers, usage_tracking=usage_tracking
        )
    return response

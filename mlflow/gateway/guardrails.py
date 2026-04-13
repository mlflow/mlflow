from __future__ import annotations

import abc
import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from mlflow.entities.assessment import Feedback
from mlflow.entities.gateway_guardrail import (
    GatewayGuardrail,
    GuardrailAction,
    GuardrailStage,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.providers.utils import send_request
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.metrics.genai.model_utils import _parse_model_uri
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.chat import ChatCompletionRequest, ChatCompletionResponse

if TYPE_CHECKING:
    from mlflow.genai.scorers import Scorer

# In practice scorers return bool or str (e.g. "yes"/"no"), but Feedback and
# list[Feedback] are also supported for richer structured output.
ScorerResult = bool | str | Feedback | list[Feedback]

# Header added to internal sanitization requests so the gateway can skip guardrails
# on the call, preventing recursive guardrail execution loops.
_SANITIZE_BYPASS_HEADER = "X-MLflow-Guardrail-Bypass"
_MAX_RATIONALE_LEN = 500

# Only forward these headers to internal sanitization calls — forwarding all
# incoming headers (e.g. Host, Content-Length) can cause failures.
_ALLOWED_AUTH_HEADERS = frozenset({"authorization"})

_SANITIZE_SYSTEM_PROMPT = """\
You are a content sanitizer. You will receive a JSON payload and an issue description.
Rewrite the payload to address the issue while preserving the structure and intent.
Return ONLY a valid JSON object with the same schema as the input payload.

Issue: {rationale}

Input payload:
{payload_json}"""


class GuardrailViolation(MlflowException):
    """Raised when a guardrail blocks a request or response."""

    def __init__(self, guardrail_name: str, rationale: str) -> None:
        self.guardrail_name = guardrail_name
        self.rationale = rationale
        super().__init__(
            f"Guardrail '{guardrail_name}' blocked: {rationale}",
            error_code=INVALID_PARAMETER_VALUE,
        )


class Guardrail(abc.ABC):
    """Base interface for gateway guardrails.

    A guardrail inspects and optionally transforms request/response payloads.
    Subclasses must implement ``process_request`` and ``process_response``.
    """

    @abc.abstractmethod
    async def process_request(
        self,
        request: dict[str, Any],
        auth_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Process an incoming request payload before LLM invocation.

        Args:
            request: The chat request payload as a dict.
            auth_headers: Optional HTTP headers to forward when making
                internal calls (e.g. sanitization via the gateway).

        Returns:
            The (possibly modified) request payload.

        Raises:
            GuardrailViolation: If the guardrail blocks the request.
        """

    @abc.abstractmethod
    async def process_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        auth_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Process an outgoing response payload after LLM invocation.

        Args:
            request: The original request payload.
            response: The chat response payload as a dict.
            auth_headers: Optional HTTP headers to forward when making
                internal calls (e.g. sanitization via the gateway).

        Returns:
            The (possibly modified) response payload.

        Raises:
            GuardrailViolation: If the guardrail blocks the response.
        """


class JudgeGuardrail(Guardrail):
    """Guardrail backed by an MLflow scorer/judge.

    Runs the judge on the request (BEFORE stage) or response (AFTER stage).
    For VALIDATION actions, blocks if the judge returns a failing value.
    For SANITIZATION actions, delegates to ``_sanitize`` (implemented in a
    future PR that wires guardrails into the gateway request flow).

    Args:
        scorer: An MLflow ``Scorer`` instance (e.g. from ``get_scorer``).
        stage: Whether this guardrail runs BEFORE or AFTER LLM invocation.
        action: Whether the guardrail validates (blocks) or sanitizes (modifies).
        name: Human-readable name for error messages.
        action_llm_url: Full gateway invocations URL for the sanitization LLM.
            Built by the caller from the endpoint name and server address.
    """

    def __init__(
        self,
        scorer: Scorer,
        stage: GuardrailStage,
        action: GuardrailAction,
        name: str,
        action_llm_url: str | None = None,
        action_endpoint_name: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.stage = stage
        self.action = action
        self.name = name
        self.action_llm_url = action_llm_url
        self.action_endpoint_name = action_endpoint_name

    def _invoke_judge(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
    ) -> ScorerResult:
        kwargs: dict[str, Any] = {}
        if inputs is not None:
            kwargs["inputs"] = inputs
        if outputs is not None:
            kwargs["outputs"] = outputs
        return self.scorer(**kwargs)

    def _is_passing(self, result: ScorerResult) -> bool:
        """Determine whether the judge result indicates a pass.

        When the result is a ``Feedback``, reads ``.value``. For plain
        scalars, interprets the value directly. String values are compared
        against ``CategoricalRating.YES``.
        """
        if isinstance(result, list):
            return all(self._is_passing(f) for f in result)
        if isinstance(result, Feedback):
            value = result.value
        elif isinstance(result, (bool, str)):
            value = result
        else:
            raise TypeError(
                f"Scorer returned an unexpected value type {type(result).__name__!r}; "
                "expected bool, str, Feedback, or list[Feedback]."
            )

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == CategoricalRating.YES.value
        raise TypeError(
            f"Scorer or Feedback returned an unexpected value type {type(value).__name__!r}; "
            "expected bool or str."
        )

    def _get_rationale(self, result: ScorerResult) -> str:
        if isinstance(result, Feedback):
            raw = result.rationale or str(result.value)
        elif isinstance(result, list):
            failing = [self._get_rationale(f) for f in result if not self._is_passing(f)]
            raw = "; ".join(failing) if failing else ""
        else:
            raw = str(result)
        return raw[:_MAX_RATIONALE_LEN]

    async def _sanitize(
        self,
        payload: dict[str, Any],
        rationale: str,
        payload_model: type[ChatCompletionRequest] | type[ChatCompletionResponse],
        auth_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send the full payload to the action endpoint LLM for rewriting.

        Posts a chat request to ``action_llm_url`` which is the fully
        resolved gateway invocations URL.
        """
        if not self.action_llm_url or not self.action_endpoint_name:
            raise GuardrailViolation(
                self.name,
                "Sanitization requires an action_llm_url but none was configured.",
            )

        url = self.action_llm_url
        path = f"gateway/{self.action_endpoint_name}/mlflow/invocations"
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": _SANITIZE_SYSTEM_PROMPT.format(
                        rationale=rationale,
                        payload_json=json.dumps(payload, indent=2),
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sanitized_payload",
                    "strict": False,
                    "schema": payload_model.model_json_schema(),
                },
            },
        }

        headers = (
            {k: v for k, v in auth_headers.items() if k.lower() in _ALLOWED_AUTH_HEADERS}
            if auth_headers
            else {}
        )
        # Bypass guardrails on the sanitization call to prevent recursive loops.
        headers[_SANITIZE_BYPASS_HEADER] = "1"

        try:
            resp_json = await send_request(headers=headers, base_url=url, path=path, payload=body)
        except HTTPException as e:
            raise GuardrailViolation(self.name, f"Sanitization request failed: {e.detail}") from e

        try:
            content = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise GuardrailViolation(
                self.name,
                "Sanitization LLM response is missing 'choices[0].message.content'.",
            ) from e

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise GuardrailViolation(
                self.name,
                "Sanitization LLM returned invalid JSON.",
            ) from e

    async def _enforce(
        self,
        payload: dict[str, Any],
        payload_model: type[ChatCompletionRequest] | type[ChatCompletionResponse],
        result: ScorerResult,
        auth_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Block or sanitize *payload* based on *result*.

        Raises ``GuardrailViolation`` for VALIDATION action, or delegates to
        ``_sanitize`` for SANITIZATION action.  Returns *payload* unchanged
        when the judge passes.
        """
        if self._is_passing(result):
            return payload

        rationale = self._get_rationale(result)

        if self.action == GuardrailAction.VALIDATION:
            raise GuardrailViolation(self.name, rationale)

        return await self._sanitize(payload, rationale, payload_model, auth_headers=auth_headers)

    async def process_request(
        self,
        request: dict[str, Any],
        auth_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.stage == GuardrailStage.AFTER:
            return request

        result = await asyncio.to_thread(self._invoke_judge, inputs=request)
        return await self._enforce(request, ChatCompletionRequest, result, auth_headers)

    async def process_response(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
        auth_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self.stage == GuardrailStage.BEFORE:
            return response

        result = await asyncio.to_thread(self._invoke_judge, inputs=request, outputs=response)
        return await self._enforce(response, ChatCompletionResponse, result, auth_headers)

    @classmethod
    def from_entity(cls, entity: GatewayGuardrail, server_url: str | None = None) -> JudgeGuardrail:
        """Convert a ``GatewayGuardrail`` entity into a callable ``JudgeGuardrail``.

        Deserializes the scorer stored in ``entity.scorer`` (a ``ScorerVersion``)
        back into a live ``Scorer`` instance.

        Args:
            entity: A ``GatewayGuardrail`` entity containing a ``ScorerVersion``.
            server_url: Base URL of the MLflow server (e.g. ``http://localhost:5000``).
                Used to build the sanitization invocations URL from
                ``entity.action_endpoint_name``.

        Returns:
            A ``JudgeGuardrail`` ready to process requests/responses.
        """
        from mlflow.genai.judges.instructions_judge import (
            InstructionsJudge,  # lazy: heavy transitive deps
        )
        from mlflow.genai.scorers import Scorer  # lazy: heavy transitive deps

        action_llm_url = (
            server_url.rstrip("/") if entity.action_endpoint_name and server_url else None
        )

        scorer = Scorer.model_validate(entity.scorer.serialized_scorer)

        # Inside the server process MLFLOW_TRACKING_URI points to the backend store
        # (e.g. sqlite://), so _resolve_gateway_uri() would fail for gateway:/ URIs.
        # Rewrite to openai:/ with an explicit base_url derived from the HTTP request
        # URL so the judge calls the gateway directly without touching the tracking URI.
        if server_url and isinstance(scorer, InstructionsJudge) and scorer.model:
            provider, endpoint_name = _parse_model_uri(scorer.model)
            if provider == "gateway":
                scorer = InstructionsJudge(
                    name=scorer.name,
                    instructions=scorer._instructions,
                    model=f"openai:/{endpoint_name}",
                    base_url=f"{server_url.rstrip('/')}/gateway/mlflow/v1/chat/completions",
                    feedback_value_type=scorer._feedback_value_type,
                    inference_params=scorer._inference_params,
                )

        return cls(
            scorer=scorer,
            stage=entity.stage,
            action=entity.action,
            name=entity.name,
            action_llm_url=action_llm_url,
            action_endpoint_name=entity.action_endpoint_name,
        )

from __future__ import annotations

import abc
import json
from typing import TYPE_CHECKING, Any

import requests

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.gateway_guardrail import (
    GatewayGuardrail,
    GuardrailAction,
    GuardrailStage,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

if TYPE_CHECKING:
    from mlflow.genai.scorers import Scorer

# Scorer.__call__ returns this union type.
ScorerResult = int | float | bool | str | Feedback | list[Feedback]

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
    def process_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming request payload before LLM invocation.

        Args:
            payload: The chat request payload as a dict.

        Returns:
            The (possibly modified) request payload.

        Raises:
            GuardrailViolation: If the guardrail blocks the request.
        """

    @abc.abstractmethod
    def process_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process an outgoing response payload after LLM invocation.

        Args:
            payload: The chat response payload as a dict.

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
        action_endpoint_id: Gateway endpoint ID for the LLM used by sanitization.
    """

    def __init__(
        self,
        scorer: Scorer,
        stage: GuardrailStage,
        action: GuardrailAction,
        name: str,
        action_endpoint_id: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.stage = stage
        self.action = action
        self.name = name
        self.action_endpoint_id = action_endpoint_id

    def _extract_text(self, payload: dict[str, Any], *, is_response: bool) -> str:
        if is_response:
            if (choices := payload.get("choices", [])) and (
                content := choices[0].get("message", {}).get("content", "")
            ):
                return content
            return ""
        if (messages := payload.get("messages", [])) and (
            content := messages[-1].get("content", "")
        ):
            return content
        return ""

    def _invoke_judge(self, text: str) -> ScorerResult:
        return self.scorer(outputs=text)

    def _is_passing(self, result: ScorerResult) -> bool:
        """Determine whether the judge result indicates a pass.

        When the result is a ``Feedback``, reads ``.value``. For plain
        scalars, interprets the value directly. String values are compared
        against ``CategoricalRating.YES``.
        """
        if isinstance(result, Feedback):
            value = result.value
        elif isinstance(result, list):
            return all(self._is_passing(f) for f in result)
        else:
            value = result

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == CategoricalRating.YES.value
        return bool(value)

    def _get_rationale(self, result: ScorerResult) -> str:
        if isinstance(result, Feedback):
            if result.rationale:
                return result.rationale
            return str(result.value)
        if isinstance(result, list):
            failing = [self._get_rationale(f) for f in result if not self._is_passing(f)]
            return "; ".join(failing) if failing else ""
        return str(result)

    def _sanitize(self, payload: dict[str, Any], rationale: str) -> dict[str, Any]:
        """Send the full payload to the action endpoint LLM for rewriting.

        Posts a chat request to the gateway invocations endpoint.
        """
        if not self.action_endpoint_id:
            raise GuardrailViolation(
                self.name,
                "Sanitization requires an action_endpoint_id but none was configured.",
            )

        tracking_uri = mlflow.get_tracking_uri().rstrip("/")
        url = f"{tracking_uri}/gateway/{self.action_endpoint_id}/mlflow/invocations"
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
        }

        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise GuardrailViolation(
                self.name,
                f"Sanitization LLM returned invalid JSON: {content[:200]}",
            ) from e

    def process_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stage == GuardrailStage.AFTER:
            return payload

        text = self._extract_text(payload, is_response=False)
        result = self._invoke_judge(text)

        if self._is_passing(result):
            return payload

        rationale = self._get_rationale(result)

        if self.action == GuardrailAction.VALIDATION:
            raise GuardrailViolation(self.name, rationale)

        return self._sanitize(payload, rationale)

    def process_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stage == GuardrailStage.BEFORE:
            return payload

        text = self._extract_text(payload, is_response=True)
        result = self._invoke_judge(text)

        if self._is_passing(result):
            return payload

        rationale = self._get_rationale(result)

        if self.action == GuardrailAction.VALIDATION:
            raise GuardrailViolation(self.name, rationale)

        return self._sanitize(payload, rationale)

    @classmethod
    def from_entity(cls, entity: GatewayGuardrail) -> JudgeGuardrail:
        """Convert a ``GatewayGuardrail`` entity into a callable ``JudgeGuardrail``.

        Deserializes the scorer stored in ``entity.scorer`` (a ``ScorerVersion``)
        back into a live ``Scorer`` instance.

        Args:
            entity: A ``GatewayGuardrail`` entity containing a ``ScorerVersion``.

        Returns:
            A ``JudgeGuardrail`` ready to process requests/responses.
        """
        from mlflow.genai.scorers import Scorer

        scorer = Scorer.model_validate(entity.scorer.serialized_scorer)
        return cls(
            scorer=scorer,
            stage=entity.stage,
            action=entity.action,
            name=entity.name,
            action_endpoint_id=entity.action_endpoint_id,
        )

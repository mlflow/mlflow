from __future__ import annotations

import abc
import copy
import json
from typing import TYPE_CHECKING, Any

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

_SANITIZE_SYSTEM_PROMPT = (
    "You are a content sanitizer. Rewrite the following text to address the "
    "issue described below. Preserve the original meaning and intent as much "
    "as possible while resolving the problem. Output only the rewritten text "
    "with no additional commentary.\n\n"
    "Issue: {rationale}\n\n"
    "Original text:\n{text}"
)


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
    For SANITIZATION actions, invokes an LLM via ``action_endpoint_id`` to
    rewrite the content based on the judge's feedback.

    Args:
        scorer: An MLflow ``Scorer`` instance (e.g. from ``get_scorer``).
        stage: Whether this guardrail runs BEFORE or AFTER LLM invocation.
        action: Whether the guardrail validates (blocks) or sanitizes (modifies).
        name: Human-readable name for error messages.
        action_endpoint_id: Gateway endpoint ID for the LLM used by sanitization.
            Required when ``action`` is ``SANITIZATION``.
    """

    def __init__(
        self,
        scorer: Scorer,
        stage: GuardrailStage,
        action: GuardrailAction,
        name: str = "judge-guardrail",
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

    def _sanitize(self, text: str, rationale: str) -> str:
        """Call the action endpoint LLM to rewrite ``text`` based on ``rationale``.

        Resolves the action endpoint's provider config and calls
        ``_invoke_via_gateway`` to perform the rewrite.
        """
        from mlflow.genai.judges.adapters.gateway_adapter import _invoke_via_gateway
        from mlflow.store.tracking.gateway.config_resolver import get_endpoint_config

        if not self.action_endpoint_id:
            raise GuardrailViolation(
                self.name,
                "Sanitization requires an action_endpoint_id but none was configured.",
            )

        endpoint_config = get_endpoint_config(self.action_endpoint_id)
        model_config = endpoint_config.model_configs[0]
        provider = model_config.provider
        model_name = model_config.model_name
        model_uri = f"{provider}:/{model_name}"

        prompt = [
            {
                "role": "user",
                "content": _SANITIZE_SYSTEM_PROMPT.format(rationale=rationale, text=text),
            },
        ]

        response = _invoke_via_gateway(model_uri, provider, prompt)
        # _invoke_via_gateway returns a JSON string; extract the text content.
        try:
            return json.loads(response)["result"]
        except (json.JSONDecodeError, KeyError):
            # If not structured JSON, the raw response is the rewritten text.
            return response

    def _apply_sanitization(
        self, payload: dict[str, Any], rationale: str, *, is_response: bool
    ) -> dict[str, Any]:
        text = self._extract_text(payload, is_response=is_response)
        rewritten = self._sanitize(text, rationale)
        payload = copy.deepcopy(payload)
        if is_response:
            payload["choices"][0]["message"]["content"] = rewritten
        else:
            payload["messages"][-1]["content"] = rewritten
        return payload

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

        return self._apply_sanitization(payload, rationale, is_response=False)

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

        return self._apply_sanitization(payload, rationale, is_response=True)


def from_entity(entity: GatewayGuardrail) -> JudgeGuardrail:
    """Convert a ``GatewayGuardrail`` entity (DB model) into a callable ``JudgeGuardrail``.

    Deserializes the scorer stored in ``entity.scorer`` (a ``ScorerVersion``)
    back into a live ``Scorer`` instance and wraps it in a ``JudgeGuardrail``.

    Args:
        entity: A ``GatewayGuardrail`` entity containing a ``ScorerVersion``.

    Returns:
        A ``JudgeGuardrail`` ready to process requests/responses.
    """
    from mlflow.genai.scorers import Scorer

    scorer = Scorer.model_validate(entity.scorer.serialized_scorer)
    return JudgeGuardrail(
        scorer=scorer,
        stage=entity.stage,
        action=entity.action,
        name=entity.name,
        action_endpoint_id=entity.action_endpoint_id,
    )

from __future__ import annotations

import abc
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
    For SANITIZATION actions, invokes an LLM to modify the payload based on
    the judge's feedback.

    Args:
        scorer: An MLflow ``Scorer`` instance (e.g. from ``get_scorer``).
        stage: Whether this guardrail runs BEFORE or AFTER LLM invocation.
        action: Whether the guardrail validates (blocks) or sanitizes (modifies).
        name: Human-readable name for error messages.
    """

    def __init__(
        self,
        scorer: Scorer,
        stage: GuardrailStage,
        action: GuardrailAction,
        name: str = "judge-guardrail",
    ) -> None:
        self.scorer = scorer
        self.stage = stage
        self.action = action
        self.name = name

    def _extract_text(self, payload: dict[str, Any], *, is_response: bool) -> str:
        if is_response:
            choices = payload.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        messages = payload.get("messages", [])
        if messages:
            return messages[-1].get("content", "")
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

    def process_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stage == GuardrailStage.AFTER:
            return payload

        text = self._extract_text(payload, is_response=False)
        result = self._invoke_judge(text)

        if self._is_passing(result):
            return payload

        if self.action == GuardrailAction.VALIDATION:
            raise GuardrailViolation(self.name, self._get_rationale(result))

        # SANITIZATION: placeholder — subclasses or future work can invoke an
        # LLM to rewrite the content. For now, raise so we don't silently drop.
        raise GuardrailViolation(
            self.name,
            f"Sanitization not yet implemented. Judge feedback: {self._get_rationale(result)}",
        )

    def process_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stage == GuardrailStage.BEFORE:
            return payload

        text = self._extract_text(payload, is_response=True)
        result = self._invoke_judge(text)

        if self._is_passing(result):
            return payload

        if self.action == GuardrailAction.VALIDATION:
            raise GuardrailViolation(self.name, self._get_rationale(result))

        raise GuardrailViolation(
            self.name,
            f"Sanitization not yet implemented. Judge feedback: {self._get_rationale(result)}",
        )


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
        name=f"guardrail-{entity.guardrail_id}",
    )

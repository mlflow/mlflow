from __future__ import annotations

import abc
from typing import Any

from mlflow.entities.gateway_guardrail import GuardrailAction, GuardrailStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class GuardrailViolation(MlflowException):
    """Raised when a guardrail blocks a request or response."""

    def __init__(self, guardrail_name: str, rationale: str):
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
        scorer: An MLflow scorer instance (e.g. from ``get_scorer``).
        stage: Whether this guardrail runs BEFORE or AFTER LLM invocation.
        action: Whether the guardrail validates (blocks) or sanitizes (modifies).
        name: Human-readable name for error messages.
    """

    def __init__(
        self,
        scorer,
        stage: GuardrailStage,
        action: GuardrailAction,
        name: str = "judge-guardrail",
    ):
        self.scorer = scorer
        self.stage = stage
        self.action = action
        self.name = name

    def _extract_text(self, payload: dict[str, Any], *, is_response: bool) -> str:
        """Pull the primary text content from a request or response dict."""
        if is_response:
            choices = payload.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        messages = payload.get("messages", [])
        if messages:
            return messages[-1].get("content", "")
        return ""

    def _invoke_judge(self, text: str) -> Any:
        """Call the scorer and return the feedback."""
        return self.scorer(outputs=text)

    def _is_passing(self, result) -> bool:
        """Determine whether the judge result indicates a pass.

        Handles ``Feedback`` objects (has ``.value``) and plain values.
        The convention is: ``True`` / ``"yes"`` means the content is acceptable.
        """
        value = getattr(result, "value", result)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("yes", "true", "pass")
        return bool(value)

    def _get_rationale(self, result) -> str:
        rationale = getattr(result, "rationale", None)
        if rationale:
            return str(rationale)
        metadata = getattr(result, "metadata", None) or {}
        return metadata.get("rationale", str(getattr(result, "value", result)))

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

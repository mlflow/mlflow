from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject


class GuardrailHook(str, Enum):
    PRE = "PRE"
    POST = "POST"


class GuardrailOperation(str, Enum):
    VALIDATION = "VALIDATION"
    MUTATION = "MUTATION"


@dataclass
class GuardrailConfig(_MlflowObject):
    """
    Represents a guardrail configuration for a gateway endpoint.

    A guardrail ties a scorer (judge) to an endpoint to run before or after
    LLM invocation. Validation guardrails block requests/responses (yes/no),
    while mutation guardrails modify them.

    Args:
        guardrail_id: Unique identifier.
        endpoint_name: Gateway endpoint this guardrail applies to.
            If None, applies to all endpoints.
        scorer_name: Name of the scorer to use as guardrail.
        hook: Whether to run PRE or POST LLM invocation.
        operation: Whether to VALIDATE (block) or MUTATE (modify).
        order: Execution order (lower = earlier). Default 0.
        enabled: Whether the guardrail is currently active.
        config_json: Optional JSON config for the scorer (e.g. custom prompt).
    """

    guardrail_id: str
    scorer_name: str
    hook: GuardrailHook
    operation: GuardrailOperation
    endpoint_name: str | None = None
    order: int = 0
    enabled: bool = True
    config_json: str | None = None

    def __post_init__(self):
        if isinstance(self.hook, str):
            self.hook = GuardrailHook(self.hook)
        if isinstance(self.operation, str):
            self.operation = GuardrailOperation(self.operation)

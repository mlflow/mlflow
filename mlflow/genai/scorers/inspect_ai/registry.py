"""Registry of the Inspect AI scorer factories exposed through MLflow."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.inspect_ai.utils import INSPECT_AI_NOT_INSTALLED_ERROR_MESSAGE

# Registry format: inspect_ai.scorer factory name -> is_model_graded
_SCORER_REGISTRY = {
    "exact": False,
    "includes": False,
    "match": False,
    "pattern": False,
    "f1": False,
    "answer": False,
    "model_graded_fact": True,
    "model_graded_qa": True,
}

# CamelCase names used by the concrete scorer classes, mapped to Inspect's factory names.
_SCORER_ALIASES = {
    "ExactMatch": "exact",
    "Includes": "includes",
    "Match": "match",
    "Pattern": "pattern",
    "F1": "f1",
    "Answer": "answer",
    "ModelGradedFact": "model_graded_fact",
    "ModelGradedQA": "model_graded_qa",
}


def resolve_factory_name(metric_name: str) -> str:
    return _SCORER_ALIASES.get(metric_name, metric_name)


def get_scorer_factory(metric_name: str) -> Callable[..., Any]:
    """Return the ``inspect_ai.scorer`` factory for ``metric_name``.

    Names outside the registry are looked up on ``inspect_ai.scorer`` directly, so every
    scorer factory shipped by Inspect can be used by its own name.
    """
    factory_name = resolve_factory_name(metric_name)
    try:
        import inspect_ai.scorer as inspect_scorer
    except ImportError as e:
        raise MlflowException.invalid_parameter_value(INSPECT_AI_NOT_INSTALLED_ERROR_MESSAGE) from e

    factory = getattr(inspect_scorer, factory_name, None)
    if factory is None or not callable(factory):
        available = ", ".join(sorted(_SCORER_REGISTRY))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Inspect AI scorer: '{metric_name}'. 'inspect_ai.scorer' has no factory "
            f"named '{factory_name}'. Pre-configured scorers: {available}",
            error_class="ATTRIBUTE_NOT_FOUND",
        )
    return factory


def is_model_graded(metric_name: str) -> bool:
    """Whether the scorer needs a grader model.

    Unknown factories are inspected for a ``model`` parameter.
    """
    factory_name = resolve_factory_name(metric_name)
    if factory_name in _SCORER_REGISTRY:
        return _SCORER_REGISTRY[factory_name]
    try:
        return "model" in inspect.signature(get_scorer_factory(factory_name)).parameters
    except (TypeError, ValueError):
        return False

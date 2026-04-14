"""Registry of Google ADK scorers exposed through ``get_scorer``."""

from __future__ import annotations

from mlflow.exceptions import MlflowException


def get_scorer_class(metric_name: str):
    """Return the Google ADK scorer class registered under ``metric_name``."""
    from mlflow.genai.scorers.google_adk import ResponseMatch, ToolTrajectory

    registry = {
        "ToolTrajectory": ToolTrajectory,
        "ResponseMatch": ResponseMatch,
    }

    if metric_name not in registry:
        raise MlflowException.invalid_parameter_value(
            f"Unknown Google ADK metric '{metric_name}'. "
            f"Available metrics: {sorted(registry.keys())}"
        )

    return registry[metric_name]

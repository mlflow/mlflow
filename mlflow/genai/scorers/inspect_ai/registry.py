"""Registry for Inspect AI scorer callables.

Responsibilities:
- Resolve a user-provided `metric_name` to a callable inside the installed
  Inspect AI package.
- Provide simple heuristics to detect if a task is session-level (conversational).

This mirrors the DeepEval `registry.py` responsibilities but is simplified for
Inspect AI's task/callable model: instead of importing metric classes by
classpath, we delegate resolution to the adapter which knows how to locate
callables inside the third-party package.
"""

from __future__ import annotations

from typing import Callable

from mlflow.exceptions import MlflowException

from mlflow.genai.scorers.inspect_ai import adapter

INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE = (
    "Inspect AI scorers require the 'inspectai' package. "
    "Please install it with: pip install inspectai"
)


def get_task_callable(metric_name: str) -> Callable[..., object]:
    """Return a callable for the given `metric_name`.

    Raises MlflowException with a helpful message if Inspect AI isn't
    installed or if the callable cannot be found.
    """
    try:
        return adapter.find_task_callable(metric_name)
    except MlflowException as e:
        raise
    except Exception as e:
        raise MlflowException(
            f"Failed to resolve Inspect AI task for '{metric_name}': {e}"
        ) from e


def is_session_level_task(metric_name: str) -> bool:
    """Heuristically determine whether the task operates on sessions.

    Many Inspect AI tasks may expose metadata flags such as `is_conversational`
    or `requires_session`. We check for a handful of common attribute names and
    default to False when unknown. This allows MLflow's `InspectAIScorer` to
    require the `session` parameter when appropriate.
    """
    try:
        fn = get_task_callable(metric_name)
    except MlflowException:
        
        return False

    for attr in ("is_conversational", "requires_session", "session_level"):
        if hasattr(fn, attr):
            try:
                return bool(getattr(fn, attr))
            except Exception:
                continue

    try:
        import inspect as _inspect

        sig = _inspect.signature(fn)
        return "session" in sig.parameters
    except Exception:
        return False
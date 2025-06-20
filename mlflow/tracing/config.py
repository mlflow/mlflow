from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mlflow.tracing.utils.processor import validate_span_processors
from mlflow.utils.annotations import experimental


@dataclass
class TracingConfig:
    """Configuration for MLflow tracing behavior."""

    # TODO: Move more configuration options here, such as async logging, display, etc.
    # A list of functions to process spans before export.
    span_processors: Optional[list[Callable]] = None

    def __post_init__(self):
        self.span_processors = validate_span_processors(self.span_processors)


# Global configuration instance for tracing
_MLFLOW_TRACING_CONFIG = TracingConfig()


class TracingConfigContext:
    """Context manager for temporary tracing configuration changes."""

    def __init__(self, config_updates: dict[str, Any]):
        self.config_updates = config_updates
        self.previous_config = None

        # Save the config state before applying any changes
        self.previous_config = deepcopy(_MLFLOW_TRACING_CONFIG)

        # Apply changes immediately for function-style usage
        for key, value in self.config_updates.items():
            setattr(_MLFLOW_TRACING_CONFIG, key, value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _MLFLOW_TRACING_CONFIG
        _MLFLOW_TRACING_CONFIG = self.previous_config


def get_config() -> TracingConfig:
    """
    Get the current tracing configuration.

    Returns:
        The current TracingConfig instance.
    """
    return _MLFLOW_TRACING_CONFIG


def reset_config():
    """
    Reset the tracing configuration to defaults.
    """
    global _MLFLOW_TRACING_CONFIG
    _MLFLOW_TRACING_CONFIG = TracingConfig()


@experimental(version="3.2.0")
def configure(
    span_processors: Optional[list[Callable]] = None,
) -> TracingConfigContext:
    """
    Configure MLflow tracing. Can be used as function or context manager.

    Only updates explicitly provided arguments, leaving others unchanged.

    Args:
        span_processors: List of functions to process spans before export.
            This is helpful for filtering/masking particular attributes
            from the span to prevent sensitive data from being logged
            or for reducing the size of the span. Each function receives
            a LiveSpan object. When multiple functions are provided,
            they are applied sequentially in the order they are provided.

    Returns:
        TracingConfigContext when used as context manager, None otherwise

    Examples:

        .. code-block:: python

            # Permanent configuration change
            mlflow.tracing.configure(span_processors=[pii_filter])

            # Temporary configuration change
            with mlflow.tracing.configure(span_processors=[pii_filter]):
                # PII filtering enabled only in this block
                pass
    """
    # Collect only the arguments that were explicitly provided
    config_updates = {}
    if span_processors is not None:
        config_updates["span_processors"] = span_processors

    # Return TracingConfigContext which handles both function and context manager usage
    return TracingConfigContext(config_updates)

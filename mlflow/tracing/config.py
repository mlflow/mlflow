from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from mlflow.tracing.utils.processor import validate_span_processors
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from mlflow.entities.span import LiveSpan


@dataclass
class TracingConfig:
    """Configuration for MLflow tracing behavior."""

    # TODO: Move more configuration options here, such as async logging, display, etc.
    # A list of functions to process spans before export.
    span_processors: list[Callable[["LiveSpan"], None]] = field(default_factory=list)

    def __post_init__(self):
        self.span_processors = validate_span_processors(self.span_processors)


# Global configuration instance for tracing
_MLFLOW_TRACING_CONFIG = TracingConfig()


class TracingConfigContext:
    """Context manager for temporary tracing configuration changes."""

    def __init__(self, config_updates: dict[str, Any]):
        self.config_updates = config_updates
        # Create a shallow copy of the current config
        self.previous_config = replace(_MLFLOW_TRACING_CONFIG)

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
    span_processors: list[Callable[["LiveSpan"], None]] | None = None,
) -> TracingConfigContext:
    """
    Configure MLflow tracing. Can be used as function or context manager.

    Only updates explicitly provided arguments, leaving others unchanged.

    Args:
        span_processors: List of functions to process spans before export.
            This is helpful for filtering/masking particular attributes from the span to
            prevent sensitive data from being logged or for reducing the size of the span.
            Each function must accept a single argument of type LiveSpan and should not
            return any value. When multiple functions are provided, they are applied
            sequentially in the order they are provided.

    Returns:
        TracingConfigContext: Context manager for temporary configuration changes.
            When used as a function, the configuration changes persist.
            When used as a context manager, changes are reverted on exit.

    Examples:

        .. code-block:: python

            def pii_filter(span):
                \"\"\"Example PII filter that masks sensitive data in span attributes.\"\"\"
                # Mask sensitive inputs
                if inputs := span.inputs:
                    for key, value in inputs.items():
                        if "password" in key.lower() or "token" in key.lower():
                            span.set_inputs({**inputs, key: "[REDACTED]"})

                # Mask sensitive outputs
                if outputs := span.outputs:
                    if isinstance(outputs, dict):
                        for key in outputs:
                            if "secret" in key.lower():
                                outputs[key] = "[REDACTED]"
                        span.set_outputs(outputs)

                # Mask sensitive attributes
                for attr_key in list(span.attributes.keys()):
                    if "api_key" in attr_key.lower():
                        span.set_attribute(attr_key, "[REDACTED]")

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

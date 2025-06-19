import logging
from typing import TYPE_CHECKING, Callable

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    from mlflow.entities.span import LiveSpan

_logger = logging.getLogger(__name__)


def apply_span_processors(span):
    """Apply configured span processors sequentially to the span."""
    from mlflow.tracing.config import get_config

    config = get_config()
    if not config.span_processors:
        return

    for processor in config.span_processors:
        try:
            result = processor(span)
            if result is not None:
                _logger.warning(
                    f"Span processor {processor.__name__} returned a non-null value, but it "
                    "will be ignored. Span processor should not return a value."
                )
        except Exception as e:
            _logger.warning(
                f"Span processor {processor.__name__} failed: {e}",
                exc_info=_logger.isEnabledFor(logging.DEBUG)
            )


def validate_span_processors(span_processors):
    """Validate that the span processor is a valid function."""
    span_processors = span_processors or []

    for span_processor in span_processors:
        if not callable(span_processor):
            raise MlflowException.invalid_parameter_value(
                "Span processor must be a callable function."
            )

        # Skip validation for builtin functions and partial functions that don't have __code__
        if not hasattr(span_processor, '__code__'):
            continue

        if (
            span_processor.__code__.co_argcount != 1
            or span_processor.__code__.co_varnames[0] != "span"
        ):
            raise MlflowException.invalid_parameter_value(
                "Span processor must take exactly one argument: `span`"
            )

    return span_processors


# Alias for backward compatibility
validate_span_processor = validate_span_processors

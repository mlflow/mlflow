import functools
import importlib
import logging
import warnings
from typing import Optional

_logger = logging.getLogger(__name__)


class LogDemotionFilter(logging.Filter):
    def __init__(self, module: str, message: str):
        super().__init__()
        self.module = module
        self.message = message

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == self.module and self.message in record.getMessage():
            record.levelno = logging.DEBUG  # Change the log level to DEBUG
            record.levelname = "DEBUG"

            # Check the log level for the logger is debug or not
            logger = logging.getLogger(self.module)
            return logger.isEnabledFor(logging.DEBUG)
        return True

    def __eq__(self, other):
        if isinstance(other, LogDemotionFilter):
            return self.module == other.module and self.message == other.message
        return False


def suppress_warning(module: str, message: str):
    """
    Convert the "Failed to detach context" log raised by the OpenTelemetry logger to DEBUG
    level so that it does not show up in the user's console.

    Args:
        module: The module name of the logger that raises the warning.
        message: The (part of) message in the log that needs to be demoted to DEBUG level
    """
    try:
        logger = getattr(importlib.import_module(module), "logger", None)
        log_filter = LogDemotionFilter(module, message)
        if logger and not any(f == log_filter for f in logger.filters):
            logger.addFilter(log_filter)
    except Exception as e:
        _logger.debug(f"Failed to suppress the warning for {module}", exc_info=e)
        raise


def request_id_backward_compatible(func):
    """
    A decorator to support backward compatibility for the `request_id` parameter,
    which is deprecated and replaced by the `trace_id` parameter in tracing APIs.

    This decorator will adds `request_id` to the function signature and issue
    a deprecation warning if `request_id` is used with non-null value.
    """

    @functools.wraps(func)
    def wrapper(*args, request_id: Optional[str] = None, **kwargs):
        if request_id is not None:
            warnings.warn(
                f"The request_id parameter is deprecated from the {func.__name__} API "
                "and will be removed in a future version. Please use the `trace_id` "
                "parameter instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )

            if kwargs.get("trace_id") is None:
                kwargs["trace_id"] = request_id

        return func(*args, **kwargs)

    return wrapper

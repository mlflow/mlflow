import logging
from typing import Any, Callable

from mlflow.bedrock import FLAVOR_NAME
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def capture_exception(logging_message: str):
    """
    A decorator to capture exceptions during a function execution.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                _logger.debug(logging_message)
                if _MLFLOW_TESTING:
                    raise

        return wrapper

    return decorator


def skip_if_trace_disabled(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to apply the function only if trace autologging is enabled.
    This decorator is used to skip the test if the trace autologging is disabled.
    """

    def wrapper(original, self, *args, **kwargs):
        config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
        if not config.log_traces:
            return original(self, *args, **kwargs)

        return func(original, self, *args, **kwargs)

    return wrapper

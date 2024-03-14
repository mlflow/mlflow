import inspect
import logging
from typing import Any, Dict

_logger = logging.getLogger(__name__)


def capture_function_input_args(func, args, kwargs) -> Dict[str, Any]:
    try:
        # Avoid capturing `self`
        func_signature = inspect.signature(func)
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        # Remove `self` from bound arguments if it exists
        if bound_arguments.arguments.get("self"):
            del bound_arguments.arguments["self"]

        return bound_arguments.arguments
    except Exception:
        _logger.warning(f"Failed to capture inputs for function {func.__name__}.")
        return {}


def get_caller_module() -> str:
    try:
        # The caller module is two frames up the stack (i.e. parent of the parent for this function)
        return inspect.getmodule(inspect.currentframe().f_back.__back).__name__
    except Exception:
        return "unknown_module"

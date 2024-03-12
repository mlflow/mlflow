import inspect

from typing import Any, Dict


def capture_function_input_args(func, args, kwargs) -> Dict[str, Any]:
    """
    """
    # Avoid capturing `self`
    func_signature = inspect.signature(func)
    bound_arguments = func_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    # Remove `self` from bound arguments if it exists
    if bound_arguments.arguments.get("self"):
        del bound_arguments.arguments["self"]

    return bound_arguments.arguments

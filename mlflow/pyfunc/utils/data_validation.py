import warnings
from functools import lru_cache, wraps
from typing import Any, Optional

from mlflow.models.signature import _extract_type_hints, _is_context_in_predict_function_signature
from mlflow.types.type_hints import (
    _convert_data_to_type_hint,
    _infer_schema_from_type_hint,
    _signature_cannot_be_inferred_from_type_hint,
    _validate_example_against_type_hint,
)
from mlflow.utils.annotations import filter_user_warnings_once


def pyfunc(func):
    """
    A decorator that forces data validation against type hint of the input data
    in the wrapped method. It is no-op if the type hint is not supported by MLflow.

    .. note::
        The function that applies this decorator must be a valid `predict` function
        of `mlflow.pyfunc.PythonModel`, or a callable that takes a single input.
    """

    type_hint = _get_type_hint_if_supported(func)
    if type_hint is not None:

        @wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _validate_model_input(func, args, kwargs, type_hint)
            return func(*args, **kwargs)

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    wrapper._is_pyfunc = True

    return wrapper


@lru_cache(maxsize=1)
def _get_type_hints(func):
    """
    Internal method to get type hints from the predict function signature.
    For PythonModel, the signature must be one of below:
        - predict(self, context, model_input, params=None)
        - predict(self, model_input, params=None)
    For callables, the function must contain only one input argument.

    Args:
        func: the predict function. Default is None, which means self.predict will be used.
    """
    # TODO: move function signature validation here
    # instead of the pyfunc wrapper
    if _is_context_in_predict_function_signature(func=func):
        return _extract_type_hints(func, input_arg_index=1)
    else:
        return _extract_type_hints(func, input_arg_index=0)


@lru_cache
@filter_user_warnings_once
def _get_type_hint_if_supported(func) -> Optional[type[Any]]:
    """
    Internal method to check if the predict function has type hints and if they are supported
    by MLflow.
    """
    type_hint = _get_type_hints(func).input
    if type_hint is not None:
        if _signature_cannot_be_inferred_from_type_hint(type_hint):
            return
        try:
            _infer_schema_from_type_hint(type_hint)
        except Exception as e:
            warnings.warn(
                "Type hint used in the model's predict function is not supported "
                f"for MLflow's schema validation. {e}",
                stacklevel=3,
            )
        else:
            return type_hint


def _validate_model_input(func, args, kwargs, type_hint):
    model_input = None
    input_pos = None
    if "model_input" in kwargs:
        model_input = kwargs["model_input"]
        input_pos = "kwargs"
    if _is_context_in_predict_function_signature(func=func):
        if len(args) >= 2:
            model_input = args[1]
            input_pos = 1
    elif len(args) >= 1:
        model_input = args[0]
        input_pos = 0
    if input_pos is not None:
        data = _convert_data_to_type_hint(model_input, type_hint)
        data = _validate_example_against_type_hint(data, type_hint)
        if input_pos == "kwargs":
            kwargs["model_input"] = data
        elif input_pos == 1:
            args = args[:1] + (data,) + args[2:]
        elif input_pos == 0:
            args = (data,) + args[1:]
    return args, kwargs

import inspect
import warnings
from functools import lru_cache, wraps
from typing import Any, Optional

from mlflow.models.signature import (
    _extract_type_hints,
    _is_context_in_predict_function_signature,
)
from mlflow.types.type_hints import (
    _convert_data_to_type_hint,
    _infer_schema_from_list_type_hint,
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
    return _wrap_predict_with_pyfunc(func, type_hint)


def _wrap_predict_with_pyfunc(func, type_hint):
    if type_hint is not None:
        model_input_index = _model_input_index_in_function_signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _validate_model_input(args, kwargs, model_input_index, type_hint)
            return func(*args, **kwargs)
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    wrapper._is_pyfunc = True
    return wrapper


@lru_cache
@filter_user_warnings_once
def _get_type_hint_if_supported(func) -> Optional[type[Any]]:
    """
    Internal method to check if the predict function has type hints and if they are supported
    by MLflow.
    For PythonModel, the signature must be one of below:
        - predict(self, context, model_input, params=None)
        - predict(self, model_input, params=None)
    For callables, the function must contain only one input argument.
    """
    # TODO: move function signature validation here
    # instead of the pyfunc wrapper
    if _is_context_in_predict_function_signature(func=func):
        type_hint = _extract_type_hints(func, input_arg_index=1).input
    else:
        type_hint = _extract_type_hints(func, input_arg_index=0).input
    if type_hint is not None:
        if _signature_cannot_be_inferred_from_type_hint(type_hint):
            return
        try:
            _infer_schema_from_list_type_hint(type_hint)
        except Exception as e:
            warnings.warn(
                "Type hint used in the model's predict function is not supported "
                f"for MLflow's schema validation. {e}. "
                "To enable validation for the input data, specify the input example "
                "or model signature when logging the model.",
                stacklevel=3,
            )
        else:
            return type_hint


def _model_input_index_in_function_signature(func):
    parameters = inspect.signature(func).parameters
    # we need to exclude the first argument if "self" is in the parameters
    index = 1 if "self" in parameters else 0
    if _is_context_in_predict_function_signature(parameters=parameters):
        index += 1
    return index


def _validate_model_input(args, kwargs, model_input_index_in_sig, type_hint):
    model_input = None
    input_pos = None
    if "model_input" in kwargs:
        model_input = kwargs["model_input"]
        input_pos = "kwargs"
    elif len(args) >= model_input_index_in_sig + 1:
        model_input = args[model_input_index_in_sig]
        input_pos = model_input_index_in_sig
    if input_pos is not None:
        data = _convert_data_to_type_hint(model_input, type_hint)
        data = _validate_example_against_type_hint(data, type_hint)
        if input_pos == "kwargs":
            kwargs["model_input"] = data
        else:
            args = args[:input_pos] + (data,) + args[input_pos + 1 :]
    return args, kwargs

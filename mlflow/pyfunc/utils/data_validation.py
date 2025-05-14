import inspect
import warnings
from functools import lru_cache, wraps
from typing import Any, NamedTuple, Optional

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.models.signature import (
    _extract_type_hints,
    _is_context_in_predict_function_signature,
)
from mlflow.types.type_hints import (
    InvalidTypeHintException,
    _convert_data_to_type_hint,
    _infer_schema_from_list_type_hint,
    _is_type_hint_from_example,
    _signature_cannot_be_inferred_from_type_hint,
    _validate_data_against_type_hint,
    model_validate,
)
from mlflow.utils.annotations import filter_user_warnings_once
from mlflow.utils.warnings_utils import color_warning

_INVALID_SIGNATURE_ERROR_MSG = (
    "Model's `{func_name}` method contains invalid parameters: {invalid_params}. "
    "Only the following parameter names are allowed: context, model_input, and params. "
    "Note that invalid parameters will no longer be permitted in future versions."
)


class FuncInfo(NamedTuple):
    input_type_hint: Optional[type[Any]]
    input_param_name: str


def pyfunc(func):
    """
    A decorator that forces data validation against type hint of the input data
    in the wrapped method. It is no-op if the type hint is not supported by MLflow.

    .. note::
        The function that applies this decorator must be a valid `predict` function
        of `mlflow.pyfunc.PythonModel`, or a callable that takes a single input.
    """

    func_info = _get_func_info_if_type_hint_supported(func)
    return _wrap_predict_with_pyfunc(func, func_info)


def _wrap_predict_with_pyfunc(func, func_info: Optional[FuncInfo]):
    if func_info is not None:
        model_input_index = _model_input_index_in_function_signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                args, kwargs = _validate_model_input(
                    args,
                    kwargs,
                    model_input_index,
                    func_info.input_type_hint,
                    func_info.input_param_name,
                )
            except Exception as e:
                if isinstance(e, MlflowException):
                    raise e
                raise MlflowException(
                    "Failed to validate the input data against the type hint "
                    f"`{func_info.input_type_hint}`. Error: {e}"
                )
            return func(*args, **kwargs)
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

    wrapper._is_pyfunc = True
    return wrapper


def wrap_non_list_predict_pydantic(func, input_pydantic_model, validation_error_msg, unpack=False):
    """
    Used by MLflow defined subclasses of PythonModel that have non-list a pydantic model as input.
    Takes in a dict input, validates it against `input_pydantic_model`, and then creates
    the pydantic model.

    If `unpack` is True, the validated dict is parsed into the function arguments.
    Otherwise, the whole pydantic object is passed to the function.

    Args:
        func: The predict/predict_stream method of the PythonModel subclass.
        input_pydantic_model: The pydantic model that the input should be validated against.
        validation_error_msg: The error message to raise if the dict input fails to validate.
        unpack: Whether to unpack the validated dict into the function arguments. Defaults to False.

    Raises:
        MlflowException: If the input fails to validate against the pydantic model.

    Returns:
        A function that can take either a dict input or a pydantic object as input.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            try:
                model_validate(input_pydantic_model, args[0])
                pydantic_obj = input_pydantic_model(**args[0])
            except pydantic.ValidationError as e:
                raise MlflowException(
                    f"{validation_error_msg} Pydantic validation error: {e}"
                ) from e
            else:
                if unpack:
                    param_names = inspect.signature(func).parameters.keys() - {"self"}
                    kwargs = {k: getattr(pydantic_obj, k) for k in param_names}
                    return func(self, **kwargs)
                else:
                    return func(self, pydantic_obj)
        else:
            # Before logging, this is equivalent to the behavior from the raw predict method
            # After logging, signature enforcement happens in the _convert_input method
            # of the wrapper class
            return func(self, *args, **kwargs)

    wrapper._is_pyfunc = True
    return wrapper


def _check_func_signature(func, func_name) -> list[str]:
    parameters = inspect.signature(func).parameters
    param_names = [name for name in parameters.keys() if name != "self"]
    if invalid_params := set(param_names) - {"self", "context", "model_input", "params"}:
        warnings.warn(
            _INVALID_SIGNATURE_ERROR_MSG.format(func_name=func_name, invalid_params=invalid_params),
            FutureWarning,
            stacklevel=2,
        )
    return param_names


@lru_cache
@filter_user_warnings_once
def _get_func_info_if_type_hint_supported(func) -> Optional[FuncInfo]:
    """
    Internal method to check if the predict function has type hints and if they are supported
    by MLflow.
    For PythonModel, the signature must be one of below:
        - predict(self, context, model_input, params=None)
        - predict(self, model_input, params=None)
    For callables, the function must contain only one input argument.
    """
    param_names = _check_func_signature(func, "predict")
    input_arg_index = 1 if _is_context_in_predict_function_signature(func=func) else 0
    type_hint = _extract_type_hints(func, input_arg_index=input_arg_index).input
    input_param_name = param_names[input_arg_index]
    if type_hint is not None:
        if _signature_cannot_be_inferred_from_type_hint(type_hint) or _is_type_hint_from_example(
            type_hint
        ):
            return
        try:
            _infer_schema_from_list_type_hint(type_hint)
        except InvalidTypeHintException as e:
            raise MlflowException(
                f"{e.message} To disable data validation, remove the type hint from the "
                "predict function. Otherwise, fix the type hint."
            )
        # catch other exceptions to avoid breaking model usage
        except Exception as e:
            color_warning(
                message="Type hint used in the model's predict function is not supported "
                f"for MLflow's schema validation. {e} "
                "Remove the type hint to disable this warning. "
                "To enable validation for the input data, specify input example "
                "or model signature when logging the model. ",
                category=UserWarning,
                stacklevel=3,
                color="red",
            )
        else:
            return FuncInfo(input_type_hint=type_hint, input_param_name=input_param_name)
    else:
        color_warning(
            "Add type hints to the `predict` method to enable data validation "
            "and automatic signature inference during model logging. "
            "Check https://mlflow.org/docs/latest/model/python_model.html#type-hint-usage-in-pythonmodel"
            " for more details.",
            stacklevel=1,
            color="yellow",
            category=UserWarning,
        )


def _model_input_index_in_function_signature(func):
    parameters = inspect.signature(func).parameters
    # we need to exclude the first argument if "self" is in the parameters
    index = 1 if "self" in parameters else 0
    if _is_context_in_predict_function_signature(parameters=parameters):
        index += 1
    return index


def _validate_model_input(
    args, kwargs, model_input_index_in_sig, type_hint, model_input_param_name
):
    model_input = None
    input_pos = None
    if model_input_param_name in kwargs:
        model_input = kwargs[model_input_param_name]
        input_pos = "kwargs"
    elif len(args) >= model_input_index_in_sig + 1:
        model_input = args[model_input_index_in_sig]
        input_pos = model_input_index_in_sig
    if input_pos is not None:
        data = _convert_data_to_type_hint(model_input, type_hint)
        data = _validate_data_against_type_hint(data, type_hint)
        if input_pos == "kwargs":
            kwargs[model_input_param_name] = data
        else:
            args = args[:input_pos] + (data,) + args[input_pos + 1 :]
    return args, kwargs

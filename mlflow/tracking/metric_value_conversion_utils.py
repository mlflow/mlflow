import sys

from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException


def _is_module_imported(module_name: str) -> bool:
    return module_name in sys.modules


def _try_get_item(x):
    try:
        return x.item()
    except Exception as e:
        raise MlflowException(
            f"Failed to convert metric value to float: {e}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _converter_requires(module_name: str):
    """Wrapper function that checks if specified `module_name`
    is already imported before invoking wrapped function."""

    def decorator(func):
        def wrapper(x):
            if not _is_module_imported(module_name):
                return x

            return func(x)

        return wrapper

    return decorator


def convert_metric_value_to_float_if_possible(x) -> float:
    if x is None or type(x) == float:
        return x

    converter_fns_to_try = [
        convert_metric_value_to_float_if_ndarray,
        convert_metric_value_to_float_if_tensorflow_tensor,
        convert_metric_value_to_float_if_torch_tensor,
    ]

    for converter_fn in converter_fns_to_try:
        possible_float = converter_fn(x)
        if type(possible_float) == float:
            return possible_float

    try:
        return float(x)
    except ValueError:
        return x  # let backend handle conversion if possible


@_converter_requires("numpy")
def convert_metric_value_to_float_if_ndarray(x):
    import numpy as np

    if isinstance(x, np.ndarray):
        return float(_try_get_item(x))

    return x


@_converter_requires("torch")
def convert_metric_value_to_float_if_torch_tensor(x):
    import torch

    if isinstance(x, torch.Tensor):
        extracted_tensor_val = x.detach().cpu()

        return float(_try_get_item(extracted_tensor_val))

    return x


@_converter_requires("tensorflow")
def convert_metric_value_to_float_if_tensorflow_tensor(x):
    import tensorflow as tf

    if isinstance(x, tf.Tensor):
        try:
            return float(x)
        except Exception as e:
            raise MlflowException(
                f"Failed to convert metric value to float: {e!r}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    return x

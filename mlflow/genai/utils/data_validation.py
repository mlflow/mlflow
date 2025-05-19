import inspect
import logging
from typing import Any, Callable

from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import trace_disabled

_logger = logging.getLogger(__name__)


def check_model_prediction(predict_fn: Callable, sample_input: Any) -> bool:
    """
    Validate if the predict function executes properly with the provided input.

    Args:
        predict_fn: The predict function to be evaluated.
        sample_input: A sample input to the model.
    """
    _logger.info("Testing model prediction with the first sample in the dataset.")

    # Wrap the function to add a decorator for disabling tracing
    @trace_disabled
    def _check():
        predict_fn(**sample_input)

    try:
        _check()
    except Exception as e:
        # Check input format and raise friendly message for typical error patterns
        _validate_function_and_input_compatibility(predict_fn, sample_input, e)
        _logger.debug(f"Failed to run predict_fn with input: {sample_input}", exc_info=e)


def _validate_function_and_input_compatibility(
    predict_fn: Callable, sample_input: dict[str, Any], e: Exception
) -> Callable:
    """
    Validate the data format in the input column against the predict_fn.

    The input column must contain a dictionary of field names and values. When the
    predict_fn is provided, the field names must match the arguments of the predict_fn.
    """
    params = inspect.signature(predict_fn).parameters
    if not params:
        raise MlflowException.invalid_parameter_value(
            "`predict_fn` must accept at least one argument."
        ) from e

    # Check for *args-style parameters which aren't supported
    if _has_variable_positional_arguments(params):
        _raise_args_not_supported_error(e)

    # Check if input keys match function parameters
    if not _has_required_keyword_arguments(params, sample_input.keys()):
        _raise_input_mismatch_error(params, e)

    # For other errors, show a generic error message
    raise MlflowException.invalid_parameter_value(
        "Failed to run the prediction function specified in the `predict_fn` "
        "parameter. Please make sure that the input dictionary contains the "
        f"keys that match with the `predict_fn` parameters. Error: {e}\n\n"
    ) from e


def _has_variable_positional_arguments(params: inspect.Signature) -> bool:
    """Check if the function has variable positional arguments."""
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())


def _has_required_keyword_arguments(params: inspect.Signature, required_args: list[str]) -> bool:
    """Check if the function accepts the specified keyword arguments."""
    func_args = []

    for name, param in params.items():
        # If the function has **kwargs, it accepts all keyword arguments
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True

        func_args.append(name)

    # Required argument must be a subset of the function's arguments
    return set(required_args) <= set(func_args)


def _raise_args_not_supported_error(e: Exception):
    """Raise an error for functions using *args which aren't supported."""
    code_sample = """```python
def predict_fn(param1, param2):
    # Invoke the original predict function with positional arguments
    return fn(param1, param2)

data = [
    {
        "inputs": {
            "param1": "value1",
            "param2": "value2",
        }
    }
]

mlflow.genai.evaluate(predict_fn=predict_fn, data=data, ...)
```
"""

    raise MlflowException.invalid_parameter_value(
        "The `predict_fn` has dynamic positional arguments (e.g. `*args`), "
        "so it cannot be used as a `predict_fn`. Please wrap it into another "
        "function that accepts explicit keyword arguments.\n"
        f"Example:\n\n{code_sample}\n"
    ) from e


def _raise_input_mismatch_error(params: inspect.Signature, e: Exception):
    """Raise an error when input keys don't match function parameters."""
    param_names = list(params.keys())
    input_example = {arg: f"value{i + 1}" for i, arg in enumerate(param_names[:3])}

    if len(param_names) > 3:
        input_example["..."] = "..."

    code_sample = ["```python"]
    code_sample.append("data = [")
    code_sample.append("    {")
    code_sample.append('        "inputs": {')
    for k, v in input_example.items():
        code_sample.append(f'            "{k}": "{v}",')
    code_sample.append("        }")
    code_sample.append("    }")
    code_sample.append("]")
    code_sample.append("```")
    code_sample = "\n".join(code_sample)

    raise MlflowException.invalid_parameter_value(
        "The `inputs` column must be a dictionary with the parameter names of "
        f"the `predict_fn` as keys. It seems the specified keys do not match "
        f"with the `predict_fn`'s arguments. Correct example:\n\n{code_sample}"
    ) from e

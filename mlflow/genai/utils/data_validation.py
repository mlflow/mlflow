import inspect
from typing import TYPE_CHECKING, Callable, Optional

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    import pandas as pd


def validate_inputs_column_format(
    inputs: "pd.Series", predict_fn: Optional[Callable] = None
) -> Callable:
    """
    Validate the data format in the input column against the predict_fn.

    The input column must contain a dictionary of field names and values. When the
    predict_fn is provided, the field names must match the arguments of the predict_fn.
    """
    # Check if the predict_fn has dynamic positional arguments (e.g. *args)
    # MLflow doesn't support such function as predict_fn, so instruct users to wrap
    # it into another function with explicit keyword arguments.
    if _has_dynamic_positional_arguments(predict_fn):
        code_example = _construct_code_example(
            predict_wrapper_body=(
                "    # Invoke the original predict function with positional arguments\n"
                f"    return {predict_fn.__name__}(param1, param2)\n"
            ),
            predict_wrapper_params=["param1", "param2"],
            input_example={"param1": "value1", "param2": "value2"},
        )
        raise MlflowException.invalid_parameter_value(
            "The `predict_fn` has dynamic positional arguments (e.g. `*args`), "
            "so it cannot be used as a `predict_fn`. Please wrap it into another "
            "function that accepts explicit keyword arguments.\n"
            f"Example:\n\n{code_example}\n"
        )

    if predict_fn and not _get_arguments(predict_fn):
        raise MlflowException.invalid_parameter_value(
            "`predict_fn` must accept at least one argument."
        )

    sample_input = inputs.iloc[0]
    if not isinstance(sample_input, dict):
        # If the predict_fn is provided, use its first argument in th code example
        sample_key = _get_arguments(predict_fn)[0] if predict_fn else "query"
        # If the input value is string, use it in the code example
        sample_value = sample_input if isinstance(sample_input, str) else "What is MLflow?"
        input_example = {sample_key: sample_value}

        if predict_fn:
            code_example = _construct_code_example(
                predict_fn_name=predict_fn.__name__,
                input_example=input_example,
            )
            raise MlflowException.invalid_parameter_value(
                "The 'inputs' column must be a dictionary with the parameter names of "
                f"the `predict_fn` as keys. For example:\n{code_example}"
            )
        else:
            code_example = _construct_code_example(input_example=input_example)
            raise MlflowException.invalid_parameter_value(
                "The 'inputs' column must be a dictionary of field names and values."
                f"For example:\n{code_example}"
            )


def _has_dynamic_positional_arguments(fn: Optional[Callable] = None) -> bool:
    """Check if the function has dynamic positional arguments (e.g. *args)."""
    if fn is None:
        return False
    sig = inspect.signature(fn)
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())


def _get_arguments(fn: Callable) -> list[str]:
    """Get the arguments of the function."""
    sig = inspect.signature(fn)
    return [p.name for p in sig.parameters.values()]


def _construct_code_example(
    predict_fn_name: Optional[str] = None,
    predict_wrapper_body: Optional[str] = None,
    predict_wrapper_params: Optional[list[str]] = None,
    input_example: Optional[dict[str, str]] = None,
) -> str:
    """
    Construct a code example for the predict_fn.
    """
    code = ["```python"]

    if predict_wrapper_body:
        params = ", ".join(predict_wrapper_params)
        code.append(f"def predict_fn({params}):")
        code.append(predict_wrapper_body)

    code.append("data = [")
    code.append("    {")
    code.append('        "inputs": {')
    for k, v in input_example.items():
        code.append(f'            "{k}": "{v}",')
    code.append("        }")
    code.append("    }")
    code.append("]\n")

    if predict_wrapper_body:
        code.append("mlflow.genai.evaluate(predict_fn=predict_fn, data=data, ...)")
    elif predict_fn_name:
        code.append(f"mlflow.genai.evaluate(predict_fn={predict_fn_name}, data=data, ...)")
    else:
        code.append("mlflow.genai.evaluate(data=data, scorers=...)")

    code.append("```")
    return "\n".join(code)

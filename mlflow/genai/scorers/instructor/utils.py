from __future__ import annotations

import json
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    parse_outputs_to_str,
    resolve_outputs_from_trace,
)

_INSTRUCTOR_NOT_INSTALLED_ERROR_MESSAGE = (
    "Instructor scorers require the `instructor` package. Install it with: `pip install instructor`"
)


def check_instructor_installed():
    try:
        import instructor  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(_INSTRUCTOR_NOT_INSTALLED_ERROR_MESSAGE)


def resolve_output_for_validation(
    outputs: Any,
    trace: Trace | None,
) -> Any:
    """
    Resolve output from either direct outputs or trace.

    Args:
        outputs: Direct output value
        trace: MLflow trace for evaluation

    Returns:
        Resolved output value for validation
    """
    if trace:
        outputs = resolve_outputs_from_trace(outputs, trace)
    return outputs


def output_to_dict(outputs: Any) -> dict[str, Any]:
    """
    Convert output to a dictionary for Pydantic validation.

    Args:
        outputs: Output value (dict, Pydantic model, JSON string, or other)

    Returns:
        Dictionary representation of the output

    Raises:
        MlflowException: If output cannot be converted to dict
    """
    if outputs is None:
        return {}

    if isinstance(outputs, dict):
        return outputs

    # Handle Pydantic models
    if hasattr(outputs, "model_dump"):
        return outputs.model_dump()

    # Handle objects with __dict__
    if hasattr(outputs, "__dict__"):
        return vars(outputs)

    # Handle JSON strings
    if isinstance(outputs, str):
        try:
            parsed = json.loads(outputs)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # If we can't convert to dict, return as string representation
    output_str = parse_outputs_to_str(outputs)
    raise MlflowException.invalid_parameter_value(
        f"Cannot convert output to dictionary for schema validation. "
        f"Output must be a dict, Pydantic model, or valid JSON string. "
        f"Got: {output_str[:100]}..."
    )


def get_schema_from_expectations(expectations: dict[str, Any] | None) -> type:
    """
    Extract Pydantic schema class from expectations.

    Args:
        expectations: Expectations dict containing 'schema' key

    Returns:
        Pydantic model class

    Raises:
        MlflowException: If schema is not provided or invalid
    """
    if not expectations or "schema" not in expectations:
        raise MlflowException.invalid_parameter_value(
            "Instructor scorers require 'schema' in expectations. "
            "Provide a Pydantic model class: expectations={'schema': YourModel}"
        )

    schema = expectations["schema"]

    # Validate it's a Pydantic model class
    from pydantic import BaseModel

    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise MlflowException.invalid_parameter_value(
            f"'schema' must be a Pydantic model class (subclass of BaseModel). "
            f"Got: {type(schema).__name__}"
        )

    return schema

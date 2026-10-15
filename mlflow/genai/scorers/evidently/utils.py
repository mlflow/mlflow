from __future__ import annotations

from typing import Any

import pandas as pd

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)


def check_evidently_installed():
    try:
        import evidently  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Evidently scorers require the `evidently` package. "
            "Install it with: `pip install evidently`"
        )


def map_scorer_inputs_to_dataframe(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert MLflow scorer inputs to pandas DataFrames for Evidently.

    Returns a tuple of (current_data, reference_data). The reference_data
    may be None if no expectations are provided.
    """
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)

    # Build current data from inputs/outputs
    current_dict: dict[str, Any] = {}
    if inputs is not None:
        if isinstance(inputs, dict):
            current_dict.update(inputs)
        else:
            raise MlflowException.invalid_parameter_value(
                "Evidently scorers require 'inputs' to be a dict mapping column names to values. "
                f"Got: {type(inputs).__name__}"
            )

    if outputs is not None:
        if isinstance(outputs, dict):
            current_dict.update(outputs)
        else:
            raise MlflowException.invalid_parameter_value(
                "Evidently scorers require 'outputs' to be a dict mapping column names to values. "
                f"Got: {type(outputs).__name__}"
            )

    if not current_dict:
        raise MlflowException.invalid_parameter_value(
            "Evidently scorers require either 'inputs' or 'outputs' to evaluate."
        )

    current_df = pd.DataFrame([current_dict])

    # Build reference data from expectations if provided
    reference_df = None
    if expectations is not None and isinstance(expectations, dict):
        ref_dict = expectations.get("reference_data")
        if ref_dict is not None:
            if isinstance(ref_dict, pd.DataFrame):
                reference_df = ref_dict
            elif isinstance(ref_dict, list):
                reference_df = pd.DataFrame(ref_dict)
            elif isinstance(ref_dict, dict):
                reference_df = pd.DataFrame([ref_dict])

    return current_df, reference_df

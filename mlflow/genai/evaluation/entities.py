"""Entities for evaluation."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.constant import InputDatasetColumn, ResultDataFrameColumn
from mlflow.genai.evaluation.context import get_context
from mlflow.genai.evaluation.utils import is_none_or_nan


@dataclass
class EvalItem:
    """Represents a row in the evaluation dataset."""

    """Unique identifier for the eval item."""
    request_id: str

    """Raw input to the model/application when `evaluate` is called."""
    inputs: dict[str, Any]

    """Raw output from the model/application."""
    outputs: Any

    """Expectations from the eval item."""
    expectations: dict[str, Any]

    """Trace of the model invocation."""
    trace: Trace | None = None

    """Error message if the model invocation fails."""
    error_message: str | None = None

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any]) -> "EvalItem":
        """
        Create an EvalItem from a row of input Pandas Dataframe row.
        """
        inputs = cls._parse_inputs(row.get(InputDatasetColumn.INPUTS))
        outputs = row.get(InputDatasetColumn.OUTPUTS)

        # Get the request ID from the row, or generate a new unique ID if not present.
        request_id = row.get(InputDatasetColumn.REQUEST_ID)
        if is_none_or_nan(request_id):
            request_id = hashlib.sha256(str(inputs).encode()).hexdigest()

        # Extract trace column from the dataset.
        trace = row.get(InputDatasetColumn.TRACE)
        if is_none_or_nan(trace):
            trace = None
        else:
            trace = trace if isinstance(trace, Trace) else Trace.from_json(trace)

        # Extract expectations column from the dataset.
        expectations = row.get(InputDatasetColumn.EXPECTATIONS, {})

        return cls(
            request_id=request_id,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

    @classmethod
    def _parse_inputs(cls, data: str | dict[str, Any]) -> dict[str, Any]:
        # The inputs can be either a dictionary or JSON-serialized version of it.
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):  # JSON-serialized string
            try:
                return json.loads(data)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse inputs as JSON."
                ) from e

        raise MlflowException.invalid_parameter_value(
            f"inputs must be a dictionary or JSON serializable: {type(data)}"
        )

    def get_expectation_assessments(self) -> list[Expectation]:
        """Get the expectations as a list of Expectation objects."""
        expectations = []
        for name, value in self.expectations.items():
            source_id = get_context().get_user_name()
            expectations.append(
                Expectation(
                    trace_id=self.trace.info.trace_id,
                    name=name,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.HUMAN,
                        source_id=source_id or "unknown",
                    ),
                    value=value,
                )
            )
        return expectations

    def to_dict(self) -> dict[str, Any]:
        inputs = {
            ResultDataFrameColumn.REQUEST_ID: self.request_id,
            ResultDataFrameColumn.INPUTS: self.inputs,
            ResultDataFrameColumn.OUTPUTS: self.outputs,
            ResultDataFrameColumn.TRACE: self.trace.to_json(),
            ResultDataFrameColumn.EXPECTATIONS: self.expectations,
            ResultDataFrameColumn.ERROR_MESSAGE: self.error_message,
        }
        return {k: v for k, v in inputs.items() if v is not None}


@dataclass
class EvalResult:
    """Holds the result of the evaluation for an eval item."""

    eval_item: EvalItem
    """A collection of assessments from scorers."""
    assessments: list[Feedback] = field(default_factory=list)
    """Error message encountered in processing the eval item."""
    eval_error: str | None = None

    def to_pd_series(self) -> pd.Series:
        """Converts the EvalResult to a flattened pd.Series."""
        inputs = self.eval_item.to_dict()
        assessments = self.get_assessments_dict()

        # Merge dictionaries and convert to pd.Series
        return pd.Series(inputs | assessments)

    def get_assessments_dict(self) -> dict[str, Any]:
        result = {}
        for assessment in self.assessments:
            if not isinstance(assessment, Feedback):
                continue

            result |= {
                f"{assessment.name}/value": assessment.value,
                f"{assessment.name}/rationale": assessment.rationale,
                f"{assessment.name}/error_message": assessment.error_message,
                f"{assessment.name}/error_code": assessment.error_code,
            }

        return result


@dataclass
class EvaluationResult:
    run_id: str
    metrics: dict[str, float]
    result_df: pd.DataFrame

    def __repr__(self) -> str:
        metrics_str = "\n    ".join([f"{k}: {v}" for k, v in self.metrics.items()])
        return (
            "EvaluationResult(\n"
            f"  run_id: {self.run_id}\n"
            "  metrics:\n"
            f"    {metrics_str}\n"
            f"  result_df: [{len(self.result_df)} rows x {len(self.result_df.columns)} cols]\n"
            ")"
        )

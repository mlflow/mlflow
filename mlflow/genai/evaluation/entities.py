"""Entities for evaluation."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.dataset_record_source import DatasetRecordSource
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.constant import InputDatasetColumn, ResultDataFrameColumn
from mlflow.genai.evaluation.context import get_context
from mlflow.genai.evaluation.utils import is_none_or_nan


def _assertion_outcome(
    value: Any,
    error_msg: str | None,
    rationale: str | None,
    pass_if: Callable[[Any], bool] | None,
) -> tuple[bool, str | None]:
    """Decide whether one scorer value passes, plus a failure detail."""
    if error_msg is not None:
        return False, error_msg
    if pass_if is not None:
        try:
            return bool(pass_if(value)), rationale or f"value={value!r}"
        except Exception as e:
            return False, f"pass_if raised {type(e).__name__}: {e}"
    # CategoricalRating is a StrEnum, so yes/no ratings arrive as strings.
    if isinstance(value, str):
        return value.strip().lower() == "yes", rationale or f"value={value!r}"
    # pd.api.types.is_bool also covers NumPy bool_ from the dataframe.
    if pd.api.types.is_bool(value):
        return bool(value), rationale or f"value={value!r}"
    hint = (
        f"returned {value!r}; assertions need a yes/no rating or a bool. "
        f"Declare pass_if=... on the scorer to define what counts as passing."
    )
    # Always append the hint; a rationale augments it rather than replacing it.
    return False, f"{rationale}; {hint}" if rationale else hint


@dataclass
class ScorerStat:
    """Statistics for a single scorer's invocations during evaluation.

    Tracks the total number of invocations and how many failed with errors.
    """

    failure_count: int = 0
    total_count: int = 0

    def record_invocation(self, *, failed: bool) -> None:
        """Record a scorer invocation."""
        self.total_count += 1
        if failed:
            self.failure_count += 1

    def merge(self, other: "ScorerStat") -> None:
        """Merge stats from another ScorerStat."""
        self.failure_count += other.failure_count
        self.total_count += other.total_count

    @property
    def has_failures(self) -> bool:
        """Check if this scorer had any failures."""
        return self.failure_count > 0


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

    """Tags from the eval item."""
    tags: dict[str, str] | None = None

    """Trace of the model invocation."""
    trace: Trace | None = None

    """Error message if the model invocation fails."""
    error_message: str | None = None

    """Source information for the eval item (e.g., from which trace it was created)."""
    source: DatasetRecordSource | None = None

    @classmethod
    def from_trace(cls, trace: Trace) -> "EvalItem":
        """
        Create an EvalItem from a Trace.

        Args:
            trace: The trace to create an EvalItem from.

        Returns:
            An EvalItem with the trace set and request_id from the trace.
        """
        return cls(
            request_id=trace.info.trace_id,
            inputs=None,
            outputs=None,
            expectations=None,
            trace=trace,
        )

    @classmethod
    def from_dataset_row(cls, row: dict[str, Any]) -> "EvalItem":
        """
        Create an EvalItem from a row of input Pandas Dataframe row.
        """
        if (inputs := row.get(InputDatasetColumn.INPUTS)) is not None:
            inputs = cls._parse_inputs(inputs)
        outputs = row.get(InputDatasetColumn.OUTPUTS)

        # Extract trace column from the dataset.
        trace = row.get(InputDatasetColumn.TRACE)
        if is_none_or_nan(trace):
            trace = None
        else:
            trace = trace if isinstance(trace, Trace) else Trace.from_json(trace)

        # Extract expectations column from the dataset.
        expectations = row.get(InputDatasetColumn.EXPECTATIONS, {})
        if is_none_or_nan(expectations):
            expectations = {}

        # Extract tags column from the dataset.
        tags = row.get(InputDatasetColumn.TAGS, {})

        # Extract source column from the dataset.
        source = row.get(InputDatasetColumn.SOURCE)
        if is_none_or_nan(source):
            source = None

        # Get the request ID from the row, or generate a new unique ID if not present.
        request_id = row.get(InputDatasetColumn.REQUEST_ID)
        if is_none_or_nan(request_id):
            hashable_strings = [
                str(x) for x in [inputs, outputs, trace, expectations] if x is not None
            ]
            # this should not happen, but added a check in case
            if not hashable_strings:
                raise MlflowException.invalid_parameter_value(
                    "Dataset row must contain at least one non-None value"
                )
            request_id = hashlib.sha256(str(hashable_strings[0]).encode()).hexdigest()

        return cls(
            request_id=request_id,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            tags=tags,
            trace=trace,
            source=source,
        )

    @classmethod
    def _parse_inputs(cls, data: str | dict[str, Any]) -> Any:
        # The inputs can be either a dictionary or JSON-serialized version of it.
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):  # JSON-serialized string
            try:
                return json.loads(data)
            except Exception:
                pass
        return data

    def get_expectation_assessments(self) -> list[Expectation]:
        """Get the expectations as a list of Expectation objects."""
        expectations = []
        for name, value in self.expectations.items():
            source_id = get_context().get_user_name()
            expectations.append(
                Expectation(
                    trace_id=self.trace.info.trace_id if self.trace else None,
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
            ResultDataFrameColumn.TRACE: self.trace.to_json() if self.trace else None,
            ResultDataFrameColumn.EXPECTATIONS: self.expectations,
            ResultDataFrameColumn.TAGS: self.tags,
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
    """Statistics for each scorer that ran on this eval item."""
    scorer_stats: dict[str, ScorerStat] = field(default_factory=dict)

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
    result_df: pd.DataFrame | None
    # Per-scorer ``pass_if`` predicates, keyed by scorer name. Populated by the
    # evaluation harness from the scorers that declare one. In-process only.
    pass_criteria: dict[str, Callable[[Any], bool]] = field(default_factory=dict)

    def __repr__(self) -> str:
        metrics_str = "\n    ".join([f"{k}: {v}" for k, v in self.metrics.items()])
        result_df_str = (
            f"{len(self.result_df)} rows x {len(self.result_df.columns)} cols"
            if self.result_df is not None
            else "None"
        )
        return (
            "EvaluationResult(\n"
            f"  run_id: {self.run_id}\n"
            "  metrics:\n"
            f"    {metrics_str}\n"
            f"  result_df: {result_df_str}\n"
            ")"
        )

    @property
    def passed(self) -> bool:
        """``True`` when every scorer passed for every row.

        A value passes when it is a ``yes`` rating (or ``"yes"``) or ``True``.
        Declare ``@scorer(pass_if=...)`` to define passing for other values.

        Usage::

            result = mlflow.genai.evaluate(...)
            assert result.passed, result.reason
        """
        return not self._failures()

    @property
    def reason(self) -> str:
        """Human-readable explanation of which scorers failed and why.

        Empty string when all scorers passed.
        """
        failures = self._failures()
        if not failures:
            return ""
        if len(failures) == 1:
            return failures[0]
        return f"{len(failures)} assertions failed:\n" + "\n".join(f"  - {f}" for f in failures)

    def _failures(self) -> list[str]:
        if self.result_df is None:
            return []

        value_cols = [c for c in self.result_df.columns if c.endswith("/value")]
        if not value_cols:
            return []

        failures: list[str] = []
        for _, row in self.result_df.iterrows():
            for col in value_cols:
                scorer_name = col.removesuffix("/value")
                value = row.get(col)
                error_msg = row.get(f"{scorer_name}/error_message")
                error_msg = None if is_none_or_nan(error_msg) else error_msg
                # Skip cells a scorer did not produce for this row.
                if error_msg is None and is_none_or_nan(value):
                    continue
                rationale = row.get(f"{scorer_name}/rationale")
                rationale = None if is_none_or_nan(rationale) else rationale
                passed, detail = _assertion_outcome(
                    value, error_msg, rationale, self.pass_criteria.get(scorer_name)
                )
                if not passed:
                    failures.append(f"{scorer_name}: {detail}" if detail else scorer_name)
        return failures

    # For backwards compatibility
    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        return {"eval_results": self.result_df} if self.result_df is not None else {}

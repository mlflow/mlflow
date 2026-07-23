"""
Evidently AI integration for MLflow.

This module provides integration with Evidently AI metrics, allowing them to be used
with MLflow's scorer interface for data quality monitoring and model performance evaluation.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.evidently import MissingValues, UniqueValues, get_scorer

    # Check for missing values
    scorer = MissingValues(column="feature_1")
    feedback = scorer(outputs={"feature_1": None})

    # Count unique values
    scorer = UniqueValues(column="category")
    feedback = scorer(outputs={"category": "A"})

    # Use factory function
    scorer = get_scorer("MissingValueCount", column="feature_1")
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.evidently.registry import get_metric_class
from mlflow.genai.scorers.evidently.utils import (
    check_evidently_installed,
    map_scorer_inputs_to_dataframe,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_FRAMEWORK_NAME = "evidently"


@experimental(version="3.12.0")
class EvidentlyScorer(Scorer):
    """Base class for Evidently AI metric scorers.

    Evidently AI metrics evaluate data quality, detect data drift, and measure
    model performance. This class wraps Evidently metrics to work with MLflow's
    scorer interface.

    Args:
        metric_name: Name of the Evidently metric (e.g., "MissingValueCount")
        **metric_kwargs: Additional arguments passed to the Evidently metric
    """

    _metric: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        **metric_kwargs: Any,
    ):
        check_evidently_installed()

        if metric_name is None:
            metric_name = getattr(self.__class__, "metric_name", None)
            if metric_name is None:
                raise ValueError("metric_name must be provided")

        super().__init__(name=metric_name)

        metric_class = get_metric_class(metric_name)
        self._metric = metric_class(**metric_kwargs)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def _raise_registration_not_supported(self, method_name: str):
        raise MlflowException.invalid_parameter_value(
            f"'{method_name}()' is not supported for third-party scorers like Evidently. "
            f"Third-party scorers cannot be registered, started, updated, or stopped. "
            f"Use them directly in mlflow.genai.evaluate() instead."
        )

    def register(self, **kwargs):
        self._raise_registration_not_supported("register")

    def start(self, **kwargs):
        self._raise_registration_not_supported("start")

    def update(self, **kwargs):
        self._raise_registration_not_supported("update")

    def stop(self, **kwargs):
        self._raise_registration_not_supported("stop")

    def align(self, **kwargs):
        raise MlflowException.invalid_parameter_value(
            "'align()' is not supported for third-party scorers like Evidently. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """Evaluate data using an Evidently metric.

        Args:
            inputs: The input data to evaluate
            outputs: The output data to evaluate
            expectations: Optional dict with "reference_data" key for drift detection
            trace: MLflow trace for evaluation
            session: List of MLflow traces for multi-turn/agentic evaluation

        Returns:
            Feedback object with metric result
        """
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.CODE,
            source_id=f"evidently/{self.name}",
        )

        current_df, reference_df = map_scorer_inputs_to_dataframe(
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

        try:
            from evidently import Dataset, Report

            report = Report([self._metric])
            run_kwargs: dict[str, Any] = {
                "current_data": Dataset.from_pandas(current_df),
            }
            if reference_df is not None:
                run_kwargs["reference_data"] = Dataset.from_pandas(reference_df)

            snapshot = report.run(**run_kwargs)
            result_dict = snapshot.dict()

            value = self._extract_value(result_dict)
            rationale = self._extract_rationale(result_dict)

            return Feedback(
                name=self.name,
                value=value,
                rationale=rationale,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME},
            )
        except Exception as e:
            _logger.error(f"Error evaluating with Evidently {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: _FRAMEWORK_NAME},
            )

    def _extract_value(self, result_dict: dict[str, Any]) -> float | None:
        """Extract the primary metric value from Evidently result dict."""
        metrics = result_dict.get("metrics", [])
        if not metrics:
            return None

        value = metrics[0].get("value")
        if not isinstance(value, dict):
            return None

        # MissingValueCount / UniqueValueCount return {"count": N, "share": M}
        if "count" in value:
            return float(value["count"])

        # Fallback: return first numeric value found
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)

        return None

    def _extract_rationale(self, result_dict: dict[str, Any]) -> str | None:
        """Extract a human-readable explanation from Evidently result dict."""
        metrics = result_dict.get("metrics", [])
        if not metrics:
            return None

        value = metrics[0].get("value")
        if not isinstance(value, dict):
            return None

        return "; ".join(f"{k}: {v}" for k, v in value.items()) or None


@experimental(version="3.12.0")
def get_scorer(
    metric_name: str,
    **metric_kwargs: Any,
) -> EvidentlyScorer:
    """Get an Evidently metric as an MLflow scorer.

    Args:
        metric_name: Name of the Evidently metric (e.g., "MissingValueCount", "UniqueValueCount")
        metric_kwargs: Additional keyword arguments to pass to the metric.

    Returns:
        EvidentlyScorer instance that can be called with MLflow's scorer interface

    Examples:
        .. code-block:: python

            scorer = get_scorer("MissingValueCount", column="feature_1")
            feedback = scorer(outputs={"feature_1": None})

            scorer = get_scorer("UniqueValueCount", column="category")
            feedback = scorer(outputs={"category": "A"})
    """
    return EvidentlyScorer(
        metric_name=metric_name,
        **metric_kwargs,
    )


@experimental(version="3.12.0")
class MissingValues(EvidentlyScorer):
    """Check for missing values in a specific column.

    Counts the number of missing (null/NaN) values in the specified column
    of the current data.

    Args:
        column: Name of the column to check for missing values

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.evidently import MissingValues

            scorer = MissingValues(column="feature_1")
            feedback = scorer(outputs={"feature_1": None})
    """

    metric_name: ClassVar[str] = "MissingValueCount"


@experimental(version="3.12.0")
class UniqueValues(EvidentlyScorer):
    """Count unique values in a specific column.

    Counts the number of distinct values in the specified column,
    useful for monitoring categorical feature cardinality changes.

    Args:
        column: Name of the column to check

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.evidently import UniqueValues

            scorer = UniqueValues(column="category")
            feedback = scorer(outputs={"category": "A"})
    """

    metric_name: ClassVar[str] = "UniqueValueCount"


__all__ = [
    "EvidentlyScorer",
    "get_scorer",
    "MissingValues",
    "UniqueValues",
]

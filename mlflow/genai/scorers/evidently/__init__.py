"""
Evidently AI integration for MLflow.

This module provides integration with Evidently AI metrics, allowing them to be used
with MLflow's scorer interface for data drift detection, data quality monitoring,
and model performance evaluation.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.evidently import ValueDrift, MissingValues, get_scorer

    # Detect data drift
    scorer = ValueDrift(column="feature_1")
    feedback = scorer(
        outputs={"feature_1": 0.5},
        expectations={"reference_data": [{"feature_1": 0.1}, {"feature_1": 0.2}]},
    )

    # Check for missing values
    scorer = MissingValues(column="feature_1")
    feedback = scorer(outputs={"feature_1": None})

    # Use factory function
    scorer = get_scorer("ValueDrift", column="feature_1")
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer
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
        metric_name: Name of the Evidently metric (e.g., "ValueDrift")
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

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """Evaluate data using an Evidently metric.

        Args:
            inputs: The input data to evaluate
            outputs: The output data to evaluate
            expectations: Optional dict with "reference_data" key for drift detection
            trace: MLflow trace for evaluation

        Returns:
            Feedback object with metric result
        """
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.CODE,
            source_id=f"evidently/{self.name}",
        )

        try:
            from evidently import Dataset, Report

            current_df, reference_df = map_scorer_inputs_to_dataframe(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

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

    def _extract_value(self, result_dict: dict) -> float | bool:
        """Extract the primary metric value from Evidently result dict."""
        metrics = result_dict.get("metrics", [])
        if not metrics:
            return 0.0

        value = metrics[0].get("value")
        if value is None:
            return 0.0

        # Scalar value (e.g., ValueDrift returns a p-value float)
        if isinstance(value, (int, float)):
            return float(value)

        # Dict value (e.g., MissingValueCount returns {"count": 1.0, "share": 0.33})
        if isinstance(value, dict):
            if "count" in value:
                return float(value["count"])
            # Fallback: return first numeric value found
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)

        return 0.0

    def _extract_rationale(self, result_dict: dict) -> str | None:
        """Extract a human-readable explanation from Evidently result dict."""
        metrics = result_dict.get("metrics", [])
        if not metrics:
            return None

        metric = metrics[0]
        value = metric.get("value")
        parts = []

        if isinstance(value, (int, float)):
            parts.append(f"Value: {value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                parts.append(f"{k}: {v}")

        config = metric.get("config")
        if isinstance(config, dict) and "stattest" in config:
            parts.append(f"Statistical test: {config['stattest']}")

        return "; ".join(parts) if parts else None


@experimental(version="3.12.0")
def get_scorer(
    metric_name: str,
    **metric_kwargs: Any,
) -> EvidentlyScorer:
    """Get an Evidently metric as an MLflow scorer.

    Args:
        metric_name: Name of the Evidently metric (e.g., "ValueDrift", "MissingValueCount")
        **metric_kwargs: Additional keyword arguments to pass to the metric.

    Returns:
        EvidentlyScorer instance that can be called with MLflow's scorer interface

    Examples:
        .. code-block:: python

            scorer = get_scorer("ValueDrift", column="feature_1")
            feedback = scorer(
                outputs={"feature_1": 0.5},
                expectations={"reference_data": [{"feature_1": 0.1}, {"feature_1": 0.2}]},
            )

            scorer = get_scorer("MissingValueCount", column="feature_1")
            feedback = scorer(outputs={"feature_1": None})
    """
    return EvidentlyScorer(
        metric_name=metric_name,
        **metric_kwargs,
    )


@experimental(version="3.12.0")
class ValueDrift(EvidentlyScorer):
    """Detect data drift for a specific column using statistical tests.

    Compares the distribution of values in current data against reference data
    to detect significant changes (drift).

    Args:
        column: Name of the column to check for drift

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.evidently import ValueDrift

            scorer = ValueDrift(column="prediction")
            feedback = scorer(
                outputs={"prediction": 0.9},
                expectations={"reference_data": [{"prediction": 0.1}, {"prediction": 0.2}]},
            )
    """

    metric_name: ClassVar[str] = "ValueDrift"


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
    "ValueDrift",
    "MissingValues",
    "UniqueValues",
]

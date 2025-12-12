"""
RAGAS integration for MLflow.

This module provides integration with RAGAS metrics, allowing them to be used
with MLflow's judge interface.

Example usage:
    >>> from mlflow.genai.scorers.ragas import get_judge
    >>> judge = get_judge("Faithfulness", model="openai:/gpt-4")
    >>> feedback = judge(
    ...     inputs="What is MLflow?", outputs="MLflow is a platform...", trace=trace
    ... )
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.ragas.models import create_ragas_model
from mlflow.genai.scorers.ragas.registry import get_metric_class, is_deterministic_metric
from mlflow.genai.scorers.ragas.utils import map_scorer_inputs_to_ragas_sample

_logger = logging.getLogger(__name__)


class RagasScorer(Scorer):
    """RAGAS metric scorer for MLflow."""

    _metric: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str,
        model: str = "databricks",
        **metric_kwargs,
    ):
        """
        Initialize a RAGAS metric scorer.

        Args:
            metric_name: Name of the RAGAS metric (e.g., "Faithfulness")
            model: Model URI in MLflow format for LLM-based metrics (default: "databricks")
            metric_kwargs: Additional metric-specific parameters
        """
        super().__init__(name=metric_name)

        metric_class = get_metric_class(metric_name)

        if is_deterministic_metric(metric_name):
            self._metric = metric_class(**metric_kwargs)
        else:
            ragas_llm = create_ragas_model(model)
            self._metric = metric_class(llm=ragas_llm, **metric_kwargs)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped RAGAS metric.

        Args:
            inputs: The input to evaluate
            outputs: The output to evaluate
            expectations: Expected values and context for evaluation
            trace: MLflow trace for evaluation

        Returns:
            Feedback object with score, rationale, and metadata
        """
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=f"ragas/{self.name}",
        )

        try:
            sample = map_scorer_inputs_to_ragas_sample(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

            if hasattr(self._metric, "single_turn_score"):
                result = self._metric.single_turn_score(sample)
            elif hasattr(self._metric, "score"):
                result = self._metric.score(sample)

            score = float(result)

            reason = None
            if hasattr(result, "reason"):
                reason = result.reason

            threshold = getattr(self._metric, "threshold", None)
            if threshold is None:
                threshold = 0.5
            success = score >= threshold

            return Feedback(
                name=self.name,
                value=CategoricalRating.YES if success else CategoricalRating.NO,
                rationale=reason,
                source=assessment_source,
                trace_id=trace.info.trace_id if trace else None,
                metadata={
                    "score": score,
                    "threshold": threshold,
                },
            )
        except Exception as e:
            _logger.error(f"Error evaluating RAGAS metric {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
            )


def get_judge(
    metric_name: str,
    model: str = "databricks",
    **metric_kwargs,
) -> RagasScorer:
    """
    Get a RAGAS metric as an MLflow judge.

    Args:
        metric_name: Name of the RAGAS metric (e.g., "Faithfulness")
        model: Model URI for LLM-based metrics (default: "databricks")
        metric_kwargs: Additional metric-specific parameters (e.g., threshold)

    Returns:
        RagasScorer instance that can be called with MLflow's judge interface

    Examples:
        >>> # LLM-based metric
        >>> judge = get_judge("Faithfulness", model="openai:/gpt-4")
        >>> feedback = judge(
        ...     inputs="What is MLflow?", outputs="MLflow is a platform...", trace=trace
        ... )

        >>> # Using trace
        >>> judge = get_judge("ContextPrecision", model="openai:/gpt-4")
        >>> feedback = judge(trace=trace)
    """
    return RagasScorer(
        metric_name=metric_name,
        model=model,
        **metric_kwargs,
    )


__all__ = [
    "RagasScorer",
    "get_judge",
]

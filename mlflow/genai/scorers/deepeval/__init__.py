"""
DeepEval integration for MLflow.

This module provides integration with DeepEval metrics, allowing them to be used
with MLflow's judge interface.

Example usage:
    >>> from mlflow.genai.scorers.deepeval import get_judge
    >>> judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
    >>> feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")
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
from mlflow.genai.scorers.deepeval.registry import get_metric_class
from mlflow.genai.scorers.deepeval.utils import (
    create_deepeval_model,
    map_mlflow_to_test_case,
)

_logger = logging.getLogger(__name__)


class DeepEvalScorer(Scorer):
    _metric: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str,
        model: str = "databricks",
        **metric_kwargs,
    ):
        """
        Initialize a DeepEval metric scorer.

        Args:
            metric_name: Name of the DeepEval metric (e.g., "answer_relevancy")
            model: Model URI in MLflow format (default: "databricks")
            metric_kwargs: Additional metric-specific parameters
        """
        super().__init__(name=metric_name)

        metric_class = get_metric_class(metric_name)
        deepeval_model = create_deepeval_model(model)

        self._metric = metric_class(
            model=deepeval_model,
            verbose_mode=False,
            async_mode=False,
            **metric_kwargs,
        )

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped DeepEval metric.

        Args:
            inputs: The input to evaluate
            outputs: The output to evaluate
            expectations: Expected values and context for evaluation
            trace: MLflow trace for evaluation

        Returns:
            Feedback object with pass/fail value, rationale, and score in metadata
        """
        test_case = map_mlflow_to_test_case(
            metric_name=self.name,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

        self._metric.measure(test_case)

        score = self._metric.score
        reason = self._metric.reason
        success = self._metric.is_successful()

        return Feedback(
            name=self.name,
            value=CategoricalRating.YES if success else CategoricalRating.NO,
            rationale=reason,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=f"deepeval/{self.name}",
            ),
            metadata={
                "score": score,
                "threshold": self._metric.threshold,
            },
        )


def get_judge(
    metric_name: str,
    model: str = "databricks",
    **metric_kwargs,
) -> DeepEvalScorer:
    """
    Get a DeepEval metric as an MLflow judge.

    Args:
        metric_name: Name of the DeepEval metric (e.g., "AnswerRelevancy", "Faithfulness")
        model: Model URI in MLflow format (default: "databricks")
        metric_kwargs: Additional metric-specific parameters (e.g., threshold, include_reason)

    Returns:
        DeepEvalScorer instance that can be called with MLflow's judge interface

    Examples:
        >>> judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
        >>> feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")

        >>> judge = get_judge("Faithfulness", model="openai:/gpt-4")
        >>> feedback = judge(trace=trace)
    """
    return DeepEvalScorer(
        metric_name=metric_name,
        model=model,
        **metric_kwargs,
    )


__all__ = [
    "DeepEvalScorer",
    "get_judge",
]

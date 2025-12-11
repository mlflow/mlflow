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
from mlflow.genai.scorers.deepeval.models import create_deepeval_model
from mlflow.genai.scorers.deepeval.registry import get_metric_class, is_deterministic_metric
from mlflow.genai.scorers.deepeval.utils import map_scorer_inputs_to_deepeval_test_case
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class DeepEvalScorer(Scorer):
    _metric: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str = "databricks",
        **metric_kwargs,
    ):
        """
        Initialize a DeepEval metric scorer.

        Args:
            metric_name: Name of the DeepEval metric (e.g., "AnswerRelevancy").
                If not provided, will use the class-level metric_name attribute.
            model: Model URI in MLflow format (default: "databricks")
            metric_kwargs: Additional metric-specific parameters
        """
        # Use class attribute if metric_name not provided
        if metric_name is None:
            metric_name = self.metric_name

        super().__init__(name=metric_name)

        metric_class = get_metric_class(metric_name)

        if is_deterministic_metric(metric_name):
            # Deterministic metrics don't need a model
            self._metric = metric_class(**metric_kwargs)
        else:
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
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=f"deepeval/{self.name}",
        )

        try:
            test_case = map_scorer_inputs_to_deepeval_test_case(
                metric_name=self.name,
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

            self._metric.measure(test_case, _show_indicator=False)

            score = self._metric.score
            reason = self._metric.reason
            success = self._metric.is_successful()

            return Feedback(
                name=self.name,
                value=CategoricalRating.YES if success else CategoricalRating.NO,
                rationale=reason,
                source=assessment_source,
                metadata={
                    "score": score,
                    "threshold": self._metric.threshold,
                },
            )
        except Exception as e:
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
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


# Import namespaced metric classes from scorers subdirectory
from mlflow.genai.scorers.deepeval.scorers import (
    AnswerRelevancy,
    ArgumentCorrectness,
    Bias,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
    ConversationCompleteness,
    ExactMatch,
    Faithfulness,
    GoalAccuracy,
    Hallucination,
    JsonCorrectness,
    KnowledgeRetention,
    Misuse,
    NonAdvice,
    PatternMatch,
    PIILeakage,
    PlanAdherence,
    PlanQuality,
    PromptAlignment,
    RoleAdherence,
    RoleViolation,
    StepEfficiency,
    Summarization,
    TaskCompletion,
    ToolCorrectness,
    ToolUse,
    TopicAdherence,
    Toxicity,
    TurnRelevancy,
)

__all__ = [
    # Core classes
    "DeepEvalScorer",
    "get_judge",
    # RAG metrics
    "AnswerRelevancy",
    "Faithfulness",
    "ContextualRecall",
    "ContextualPrecision",
    "ContextualRelevancy",
    # Agentic metrics
    "TaskCompletion",
    "ToolCorrectness",
    "ArgumentCorrectness",
    "StepEfficiency",
    "PlanAdherence",
    "PlanQuality",
    # Conversational metrics
    "TurnRelevancy",
    "RoleAdherence",
    "KnowledgeRetention",
    "ConversationCompleteness",
    "GoalAccuracy",
    "ToolUse",
    "TopicAdherence",
    # Safety metrics
    "Bias",
    "Toxicity",
    "NonAdvice",
    "Misuse",
    "PIILeakage",
    "RoleViolation",
    # General metrics
    "Hallucination",
    "Summarization",
    "JsonCorrectness",
    "PromptAlignment",
    # Deterministic metrics
    "ExactMatch",
    "PatternMatch",
    "experimental",
]

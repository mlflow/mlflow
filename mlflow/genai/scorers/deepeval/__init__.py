"""
DeepEval integration for MLflow.

This module provides integration with DeepEval metrics, allowing them to be used
with MLflow's scorer interface.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.deepeval import get_scorer

    scorer = get_scorer("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
    feedback = scorer(inputs="What is MLflow?", outputs="MLflow is a platform...")
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import CategoricalRating, get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.deepeval.models import create_deepeval_model
from mlflow.genai.scorers.deepeval.registry import (
    get_metric_class,
    is_deterministic_metric,
)
from mlflow.genai.scorers.deepeval.utils import (
    map_scorer_inputs_to_deepeval_test_case,
    map_session_to_deepeval_conversational_test_case,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class DeepEvalScorer(Scorer):
    """
    Base scorer class for DeepEval metrics.

    Args:
        metric_name: Name of the DeepEval metric (e.g., "AnswerRelevancy").
            If not provided, will use the class-level metric_name attribute.
        model: {{ model }}
        metric_kwargs: Additional metric-specific parameters
    """

    _metric: Any = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        **metric_kwargs: Any,
    ):
        # Use class attribute if metric_name not provided
        if metric_name is None:
            metric_name = self.metric_name

        super().__init__(name=metric_name)

        metric_class = get_metric_class(metric_name)

        self._is_deterministic = is_deterministic_metric(metric_name)

        if self._is_deterministic:
            # Deterministic metrics don't need a model
            self._metric = metric_class(**metric_kwargs)
            self._model_uri = None
        else:
            model = model or get_default_model()
            self._model_uri = model
            deepeval_model = create_deepeval_model(model)
            self._metric = metric_class(
                model=deepeval_model,
                verbose_mode=False,
                async_mode=False,
                **metric_kwargs,
            )

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    @property
    def is_session_level_scorer(self) -> bool:
        from deepeval.metrics.base_metric import BaseConversationalMetric

        return isinstance(self._metric, BaseConversationalMetric)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped DeepEval metric.

        Args:
            inputs: The input to evaluate
            outputs: The output to evaluate
            expectations: Expected values and context for evaluation
            trace: MLflow trace for evaluation
            session: List of MLflow traces for multi-turn evaluation

        Returns:
            Feedback object with pass/fail value, rationale, and score in metadata
        """
        if self._is_deterministic:
            source_type = AssessmentSourceType.CODE
            source_id = None
        else:
            source_type = AssessmentSourceType.LLM_JUDGE
            source_id = self._model_uri

        assessment_source = AssessmentSource(
            source_type=source_type,
            source_id=source_id,
        )

        try:
            if self.is_session_level_scorer:
                if session is None:
                    raise MlflowException.invalid_parameter_value(
                        f"Multi-turn scorer '{self.name}' requires 'session' parameter "
                        f"containing a list of traces from the conversation."
                    )
                test_case = map_session_to_deepeval_conversational_test_case(
                    session=session,
                    expectations=expectations,
                )
            else:
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
                    FRAMEWORK_METADATA_KEY: "deepeval",
                },
            )
        except Exception as e:
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
            )

    def _validate_kwargs(self, **metric_kwargs):
        if is_deterministic_metric(self.metric_name):
            if "model" in metric_kwargs:
                raise MlflowException.invalid_parameter_value(
                    f"{self.metric_name} got an unexpected keyword argument 'model'"
                )


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
def get_scorer(
    metric_name: str,
    model: str | None = None,
    **metric_kwargs: Any,
) -> DeepEvalScorer:
    """
    Get a DeepEval metric as an MLflow scorer.

    Args:
        metric_name: Name of the DeepEval metric (e.g., "AnswerRelevancy", "Faithfulness")
        model: {{ model }}
        metric_kwargs: Additional metric-specific parameters (e.g., threshold, include_reason)

    Returns:
        DeepEvalScorer instance that can be called with MLflow's scorer interface

    Examples:

    .. code-block:: python

        scorer = get_scorer("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
        feedback = scorer(inputs="What is MLflow?", outputs="MLflow is a platform...")

        scorer = get_scorer("Faithfulness", model="openai:/gpt-4")
        feedback = scorer(trace=trace)
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
    "get_scorer",
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

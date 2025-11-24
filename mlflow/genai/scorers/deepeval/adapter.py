from __future__ import annotations

import logging
from typing import Any

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


class DeepEvalAdapter(Scorer):
    """Adapter that wraps any DeepEval metric to work with MLflow's judge interface."""

    def __init__(
        self,
        metric_name: str,
        threshold: float = 0.5,
        model: str = "databricks",
        include_reason: bool = True,
        strict_mode: bool = False,
        **metric_kwargs,
    ):
        """
        Initialize a DeepEval metric adapter.

        Args:
            metric_name: Name of the DeepEval metric (e.g., "answer_relevancy")
            threshold: Score threshold for success (default: 0.5)
            model: Model URI in MLflow format (default: "databricks")
            include_reason: Whether to include reasoning in feedback
            strict_mode: Whether to use strict mode (forces threshold to 1.0)
            metric_kwargs: Additional metric-specific parameters
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.model_uri = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.metric_kwargs = metric_kwargs

        metric_class = get_metric_class(metric_name)
        deepeval_model = create_deepeval_model(model)

        self.metric = metric_class(
            threshold=threshold,
            model=deepeval_model,
            include_reason=include_reason,
            strict_mode=strict_mode,
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
            metric_name=self.metric_name,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=trace,
        )

        self.metric.measure(test_case)

        score = self.metric.score
        reason = self.metric.reason if self.include_reason else None
        success = self.metric.is_successful()

        return Feedback(
            name=self.metric_name,
            value=CategoricalRating.YES if success else CategoricalRating.NO,
            rationale=reason,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=f"deepeval/{self.metric_name}",
            ),
            metadata={
                "score": score,
                "threshold": self.threshold,
            },
        )


def get_judge(
    metric_name: str,
    threshold: float = 0.5,
    model: str = "databricks",
    include_reason: bool = True,
    strict_mode: bool = False,
    **metric_kwargs,
) -> DeepEvalAdapter:
    """
    Get a DeepEval metric as an MLflow judge.

    Args:
        metric_name: Name of the DeepEval metric (e.g., "AnswerRelevancy", "Faithfulness")
        threshold: Score threshold for success (default: 0.5)
        model: Model URI in MLflow format (default: "databricks")
        include_reason: Whether to include reasoning in feedback
        strict_mode: Whether to use strict mode (forces threshold to 1.0)
        metric_kwargs: Additional metric-specific parameters

    Returns:
        DeepEvalAdapter instance that can be called with MLflow's judge interface

    Examples:
        >>> judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
        >>> feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")

        >>> judge = get_judge("Faithfulness")
        >>> feedback = judge(
        ...     outputs="Paris is the capital of France",
        ...     expectations={"retrieval_context": ["Paris is France's capital city"]},
        ... )
    """
    return DeepEvalAdapter(
        metric_name=metric_name,
        threshold=threshold,
        model=model,
        include_reason=include_reason,
        strict_mode=strict_mode,
        **metric_kwargs,
    )

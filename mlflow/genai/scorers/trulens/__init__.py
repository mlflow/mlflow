"""
TruLens evaluation framework integration for MLflow.

This module provides integration with TruLens feedback functions, allowing them to be used
with MLflow's scorer interface.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.trulens import get_scorer

    scorer = get_scorer("Groundedness", model="openai:/gpt-5")
    feedback = scorer(trace=trace)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import CategoricalRating, get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.trulens.models import create_trulens_provider
from mlflow.genai.scorers.trulens.registry import get_feedback_method_name
from mlflow.genai.scorers.trulens.utils import (
    format_rationale,
    map_scorer_inputs_to_trulens_args,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)

# Threshold for determining pass/fail
_DEFAULT_THRESHOLD = 0.5


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class TruLensScorer(Scorer):
    """
    Base class for TruLens metric scorers.

    Args:
        metric_name: Name of the TruLens metric (e.g., "Groundedness")
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)
    """

    _provider: Any = PrivateAttr()
    _model: str = PrivateAttr()
    _method_name: str = PrivateAttr()
    _threshold: float = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        **kwargs: Any,
    ):
        if metric_name is None:
            metric_name = self.metric_name

        super().__init__(name=metric_name)
        model = model or get_default_model()
        self._model = model
        self._threshold = threshold

        self._provider = create_trulens_provider(model, **kwargs)
        self._method_name = get_feedback_method_name(metric_name)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        assessment_source = AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id=self._model,
        )

        try:
            args = map_scorer_inputs_to_trulens_args(
                metric_name=self.name,
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )

            feedback_method = getattr(self._provider, self._method_name)
            score, reasons = feedback_method(**args)

            rationale = format_rationale(reasons)
            value = CategoricalRating.YES if score >= self._threshold else CategoricalRating.NO

            return Feedback(
                name=self.name,
                value=value,
                rationale=rationale,
                source=assessment_source,
                metadata={
                    FRAMEWORK_METADATA_KEY: "trulens",
                    "score": score,
                    "threshold": self._threshold,
                },
            )
        except Exception as e:
            _logger.error(f"Error evaluating TruLens metric {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
                metadata={FRAMEWORK_METADATA_KEY: "trulens"},
            )


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
def get_scorer(
    metric_name: str,
    model: str | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> TruLensScorer:
    """
    Get a TruLens metric as an MLflow scorer.

    Args:
        metric_name: Name of the TruLens metric (e.g., "Groundedness", "ContextRelevance")
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)
        kwargs: Additional keyword arguments to pass to the TruLens provider.

    Returns:
        TruLensScorer instance that can be called with MLflow's scorer interface

    Examples:
        .. code-block:: python

            scorer = get_scorer("Groundedness", model="openai:/gpt-5")
            feedback = scorer(trace=trace)

            scorer = get_scorer("AnswerRelevance", model="openai:/gpt-5")
            feedback = scorer(trace=trace)
    """
    return TruLensScorer(
        metric_name=metric_name,
        model=model,
        threshold=threshold,
        **kwargs,
    )


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class Groundedness(TruLensScorer):
    """
    Evaluates whether the response is grounded in the provided context.

    Detects potential hallucinations where the model generates information not
    supported by the source material.

    Args:
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.trulens import Groundedness

            scorer = Groundedness(model="openai:/gpt-5")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "Groundedness"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class ContextRelevance(TruLensScorer):
    """
    Evaluates whether the retrieved context is relevant to the input query.

    Critical for RAG applications to ensure the retrieval step provides useful
    information for generating accurate responses.

    Args:
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.trulens import ContextRelevance

            scorer = ContextRelevance(model="openai:/gpt-5")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ContextRelevance"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class AnswerRelevance(TruLensScorer):
    """
    Evaluates whether the model's response is relevant to the input query.

    Measures how well the answer addresses what was asked.

    Args:
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.trulens import AnswerRelevance

            scorer = AnswerRelevance(model="openai:/gpt-5")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "AnswerRelevance"


@experimental(version="3.10.0")
@format_docstring(_MODEL_API_DOC)
class Coherence(TruLensScorer):
    """
    Evaluates the coherence and logical flow of the model's response.

    Measures how well-structured and logically consistent the output is.

    Args:
        model: {{ model }}
        threshold: Score threshold for pass/fail (default: 0.5)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.trulens import Coherence

            scorer = Coherence()
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "Coherence"


from mlflow.genai.scorers.trulens.scorers.agent_trace import (
    ExecutionEfficiency,
    LogicalConsistency,
    PlanAdherence,
    PlanQuality,
    ToolCalling,
    ToolSelection,
    TruLensAgentScorer,
)

__all__ = [
    # Core classes
    "TruLensScorer",
    "TruLensAgentScorer",
    "get_scorer",
    # RAG metric scorers
    "Groundedness",
    "ContextRelevance",
    "AnswerRelevance",
    "Coherence",
    # Agent trace scorers
    "LogicalConsistency",
    "ExecutionEfficiency",
    "PlanAdherence",
    "PlanQuality",
    "ToolSelection",
    "ToolCalling",
]

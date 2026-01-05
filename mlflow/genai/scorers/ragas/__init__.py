"""
RAGAS integration for MLflow.

This module provides integration with RAGAS metrics, allowing them to be used
with MLflow's judge interface.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.ragas import get_scorer

    judge = get_scorer("Faithfulness", model="openai:/gpt-4")
    feedback = judge(
        inputs="What is MLflow?", outputs="MLflow is a platform...", trace=trace
    )
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import PrivateAttr
from ragas.llms import BaseRagasLLM

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import CategoricalRating, get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.ragas.models import create_ragas_model
from mlflow.genai.scorers.ragas.registry import (
    get_metric_class,
    is_agentic_metric,
    is_deterministic_metric,
    metric_requires_only_embeddings,
    no_llm_required_in_metric_constructor,
)
from mlflow.genai.scorers.ragas.utils import (
    create_mlflow_error_message_from_ragas_param,
    map_scorer_inputs_to_ragas_sample,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class RagasScorer(Scorer):
    """
    Initialize a RAGAS metric scorer.

    Args:
        metric_name: Name of the RAGAS metric (e.g., "Faithfulness")
        model: {{ model }}
        metric_kwargs: Additional metric-specific parameters
    """

    _metric: Any = PrivateAttr()
    _is_deterministic: bool = PrivateAttr(default=False)
    _model: str = PrivateAttr()
    _llm: BaseRagasLLM = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        **metric_kwargs,
    ):
        if metric_name is None:
            metric_name = self.metric_name

        super().__init__(name=metric_name)
        model = model or get_default_model()
        self._model = model
        metric_class = get_metric_class(metric_name)

        if is_deterministic_metric(metric_name):
            self._metric = metric_class(**metric_kwargs)
            self._is_deterministic = True
        else:
            ragas_llm = create_ragas_model(model)
            self._llm = ragas_llm
            if no_llm_required_in_metric_constructor(
                metric_name
            ) or metric_requires_only_embeddings(metric_name):
                self._metric = metric_class(**metric_kwargs)
            else:
                self._metric = metric_class(llm=ragas_llm, **metric_kwargs)

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.THIRD_PARTY

    def _raise_registration_not_supported(self, method_name: str):
        raise MlflowException.invalid_parameter_value(
            f"'{method_name}()' is not supported for third-party scorers like RAGAS. "
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
            "'align()' is not supported for third-party scorers like RAGAS. "
            "Alignment is only available for MLflow's built-in judges."
        )

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped RAGAS metric.

        Args:
            inputs: The input to evaluate
            outputs: The output to evaluate
            expectations: Expected values and context for evaluation
            trace: MLflow trace for evaluation
            session: List of MLflow traces for multi-turn/agentic evaluation

        Returns:
            Feedback object with score, rationale, and metadata
        """
        if self._is_deterministic:
            assessment_source = AssessmentSource(
                source_type=AssessmentSourceType.CODE,
                source_id=self.name,
            )
        else:
            assessment_source = AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=self._model,
            )

        try:
            sample = map_scorer_inputs_to_ragas_sample(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
                session=session,
                is_agentic=is_agentic_metric(self.name),
            )

            if is_agentic_metric(self.name):
                result = self._evaluate_agentic_metric(sample)
            else:
                result = self._evaluate_single_turn_metric(sample)

            raw_value = getattr(result, "value", result)
            reason = getattr(result, "reason", None)

            try:
                score = float(raw_value)
            except (TypeError, ValueError):
                score = None

            # RAGAS metrics may have thresholds to map to binary feedback
            threshold = getattr(self._metric, "threshold", None)
            metadata = {FRAMEWORK_METADATA_KEY: "ragas"}

            if score is not None and threshold is not None:
                metadata["threshold"] = threshold
                metadata["score"] = score
                value = CategoricalRating.YES if score >= threshold else CategoricalRating.NO
            else:
                value = score if score is not None else raw_value

            return Feedback(
                name=self.name,
                value=value,
                rationale=reason,
                source=assessment_source,
                trace_id=None,
                metadata=metadata,
            )
        except (KeyError, IndexError) as e:
            # RAGAS raises KeyError/IndexError when required parameters are missing
            error_msg = str(e).strip("'\"")
            mlflow_error_message = create_mlflow_error_message_from_ragas_param(
                error_msg, self.name
            )
            _logger.error(
                f"Missing required parameter for RAGAS metric {self.name}: {mlflow_error_message}"
            )
            mlflow_error = MlflowException.invalid_parameter_value(mlflow_error_message)

            return Feedback(
                name=self.name,
                error=mlflow_error,
                source=assessment_source,
            )
        except Exception as e:
            _logger.error(f"Error evaluating RAGAS metric {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
            )

    def _evaluate_single_turn_metric(self, sample):
        if hasattr(self._metric, "single_turn_score"):
            return self._metric.single_turn_score(sample)
        elif hasattr(self._metric, "score"):
            # DiscreteMetric requires llm passed into score method
            return self._metric.score(response=sample.response, llm=self._llm)
        else:
            raise MlflowException(f"RAGAS metric {self.name} is not currently supported")

    def _evaluate_agentic_metric(self, sample):
        if hasattr(self._metric, "multi_turn_score"):
            return self._metric.multi_turn_score(sample)
        else:
            raise MlflowException(f"RAGAS agentic metric {self.name} is not currently supported.")

    def _validate_kwargs(self, **metric_kwargs):
        if is_deterministic_metric(self.metric_name) or no_llm_required_in_metric_constructor(
            self.metric_name
        ):
            if "model" in metric_kwargs:
                raise MlflowException.invalid_parameter_value(
                    f"{self.metric_name} got an unexpected keyword argument 'model'"
                )


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
def get_scorer(
    metric_name: str,
    model: str | None = None,
    **metric_kwargs,
) -> RagasScorer:
    """
    Get a RAGAS metric as an MLflow judge.

    Args:
        metric_name: Name of the RAGAS metric (e.g., "Faithfulness")
        model: {{ model }}
        metric_kwargs: Additional metric-specific parameters (e.g., threshold)

    Returns:
        RagasScorer instance that can be called with MLflow's judge interface

    Examples:

    .. code-block:: python

        # LLM-based metric
        judge = get_scorer("Faithfulness", model="openai:/gpt-4")
        feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")

        # Using trace with retrieval context
        judge = get_scorer("ContextPrecision", model="openai:/gpt-4")
        feedback = judge(trace=trace)

        # Deterministic metric (no LLM needed)
        judge = get_scorer("ExactMatch")
        feedback = judge(outputs="Paris", expectations={"expected_output": "Paris"})
    """
    model = model or get_default_model()
    return RagasScorer(
        metric_name=metric_name,
        model=model,
        **metric_kwargs,
    )


from mlflow.genai.scorers.ragas.scorers import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
    AnswerAccuracy,
    AspectCritic,
    BleuScore,
    ChrfScore,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    DiscreteMetric,
    ExactMatch,
    FactualCorrectness,
    Faithfulness,
    InstanceRubrics,
    NoiseSensitivity,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    NonLLMStringSimilarity,
    ResponseGroundedness,
    ResponseRelevancy,
    RougeScore,
    RubricsScore,
    SemanticSimilarity,
    StringPresence,
    SummarizationScore,
    ToolCallAccuracy,
    ToolCallF1,
    TopicAdherence,
)

__all__ = [
    # Core classes
    "RagasScorer",
    "get_scorer",
    # RAG metrics
    "ContextPrecision",
    "NonLLMContextPrecisionWithReference",
    "ContextRecall",
    "NonLLMContextRecall",
    "ContextEntityRecall",
    "NoiseSensitivity",
    "Faithfulness",
    "ResponseRelevancy",
    "SemanticSimilarity",
    # NVIDIA metrics
    "AnswerAccuracy",
    "ContextRelevance",
    "ResponseGroundedness",
    # Comparison metrics
    "FactualCorrectness",
    "NonLLMStringSimilarity",
    "BleuScore",
    "ChrfScore",
    "RougeScore",
    "StringPresence",
    "ExactMatch",
    # General purpose metrics
    "AspectCritic",
    "DiscreteMetric",
    "RubricsScore",
    "InstanceRubrics",
    # Agentic metrics
    "TopicAdherence",
    "ToolCallAccuracy",
    "ToolCallF1",
    "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference",
    # Other tasks
    "SummarizationScore",
]

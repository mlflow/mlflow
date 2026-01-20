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

import inspect
import logging
import re
from typing import Any

from pydantic import PrivateAttr
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.llms import BaseRagasLLM

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import CategoricalRating, get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.genai.scorers.ragas.models import (
    create_default_embeddings,
    create_ragas_model,
)
from mlflow.genai.scorers.ragas.registry import (
    get_metric_class,
    is_agentic_or_multiturn_metric,
    requires_args_from_placeholders,
    requires_embeddings,
    requires_llm_at_score_time,
    requires_llm_in_constructor,
)
from mlflow.genai.scorers.ragas.utils import (
    create_mlflow_error_message_from_ragas_param,
    map_scorer_inputs_to_ragas_sample,
)
from mlflow.genai.utils.trace_utils import _wrap_async_predict_fn
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
    _llm: BaseRagasLLM | None = PrivateAttr(default=None)

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        **metric_kwargs,
    ):
        if metric_name is None:
            metric_name = self.metric_name

        self._validate_args(metric_name, model)
        super().__init__(name=metric_name)
        model = model or get_default_model()
        self._model = model
        metric_class = get_metric_class(metric_name)
        ragas_llm = create_ragas_model(model)
        constructor_kwargs = dict(metric_kwargs)

        if requires_llm_in_constructor(metric_name):
            constructor_kwargs["llm"] = ragas_llm

        if requires_embeddings(metric_name):
            if constructor_kwargs.get("embeddings") is None:
                constructor_kwargs["embeddings"] = create_default_embeddings()

        if requires_llm_at_score_time(metric_name):
            self._llm = ragas_llm

        self._metric = metric_class(**constructor_kwargs)

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
        is_deterministic = not (
            requires_llm_in_constructor(self.name) or requires_llm_at_score_time(self.name)
        )
        if is_deterministic:
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
                is_agentic_or_multiturn=is_agentic_or_multiturn_metric(self.name),
            )

            result = self._evaluate(sample)
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
        except (KeyError, IndexError, ValueError) as e:
            # RAGAS raises KeyError/IndexError/ValueError when required parameters are missing
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

    def _evaluate(self, sample: SingleTurnSample | MultiTurnSample):
        if hasattr(self._metric, "single_turn_score"):
            return self._metric.single_turn_score(sample)
        elif hasattr(self._metric, "ascore"):
            kwargs = {}

            if requires_llm_at_score_time(self.name):
                kwargs["llm"] = self._llm

            if requires_args_from_placeholders(self.name):
                kwargs.update(self._extract_prompt_params_from_sample(sample))

            # need to inspect the signature as each metric has a different one for the ascore method
            sig = inspect.signature(self._metric.ascore)
            for param_name in sig.parameters:
                if param_name == "self":
                    continue

                if hasattr(sample, param_name):
                    value = getattr(sample, param_name)
                    kwargs[param_name] = value

            sync_score = _wrap_async_predict_fn(self._metric.ascore)
            return sync_score(**kwargs)
        else:
            raise MlflowException(f"RAGAS metric {self.name} is not currently supported")

    def _extract_prompt_params_from_sample(
        self, sample: SingleTurnSample | MultiTurnSample
    ) -> dict[str, Any]:
        """
        Extract parameters from the metric's prompt template and get values from sample.

        For metrics like DiscreteMetric where the prompt contains placeholders like
        {response}, {user_input}, etc., this extracts those placeholder names and fetches
        the corresponding values from the sample.
        """
        kwargs = {}
        prompt = getattr(self._metric, "prompt", None)
        if prompt is None:
            return kwargs

        prompt_str = str(prompt)
        placeholders = re.findall(r"\{(\w+)\}", prompt_str)

        for param_name in placeholders:
            if hasattr(sample, param_name):
                value = getattr(sample, param_name)
                if value is not None:
                    kwargs[param_name] = value

        return kwargs

    def _validate_args(self, metric_name: str | None, model: str | None):
        metric_name = metric_name or self.metric_name
        if not requires_llm_in_constructor(metric_name) and model is not None:
            raise MlflowException.invalid_parameter_value(
                f"{metric_name} got an unexpected keyword argument 'model'"
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
    return RagasScorer(
        metric_name=metric_name,
        model=model,
        **metric_kwargs,
    )


from mlflow.genai.scorers.ragas.scorers import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
    AnswerAccuracy,
    AnswerRelevancy,
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
    InstanceSpecificRubrics,
    NoiseSensitivity,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    NonLLMStringSimilarity,
    ResponseGroundedness,
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
    "AnswerRelevancy",
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
    "InstanceSpecificRubrics",
    # Agentic metrics
    "TopicAdherence",
    "ToolCallAccuracy",
    "ToolCallF1",
    "AgentGoalAccuracyWithReference",
    "AgentGoalAccuracyWithoutReference",
    # Other tasks
    "SummarizationScore",
]

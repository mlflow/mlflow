"""
Phoenix (Arize) integration for MLflow.
This module provides integration with Phoenix evaluators, allowing them to be used
with MLflow's scorer interface.
Example usage:
.. code-block:: python
    from mlflow.genai.scorers.phoenix import get_scorer

    scorer = get_scorer("Hallucination", model="openai:/gpt-4")
    feedback = scorer(
        inputs="What is MLflow?", outputs="MLflow is a platform...", trace=trace
    )
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import PrivateAttr

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.scorers import FRAMEWORK_METADATA_KEY
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.phoenix.models import create_phoenix_model
from mlflow.genai.scorers.phoenix.registry import get_evaluator_class
from mlflow.genai.scorers.phoenix.utils import map_scorer_inputs_to_phoenix_record
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

_logger = logging.getLogger(__name__)


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class PhoenixScorer(Scorer):
    """
    Base class for Phoenix metric scorers.

    Args:
        metric_name: Name of the Phoenix metric (e.g., "Hallucination")
        model: {{ model }}
    """

    _evaluator: Any = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self,
        metric_name: str | None = None,
        model: str | None = None,
        **evaluator_kwargs: Any,
    ):
        if metric_name is None:
            metric_name = self.metric_name
        super().__init__(name=metric_name)
        model = model or get_default_model()
        self._model = model
        phoenix_model = create_phoenix_model(model)
        evaluator_class = get_evaluator_class(metric_name)
        self._evaluator = evaluator_class(model=phoenix_model, **evaluator_kwargs)

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Feedback:
        """
        Evaluate using the wrapped Phoenix evaluator.

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
            source_id=self._model,
        )
        try:
            record = map_scorer_inputs_to_phoenix_record(
                inputs=inputs,
                outputs=outputs,
                expectations=expectations,
                trace=trace,
            )
            label, score, explanation = self._evaluator.evaluate(record=record)
            return Feedback(
                name=self.name,
                value=label,
                rationale=explanation or None,
                source=assessment_source,
                metadata={
                    FRAMEWORK_METADATA_KEY: "phoenix",
                    "score": score,
                },
            )
        except Exception as e:
            _logger.error(f"Error evaluating Phoenix metric {self.name}: {e}")
            return Feedback(
                name=self.name,
                error=e,
                source=assessment_source,
            )


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
def get_scorer(
    metric_name: str,
    model: str | None = None,
    **evaluator_kwargs: Any,
) -> PhoenixScorer:
    """
    Get a Phoenix metric as an MLflow scorer.

    Args:
        metric_name: Name of the Phoenix metric (e.g., "Hallucination", "Relevance")
        model: {{ model }}
        evaluator_kwargs: Additional keyword arguments to pass to the Phoenix evaluator.

    Returns:
        PhoenixScorer instance that can be called with MLflow's scorer interface
    Examples:
        >>> scorer = get_scorer("Hallucination", model="openai:/gpt-4")
        >>> feedback = scorer(
        ...     inputs="What is MLflow?",
        ...     outputs="MLflow is a platform...",
        ...     expectations={"context": "MLflow is an ML platform."},
        ... )
        >>> scorer = get_scorer("Relevance", model="databricks")
        >>> feedback = scorer(trace=trace)
    """
    return PhoenixScorer(
        metric_name=metric_name,
        model=model,
        **evaluator_kwargs,
    )


@experimental(version="3.9.0")
class Hallucination(PhoenixScorer):
    """
    Detects hallucinations where the LLM fabricates information not present in the context.

    Args:
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")

    Examples:
        .. code-block:: python
            from mlflow.genai.scorers.phoenix import Hallucination

            scorer = Hallucination(model="openai:/gpt-4")
            feedback = scorer(
                inputs="What is the capital of France?",
                outputs="Paris is the capital of France.",
                expectations={"context": "France is in Europe. Its capital is Paris."},
            )
    """

    metric_name: ClassVar[str] = "Hallucination"


@experimental(version="3.9.0")
class Relevance(PhoenixScorer):
    """
    Evaluates whether the retrieved context is relevant to the input query.

    Args:
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")

    Examples:
        .. code-block:: python
            from mlflow.genai.scorers.phoenix import Relevance

            scorer = Relevance(model="databricks")
            feedback = scorer(
                inputs="What is machine learning?",
                expectations={"context": "ML is a subset of AI..."},
            )
    """

    metric_name: ClassVar[str] = "Relevance"


@experimental(version="3.9.0")
class Toxicity(PhoenixScorer):
    """
    Detects toxic content in the model's response.

    Args:
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")

    Examples:
        .. code-block:: python
            from mlflow.genai.scorers.phoenix import Toxicity

            scorer = Toxicity()
            feedback = scorer(outputs="This is a friendly response.")
    """

    metric_name: ClassVar[str] = "Toxicity"


@experimental(version="3.9.0")
class QA(PhoenixScorer):
    """
    Evaluates whether the answer correctly addresses the question based on reference.

    Args:
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")

    Examples:
        .. code-block:: python
            from mlflow.genai.scorers.phoenix import QA

            scorer = QA(model="openai:/gpt-4o")
            feedback = scorer(
                inputs="What is 2+2?",
                outputs="4",
                expectations={"context": "Basic arithmetic: 2+2=4"},
            )
    """

    metric_name: ClassVar[str] = "QA"


@experimental(version="3.9.0")
class Summarization(PhoenixScorer):
    """
    Evaluates the quality of a summarization against the original document.

    Args:
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")

    Examples:
        .. code-block:: python
            from mlflow.genai.scorers.phoenix import Summarization

            scorer = Summarization()
            feedback = scorer(
                inputs="Long document text...",
                outputs="Brief summary of the document.",
            )
    """

    metric_name: ClassVar[str] = "Summarization"


__all__ = [
    # Core classes
    "PhoenixScorer",
    "get_scorer",
    # Metric scorers
    "Hallucination",
    "Relevance",
    "Toxicity",
    "QA",
    "Summarization",
]

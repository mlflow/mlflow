"""
TruLens evaluation framework integration for MLflow GenAI scorers.

This module wraps TruLens feedback functions as MLflow scorers, enabling use of
TruLens' groundedness, context relevance, answer relevance, and coherence metrics
within the MLflow evaluation framework.

**Score Semantics:**
All TruLens scorers follow MLflow's convention where higher scores indicate better quality:
- 1.0 = best quality (fully grounded, highly relevant, perfectly coherent)
- 0.0 = worst quality (not grounded, irrelevant, incoherent)

TruLens internally normalizes scores to the 0-1 range before returning them.

**Available Scorers:**
- ``TruLensGroundednessScorer``: Evaluates if outputs are grounded in context (1.0=grounded)
- ``TruLensContextRelevanceScorer``: Evaluates context relevance to query (1.0=relevant)
- ``TruLensAnswerRelevanceScorer``: Evaluates answer relevance to query (1.0=relevant)
- ``TruLensCoherenceScorer``: Evaluates logical flow of outputs (1.0=coherent)

**Installation:**
These scorers require the ``trulens`` packages::

    pip install trulens trulens-providers-openai

For LiteLLM provider support::

    pip install trulens trulens-providers-litellm

**Example Usage:**

.. code-block:: python

    import mlflow
    from mlflow.genai.scorers import (
        TruLensGroundednessScorer,
        TruLensContextRelevanceScorer,
    )

    # Create scorers (default: OpenAI provider)
    groundedness_scorer = TruLensGroundednessScorer()
    relevance_scorer = TruLensContextRelevanceScorer()

    # Or use LiteLLM for broader model support
    groundedness_scorer_litellm = TruLensGroundednessScorer(model_provider="litellm")

    # Use with mlflow.genai.evaluate
    eval_data = [
        {
            "inputs": {"query": "What is the capital of France?"},
            "outputs": "Paris is the capital of France.",
            "context": "France is a country in Europe. Its capital is Paris.",
        }
    ]
    results = mlflow.genai.evaluate(data=eval_data, scorers=[groundedness_scorer])

For more information on TruLens, see:
https://www.trulens.org/
"""

from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


def _check_trulens_installed():
    """Check if trulens is installed and raise a helpful error if not."""
    try:
        import trulens.providers.openai  # noqa: F401

        return True
    except ImportError:
        raise MlflowException(
            "TruLens scorers require the 'trulens' package. "
            "Install it with: pip install trulens trulens-providers-openai",
            error_code=INVALID_PARAMETER_VALUE,
        )


class _TruLensScorerBase(Scorer):
    """Base class for TruLens scorer wrappers."""

    name: str
    model_name: str | None = None
    model_provider: str = "openai"

    def _get_trulens_provider(self):
        """Get the appropriate TruLens provider instance."""
        _check_trulens_installed()

        if self.model_provider == "openai":
            from trulens.providers.openai import OpenAI

            return OpenAI(model_engine=self.model_name or "gpt-4o-mini")
        elif self.model_provider == "litellm":
            try:
                from trulens.providers.litellm import LiteLLM

                return LiteLLM(model_engine=self.model_name or "gpt-4o-mini")
            except ImportError:
                raise MlflowException(
                    "LiteLLM provider requires 'trulens-providers-litellm'. "
                    "Install it with: pip install trulens-providers-litellm",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            raise MlflowException(
                f"Unsupported model provider: {self.model_provider}. "
                "Currently supported: 'openai', 'litellm'",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _validate_score(self, score: float) -> float:
        """
        Validate that score is in expected 0-1 range.

        TruLens normalizes scores internally to 0-1 range. If a score falls
        outside this range, it indicates a potential bug or version incompatibility.

        Args:
            score: The score from TruLens

        Returns:
            Validated score (clamped with warning if out of range)
        """
        if score < 0.0 or score > 1.0:
            import logging

            logging.getLogger(__name__).warning(
                f"TruLens returned score {score} outside expected 0-1 range. "
                "This may indicate a version incompatibility. Clamping to valid range."
            )
            return min(1.0, max(0.0, score))
        return score

    def _format_rationale(self, reasons: dict[str, Any] | None) -> str:
        """
        Format TruLens reasons dict into a readable rationale string.

        TruLens returns reasons as a dict with various keys depending on the
        feedback function.

        Args:
            reasons: Dict of reasons from TruLens

        Returns:
            Formatted rationale string
        """
        if not reasons:
            return "No detailed reasoning available."

        # TruLens reasons dict can have various structures
        # Common keys: 'reason', 'reasons', 'supporting_evidence', etc.
        parts = []
        for key, value in reasons.items():
            if isinstance(value, list):
                parts.append(f"{key}: {'; '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {value}")
            else:
                parts.append(f"{key}: {value}")

        return " | ".join(parts) if parts else "No detailed reasoning available."


@experimental(version="3.8.0")
class TruLensGroundednessScorer(_TruLensScorerBase):
    """
    TruLens groundedness scorer.

    Evaluates whether the response is grounded in the provided context, detecting
    potential hallucinations where the model generates information not supported
    by the source material.

    Args:
        name: The name of the scorer. Defaults to "trulens_groundedness".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import TruLensGroundednessScorer

        scorer = TruLensGroundednessScorer()
        feedback = scorer(
            outputs="The Eiffel Tower is 330 meters tall.",
            context="The Eiffel Tower, located in Paris, stands at 330 meters.",
        )
        print(feedback)
    """

    name: str = "trulens_groundedness"

    def __call__(
        self,
        *,
        outputs: Any | None = None,
        context: str | list[str] | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate groundedness of output against context.

        Args:
            outputs: The model's response to evaluate.
            context: Reference context(s) to check groundedness against.

        Returns:
            Feedback with groundedness score (0-1) and rationale.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()

        output_str = str(outputs) if outputs else ""
        context_str = "\n".join(context) if isinstance(context, list) else (context or "")

        # TruLens groundedness_measure_with_cot_reasons returns Tuple[float, dict]
        # Score is normalized to 0-1 range by TruLens
        score, reasons = provider.groundedness_measure_with_cot_reasons(
            source=context_str,
            statement=output_str,
        )

        validated_score = self._validate_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=validated_score,
            rationale=rationale,
        )


@experimental(version="3.8.0")
class TruLensContextRelevanceScorer(_TruLensScorerBase):
    """
    TruLens context relevance scorer.

    Evaluates whether the retrieved context is relevant to the input query.
    This is critical for RAG applications to ensure the retrieval step
    provides useful information for generating accurate responses.

    Args:
        name: The name of the scorer. Defaults to "trulens_context_relevance".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import TruLensContextRelevanceScorer

        scorer = TruLensContextRelevanceScorer()
        feedback = scorer(
            inputs={"query": "What is the capital of France?"},
            context="Paris is the capital and largest city of France.",
        )
        print(feedback)
    """

    name: str = "trulens_context_relevance"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        context: str | list[str] | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate context relevance to the input query.

        Args:
            inputs: Input data containing the query/request.
            context: Retrieved context(s) to evaluate for relevance.

        Returns:
            Feedback with context relevance score (0-1) and rationale.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()

        query = inputs.get("query", str(inputs)) if inputs else ""
        context_str = "\n".join(context) if isinstance(context, list) else (context or "")

        # TruLens context_relevance_with_cot_reasons returns Tuple[float, Dict]
        # Score is normalized to 0-1 range by TruLens
        score, reasons = provider.context_relevance_with_cot_reasons(
            question=query,
            context=context_str,
        )

        validated_score = self._validate_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=validated_score,
            rationale=rationale,
        )


@experimental(version="3.8.0")
class TruLensAnswerRelevanceScorer(_TruLensScorerBase):
    """
    TruLens answer relevance scorer.

    Evaluates whether the model's response is relevant to the input query.
    This measures how well the answer addresses what was asked.

    Args:
        name: The name of the scorer. Defaults to "trulens_answer_relevance".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import TruLensAnswerRelevanceScorer

        scorer = TruLensAnswerRelevanceScorer()
        feedback = scorer(
            inputs={"query": "What is machine learning?"},
            outputs="Machine learning is a branch of AI...",
        )
        print(feedback)
    """

    name: str = "trulens_answer_relevance"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate answer relevance to the input query.

        Args:
            inputs: Input data containing the query/request.
            outputs: The model's response to evaluate.

        Returns:
            Feedback with answer relevance score (0-1) and rationale.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()

        query = inputs.get("query", str(inputs)) if inputs else ""
        output_str = str(outputs) if outputs else ""

        # TruLens relevance_with_cot_reasons returns Tuple[float, Dict]
        # Score is normalized to 0-1 range by TruLens
        score, reasons = provider.relevance_with_cot_reasons(
            prompt=query,
            response=output_str,
        )

        validated_score = self._validate_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=validated_score,
            rationale=rationale,
        )


@experimental(version="3.8.0")
class TruLensCoherenceScorer(_TruLensScorerBase):
    """
    TruLens coherence scorer.

    Evaluates the coherence and logical flow of the model's response.
    This measures how well-structured and logically consistent the output is.

    Args:
        name: The name of the scorer. Defaults to "trulens_coherence".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".
        model_provider: The model provider ("openai" or "litellm"). Defaults to "openai".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import TruLensCoherenceScorer

        scorer = TruLensCoherenceScorer()
        feedback = scorer(
            outputs="Machine learning is a branch of AI. It enables systems to learn from data. "
            "This learning process improves performance over time.",
        )
        print(feedback)
    """

    name: str = "trulens_coherence"

    def __call__(
        self,
        *,
        outputs: Any | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate coherence of the output text.

        Args:
            outputs: The model's response to evaluate for coherence.

        Returns:
            Feedback with coherence score (0-1) and rationale.
        """
        _check_trulens_installed()

        provider = self._get_trulens_provider()

        output_str = str(outputs) if outputs else ""

        # TruLens coherence_with_cot_reasons returns Tuple[float, Dict]
        # Score is normalized to 0-1 range by TruLens
        score, reasons = provider.coherence_with_cot_reasons(
            text=output_str,
        )

        validated_score = self._validate_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=validated_score,
            rationale=rationale,
        )

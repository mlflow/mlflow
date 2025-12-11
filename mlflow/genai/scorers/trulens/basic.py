"""
TruLens basic scorers for input/output evaluation.

These scorers evaluate the quality of LLM outputs based on groundedness,
relevance, and coherence metrics.
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

    def _clamp_score(self, score: float) -> float:
        """Clamp score to valid 0-1 range."""
        return min(1.0, max(0.0, score))

    def _format_rationale(self, reasons: dict[str, Any] | None) -> str:
        """Format TruLens reasons dict into a readable rationale string."""
        if not reasons:
            return "No detailed reasoning available."

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

        score, reasons = provider.groundedness_measure_with_cot_reasons(
            source=context_str,
            statement=output_str,
        )

        clamped_score = self._clamp_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=clamped_score,
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

        score, reasons = provider.context_relevance_with_cot_reasons(
            question=query,
            context=context_str,
        )

        clamped_score = self._clamp_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=clamped_score,
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

        score, reasons = provider.relevance_with_cot_reasons(
            prompt=query,
            response=output_str,
        )

        clamped_score = self._clamp_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=clamped_score,
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

        score, reasons = provider.coherence_with_cot_reasons(
            text=output_str,
        )

        clamped_score = self._clamp_score(score)
        rationale = self._format_rationale(reasons)

        return Feedback(
            name=self.name,
            value=clamped_score,
            rationale=rationale,
        )

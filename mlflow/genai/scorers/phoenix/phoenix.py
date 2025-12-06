"""
Phoenix (Arize) evaluation framework integration for MLflow GenAI scorers.

This module wraps Phoenix evaluators as MLflow scorers, enabling use of Phoenix's
hallucination detection, relevance, toxicity, QA, and summarization metrics
within the MLflow evaluation framework.

**Score Semantics:**
All Phoenix scorers follow MLflow's convention where higher scores indicate better quality:
- 1.0 = best quality (factual, relevant, non-toxic, correct, good summary)
- 0.0 = worst quality (hallucinated, irrelevant, toxic, incorrect, poor summary)

Note: Phoenix internally uses inverted scoring for some metrics (e.g., hallucination=0 means
factual). This module automatically inverts these scores to match MLflow's convention.

**Available Scorers:**
- ``PhoenixHallucinationScorer``: Detects hallucinations (1.0=factual, 0.0=hallucinated)
- ``PhoenixRelevanceScorer``: Evaluates context relevance (1.0=relevant, 0.0=irrelevant)
- ``PhoenixToxicityScorer``: Detects toxic content (1.0=safe, 0.0=toxic)
- ``PhoenixQAScorer``: Evaluates QA correctness (1.0=correct, 0.0=incorrect)
- ``PhoenixSummarizationScorer``: Evaluates summary quality (1.0=good, 0.0=poor)

**Installation:**
These scorers require the ``arize-phoenix-evals`` package::

    pip install arize-phoenix-evals

**Example Usage:**

.. code-block:: python

    import mlflow
    from mlflow.genai.scorers import PhoenixHallucinationScorer, PhoenixRelevanceScorer

    # Create scorers
    hallucination_scorer = PhoenixHallucinationScorer()
    relevance_scorer = PhoenixRelevanceScorer()

    # Use with mlflow.genai.evaluate
    eval_data = [
        {
            "inputs": {"query": "What is the capital of France?"},
            "outputs": "Paris is the capital of France.",
            "context": "France is a country in Europe. Its capital is Paris.",
        }
    ]
    results = mlflow.genai.evaluate(data=eval_data, scorers=[hallucination_scorer])

For more information on Phoenix evaluators, see:
https://docs.arize.com/phoenix/evaluation/evals
"""

from typing import Any

from mlflow.entities.assessment import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental


def _check_phoenix_installed():
    """Check if phoenix.evals is installed and raise a helpful error if not."""
    try:
        import phoenix.evals  # noqa: F401

        return True
    except ImportError:
        raise MlflowException(
            "Phoenix evaluators require the 'arize-phoenix-evals' package. "
            "Install it with: pip install arize-phoenix-evals",
            error_code=INVALID_PARAMETER_VALUE,
        )


class _PhoenixScorerBase(Scorer):
    """Base class for Phoenix scorer wrappers."""

    name: str
    model_name: str | None = None

    def _get_phoenix_model(self):
        """Get the Phoenix OpenAI model instance."""
        _check_phoenix_installed()

        from phoenix.evals import OpenAIModel

        return OpenAIModel(model=self.model_name or "gpt-4o-mini")

    def _parse_result(
        self,
        result: tuple[str, float | None, str | None],
        positive_label: str,
        invert_score: bool = False,
    ) -> tuple[float, str]:
        """
        Parse Phoenix evaluator result tuple.

        Phoenix returns: Tuple[str, Optional[float], Optional[str]]
                        (label, score, explanation)

        Args:
            result: The tuple returned by Phoenix evaluator
            positive_label: The label that indicates a positive/good result
            invert_score: If True, invert the score (1 - score). Used for evaluators
                where 0 means good (e.g., hallucination=0 means factual,
                toxicity=0 means non-toxic).

        Returns:
            Tuple of (normalized_score, rationale)
        """
        label, score, explanation = result

        # If Phoenix provides a score, use it; otherwise derive from label
        if score is not None:
            normalized_score = float(score)
            if invert_score:
                normalized_score = 1.0 - normalized_score
        else:
            normalized_score = 1.0 if label == positive_label else 0.0

        rationale = explanation or f"Label: {label}"

        return normalized_score, rationale


@experimental(version="3.8.0")
class PhoenixHallucinationScorer(_PhoenixScorerBase):
    """
    Phoenix hallucination detection scorer.

    Evaluates whether the output contains hallucinations (information not supported
    by the provided context/reference).

    Args:
        name: The name of the scorer. Defaults to "phoenix_hallucination".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PhoenixHallucinationScorer

        scorer = PhoenixHallucinationScorer()
        feedback = scorer(
            inputs={"query": "What is the capital of France?"},
            outputs="Paris is the capital of France.",
            context="France is a country in Europe. Its capital is Paris.",
        )
        print(feedback)
    """

    name: str = "phoenix_hallucination"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        context: str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate for hallucinations.

        Args:
            inputs: Input data containing the query/request.
            outputs: The model's response to evaluate.
            context: Reference context to check hallucinations against.

        Returns:
            Feedback with hallucination assessment (1.0 = factual, 0.0 = hallucinated).
        """
        _check_phoenix_installed()

        from phoenix.evals import HallucinationEvaluator

        model = self._get_phoenix_model()
        evaluator = HallucinationEvaluator(model=model)

        # Build record dict as expected by Phoenix
        query = inputs.get("query", str(inputs)) if inputs else ""
        record = {
            "input": query,
            "output": str(outputs) if outputs else "",
            "reference": context or "",
        }

        result = evaluator.evaluate(record=record)
        # Phoenix hallucination: score=0 means factual (good), score=1 means hallucinated (bad)
        # We invert so that 1.0 = factual/good, 0.0 = hallucinated/bad
        score, rationale = self._parse_result(result, positive_label="factual", invert_score=True)

        return Feedback(name=self.name, value=score, rationale=rationale)


@experimental(version="3.8.0")
class PhoenixRelevanceScorer(_PhoenixScorerBase):
    """
    Phoenix relevance scorer.

    Evaluates whether the retrieved context is relevant to the input query.

    Args:
        name: The name of the scorer. Defaults to "phoenix_relevance".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PhoenixRelevanceScorer

        scorer = PhoenixRelevanceScorer()
        feedback = scorer(
            inputs={"query": "What is machine learning?"},
            context="Machine learning is a subset of AI that enables systems to learn.",
        )
        print(feedback)
    """

    name: str = "phoenix_relevance"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        context: str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate relevance of context to input query.

        Args:
            inputs: Input data containing the query/request.
            context: Retrieved context to evaluate for relevance.

        Returns:
            Feedback with relevance assessment (1.0 = relevant, 0.0 = irrelevant).
        """
        _check_phoenix_installed()

        from phoenix.evals import RelevanceEvaluator

        model = self._get_phoenix_model()
        evaluator = RelevanceEvaluator(model=model)

        # Build record dict - RelevanceEvaluator expects 'input' and 'reference'
        query = inputs.get("query", str(inputs)) if inputs else ""
        record = {
            "input": query,
            "reference": context or "",
        }

        result = evaluator.evaluate(record=record)
        score, rationale = self._parse_result(result, positive_label="relevant")

        return Feedback(name=self.name, value=score, rationale=rationale)


@experimental(version="3.8.0")
class PhoenixToxicityScorer(_PhoenixScorerBase):
    """
    Phoenix toxicity scorer.

    Evaluates whether the output contains toxic content.

    Args:
        name: The name of the scorer. Defaults to "phoenix_toxicity".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PhoenixToxicityScorer

        scorer = PhoenixToxicityScorer()
        feedback = scorer(outputs="This is a friendly response.")
        print(feedback)
    """

    name: str = "phoenix_toxicity"

    def __call__(
        self,
        *,
        outputs: Any | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate toxicity of output.

        Args:
            outputs: The model's response to evaluate.

        Returns:
            Feedback with toxicity assessment (1.0 = non-toxic/safe, 0.0 = toxic).
        """
        _check_phoenix_installed()

        from phoenix.evals import ToxicityEvaluator

        model = self._get_phoenix_model()
        evaluator = ToxicityEvaluator(model=model)

        # ToxicityEvaluator only needs 'input' field (the text to evaluate)
        record = {
            "input": str(outputs) if outputs else "",
        }

        result = evaluator.evaluate(record=record)
        # For toxicity, non-toxic is the positive outcome
        # Phoenix toxicity: score=0 means non-toxic (good), score=1 means toxic (bad)
        # We invert so that 1.0 = non-toxic/good, 0.0 = toxic/bad
        score, rationale = self._parse_result(result, positive_label="non-toxic", invert_score=True)

        return Feedback(name=self.name, value=score, rationale=rationale)


@experimental(version="3.8.0")
class PhoenixQAScorer(_PhoenixScorerBase):
    """
    Phoenix QA (Question-Answer) correctness scorer.

    Evaluates whether the answer correctly addresses the question based on reference.

    Args:
        name: The name of the scorer. Defaults to "phoenix_qa".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PhoenixQAScorer

        scorer = PhoenixQAScorer()
        feedback = scorer(
            inputs={"query": "What is 2+2?"},
            outputs="4",
            context="Basic arithmetic: 2+2=4",
        )
        print(feedback)
    """

    name: str = "phoenix_qa"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        context: str | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate QA correctness.

        Args:
            inputs: Input data containing the question.
            outputs: The model's answer.
            context: Reference context for correctness check.

        Returns:
            Feedback with QA correctness assessment (1.0 = correct, 0.0 = incorrect).
        """
        _check_phoenix_installed()

        from phoenix.evals import QAEvaluator

        model = self._get_phoenix_model()
        evaluator = QAEvaluator(model=model)

        query = inputs.get("query", str(inputs)) if inputs else ""
        record = {
            "input": query,
            "output": str(outputs) if outputs else "",
            "reference": context or "",
        }

        result = evaluator.evaluate(record=record)
        score, rationale = self._parse_result(result, positive_label="correct")

        return Feedback(name=self.name, value=score, rationale=rationale)


@experimental(version="3.8.0")
class PhoenixSummarizationScorer(_PhoenixScorerBase):
    """
    Phoenix summarization quality scorer.

    Evaluates the quality of a summarization against the original document.

    Args:
        name: The name of the scorer. Defaults to "phoenix_summarization".
        model_name: The model to use for evaluation. Defaults to "gpt-4o-mini".

    Example:

    .. code-block:: python

        from mlflow.genai.scorers import PhoenixSummarizationScorer

        scorer = PhoenixSummarizationScorer()
        feedback = scorer(
            inputs={"document": "Long document text..."},
            outputs="Brief summary of the document.",
        )
        print(feedback)
    """

    name: str = "phoenix_summarization"

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: Any | None = None,
        **kwargs,
    ) -> Feedback:
        """
        Evaluate summarization quality.

        Args:
            inputs: Input data containing the original document.
            outputs: The generated summary.

        Returns:
            Feedback with summarization quality assessment (1.0 = good, 0.0 = poor).
        """
        _check_phoenix_installed()

        from phoenix.evals import SummarizationEvaluator

        model = self._get_phoenix_model()
        evaluator = SummarizationEvaluator(model=model)

        # SummarizationEvaluator expects 'input' (document) and 'output' (summary)
        document = inputs.get("document", str(inputs)) if inputs else ""
        record = {
            "input": document,
            "output": str(outputs) if outputs else "",
        }

        result = evaluator.evaluate(record=record)
        score, rationale = self._parse_result(result, positive_label="good")

        return Feedback(name=self.name, value=score, rationale=rationale)

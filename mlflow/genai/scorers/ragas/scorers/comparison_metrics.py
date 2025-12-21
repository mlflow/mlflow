from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class FactualCorrectness(RagasScorer):
    """
    Evaluates the factual correctness of the output compared to a reference.

    This metric uses an LLM to determine if the output is factually correct when compared
    to a reference answer or ground truth.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import FactualCorrectness

            scorer = FactualCorrectness(model="openai:/gpt-4")
            feedback = scorer(
                outputs="Paris is the capital of France.",
                expectations={"expected_output": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "FactualCorrectness"


@experimental(version="3.8.0")
class NonLLMStringSimilarity(RagasScorer):
    """
    Calculates string similarity without using an LLM.

    This is a deterministic metric that computes string similarity between the output
    and expected output.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import NonLLMStringSimilarity

            scorer = NonLLMStringSimilarity()
            feedback = scorer(
                outputs="Paris",
                expectations={"expected_output": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "NonLLMStringSimilarity"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
class BleuScore(RagasScorer):
    """
    Calculates BLEU score.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import BleuScore

            scorer = BleuScore()
            feedback = scorer(
                outputs="The cat sat on the mat",
                expectations={"expected_output": "A cat was sitting on the mat"},
            )
    """

    metric_name: ClassVar[str] = "BleuScore"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
class ChrfScore(RagasScorer):
    """
    Calculates Chrf (Character F-score) score between the output and expected output.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ChrfScore

            scorer = ChrfScore()
            feedback = scorer(
                outputs="Hello world",
                expectations={"expected_output": "Hello world!"},
            )
    """

    metric_name: ClassVar[str] = "ChrfScore"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
class RougeScore(RagasScorer):
    """
    Calculates ROUGE score between the output and expected output.

    Args:
        **metric_kwargs: Additional metric-specific parameters (e.g., rouge_type)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import RougeScore

            scorer = RougeScore()
            feedback = scorer(
                outputs="Short summary of the text",
                expectations={"expected_output": "Summary of the text"},
            )
    """

    metric_name: ClassVar[str] = "RougeScore"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
class StringPresence(RagasScorer):
    """
    Checks if the expected output is present in the output.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import StringPresence

            scorer = StringPresence()
            feedback = scorer(
                outputs="The capital of France is Paris",
                expectations={"expected_output": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "StringPresence"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
class ExactMatch(RagasScorer):
    """
    Performs exact string matching between the output and expected output.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ExactMatch

            scorer = ExactMatch()
            feedback = scorer(
                outputs="Paris",
                expectations={"expected_output": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "ExactMatch"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)

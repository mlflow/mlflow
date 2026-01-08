"""RAG (Retrieval-Augmented Generation) metrics for DeepEval integration."""

from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.deepeval import DeepEvalScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class AnswerRelevancy(DeepEvalScorer):
    """
    Evaluates whether the output is relevant to the input.

    This metric measures how relevant the actual output is to the input query. It evaluates
    whether the generated response directly addresses the question asked. Higher scores indicate
    better relevance to the input.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import AnswerRelevancy

            scorer = AnswerRelevancy(threshold=0.7, model="openai:/gpt-4")
            feedback = scorer(
                inputs="What is the capital of France?",
                outputs="Paris is the capital of France.",
            )
            print(feedback.value)  # CategoricalRating.YES or CategoricalRating.NO
    """

    metric_name: ClassVar[str] = "AnswerRelevancy"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class Faithfulness(DeepEvalScorer):
    """
    Evaluates whether the output is factually consistent with the retrieval context.

    This metric determines if claims in the output can be inferred from the provided context.
    It helps detect hallucinations by checking if the generated content is grounded in the
    retrieved documents.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import Faithfulness

            scorer = Faithfulness(threshold=0.8, model="databricks")
            feedback = scorer(trace=trace)  # trace contains outputs and retrieval_context
    """

    metric_name: ClassVar[str] = "Faithfulness"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextualRecall(DeepEvalScorer):
    """
    Evaluates whether the retrieval context contains all necessary information.

    This metric measures how much of the expected output can be attributed to the nodes in
    the retrieval context. It assesses the quality of the retriever by checking if all
    required information is present in the retrieved documents.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ContextualRecall

            scorer = ContextualRecall(model="databricks")
            feedback = scorer(trace=trace)  # trace contains expected_output and retrieval_context
    """

    metric_name: ClassVar[str] = "ContextualRecall"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextualPrecision(DeepEvalScorer):
    """
    Evaluates whether relevant nodes in the retrieval context are ranked higher than
    irrelevant ones.

    This metric assesses the quality of your retriever by checking if the most relevant
    retrieved context are ranked higher than less relevant ones. It helps evaluate the
    ranking effectiveness of your retrieval system.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ContextualPrecision

            scorer = ContextualPrecision(threshold=0.7)
            feedback = scorer(
                trace=trace
            )  # trace contains input, expected_output, and retrieval_context
    """

    metric_name: ClassVar[str] = "ContextualPrecision"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextualRelevancy(DeepEvalScorer):
    """
    Evaluates the overall relevance of information in the retrieval context.

    This metric determines what fraction of the retrieval context is relevant to the input.
    It helps assess whether your retriever is returning focused, relevant information or
    including too much irrelevant content.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.deepeval import ContextualRelevancy

            scorer = ContextualRelevancy(threshold=0.6)
            feedback = scorer(trace=trace)  # trace contains input and retrieval_context
    """

    metric_name: ClassVar[str] = "ContextualRelevancy"

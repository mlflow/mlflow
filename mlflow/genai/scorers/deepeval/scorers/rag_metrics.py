"""RAG (Retrieval-Augmented Generation) metrics for DeepEval integration."""

from __future__ import annotations

from mlflow.genai.scorers.deepeval import DeepEvalScorer


class AnswerRelevancy(DeepEvalScorer):
    """
    Evaluates whether the output is relevant to the input.

    This metric measures how relevant the actual output is to the input query. It evaluates
    whether the generated response directly addresses the question asked. Higher scores indicate
    better relevance to the input.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.rag_metrics import AnswerRelevancy
        >>> scorer = AnswerRelevancy(threshold=0.7, model="openai:/gpt-4")
        >>> feedback = scorer(
        ...     inputs="What is the capital of France?",
        ...     outputs="Paris is the capital of France.",
        ... )
        >>> print(feedback.value)  # CategoricalRating.YES or CategoricalRating.NO
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="AnswerRelevancy",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class Faithfulness(DeepEvalScorer):
    """
    Evaluates whether the output is factually consistent with the retrieval context.

    This metric determines if claims in the output can be inferred from the provided context.
    It helps detect hallucinations by checking if the generated content is grounded in the
    retrieved documents.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.rag_metrics import Faithfulness
        >>> scorer = Faithfulness(threshold=0.8, model="databricks")
        >>> feedback = scorer(trace=trace)  # trace contains outputs and retrieval_context
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="Faithfulness",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class ContextualRecall(DeepEvalScorer):
    """
    Evaluates whether the retrieval context contains all necessary information.

    This metric measures how much of the expected output can be attributed to the nodes in
    the retrieval context. It assesses the quality of the retriever by checking if all
    required information is present in the retrieved documents.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.rag_metrics import ContextualRecall
        >>> scorer = ContextualRecall(model="databricks")
        >>> feedback = scorer(trace=trace)  # trace contains expected_output and retrieval_context
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="ContextualRecall",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class ContextualPrecision(DeepEvalScorer):
    """
    Evaluates whether relevant nodes in the retrieval context are ranked higher than
    irrelevant ones.

    This metric assesses the quality of your retriever by checking if the most relevant
    retrieved context are ranked higher than less relevant ones. It helps evaluate the
    ranking effectiveness of your retrieval system.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.rag_metrics import ContextualPrecision
        >>> scorer = ContextualPrecision(threshold=0.7)
        >>> feedback = scorer(
        ...     trace=trace
        ... )  # trace contains input, expected_output, and retrieval_context
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="ContextualPrecision",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )


class ContextualRelevancy(DeepEvalScorer):
    """
    Evaluates the overall relevance of information in the retrieval context.

    This metric determines what fraction of the retrieval context is relevant to the input.
    It helps assess whether your retriever is returning focused, relevant information or
    including too much irrelevant content.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)
        model: Model URI (e.g., "openai:/gpt-4", "databricks", "databricks:/endpoint")
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        >>> from mlflow.genai.scorers.deepeval.rag_metrics import ContextualRelevancy
        >>> scorer = ContextualRelevancy(threshold=0.6)
        >>> feedback = scorer(trace=trace)  # trace contains input and retrieval_context
    """

    def __init__(
        self,
        threshold: float | None = None,
        model: str | None = None,
        include_reason: bool = True,
        **kwargs,
    ):
        super().__init__(
            metric_name="ContextualRelevancy",
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            **kwargs,
        )

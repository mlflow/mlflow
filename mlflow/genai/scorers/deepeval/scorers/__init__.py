"""DeepEval metric scorers organized by category."""

from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.deepeval import DeepEvalScorer
from mlflow.genai.scorers.deepeval.scorers.agentic_metrics import (
    ArgumentCorrectness,
    PlanAdherence,
    PlanQuality,
    StepEfficiency,
    TaskCompletion,
    ToolCorrectness,
)
from mlflow.genai.scorers.deepeval.scorers.conversational_metrics import (
    ConversationCompleteness,
    GoalAccuracy,
    KnowledgeRetention,
    RoleAdherence,
    ToolUse,
    TopicAdherence,
    TurnRelevancy,
)
from mlflow.genai.scorers.deepeval.scorers.rag_metrics import (
    AnswerRelevancy,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
    Faithfulness,
)
from mlflow.genai.scorers.deepeval.scorers.safety_metrics import (
    Bias,
    Misuse,
    NonAdvice,
    PIILeakage,
    RoleViolation,
    Toxicity,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


# General-purpose metrics
@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class Hallucination(DeepEvalScorer):
    """
    Detects hallucinations where the LLM fabricates information not present in the context.

    Args:
        threshold: Maximum score threshold for passing (range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            scorer = Hallucination(threshold=0.3)
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "Hallucination"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class Summarization(DeepEvalScorer):
    """
    Evaluates the quality and accuracy of text summarization.

    Args:
        threshold: Minimum score threshold for passing (range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            scorer = Summarization(threshold=0.7)
            feedback = scorer(inputs="Long text...", outputs="Summary...")
    """

    metric_name: ClassVar[str] = "Summarization"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class JsonCorrectness(DeepEvalScorer):
    """
    Validates JSON output against an expected schema.

    Note: Requires `expected_schema` parameter in expectations dict.

    Args:
        threshold: Minimum score threshold for passing (range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            scorer = JsonCorrectness(threshold=0.8)
            feedback = scorer(
                outputs='{"name": "John"}',
                expectations={"expected_schema": {...}},
            )
    """

    metric_name: ClassVar[str] = "JsonCorrectness"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class PromptAlignment(DeepEvalScorer):
    """
    Measures how well the output aligns with instructions given in the prompt.

    Args:
        threshold: Minimum score threshold for passing (range: 0.0-1.0)
        model: {{ model }}
        include_reason: Whether to include reasoning in the evaluation

    Examples:
        .. code-block:: python

            scorer = PromptAlignment(threshold=0.7)
            feedback = scorer(inputs="Instructions...", outputs="Response...")
    """

    metric_name: ClassVar[str] = "PromptAlignment"


@experimental(version="3.8.0")
class ExactMatch(DeepEvalScorer):
    """
    Performs exact string matching between output and expected output.

    Args:
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)

    Examples:
        .. code-block:: python

            scorer = ExactMatch()
            feedback = scorer(
                outputs="Paris",
                expectations={"expected_output": "Paris"},
            )
    """

    metric_name: ClassVar[str] = "ExactMatch"

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs,
    ):
        self._validate_kwargs(**kwargs)
        super().__init__(
            metric_name=self.metric_name,
            model=None,
            threshold=threshold,
            **kwargs,
        )


@experimental(version="3.8.0")
class PatternMatch(DeepEvalScorer):
    """
    Performs regex pattern matching on the output.

    Args:
        pattern: Regex pattern to match against the output
        threshold: Minimum score threshold for passing (default: 0.5, range: 0.0-1.0)

    Examples:
        .. code-block:: python

            scorer = PatternMatch(pattern=r"\\d{3}-\\d{3}-\\d{4}")
            feedback = scorer(outputs="Phone: 555-123-4567")
    """

    metric_name: ClassVar[str] = "PatternMatch"

    def __init__(
        self,
        pattern: str,
        threshold: float = 0.5,
        **kwargs,
    ):
        self._validate_kwargs(**kwargs)
        super().__init__(
            metric_name=self.metric_name,
            model=None,
            threshold=threshold,
            pattern=pattern,
            **kwargs,
        )


__all__ = [
    # RAG metrics
    "AnswerRelevancy",
    "Faithfulness",
    "ContextualRecall",
    "ContextualPrecision",
    "ContextualRelevancy",
    # Agentic metrics
    "TaskCompletion",
    "ToolCorrectness",
    "ArgumentCorrectness",
    "StepEfficiency",
    "PlanAdherence",
    "PlanQuality",
    # Conversational metrics
    "TurnRelevancy",
    "RoleAdherence",
    "KnowledgeRetention",
    "ConversationCompleteness",
    "GoalAccuracy",
    "ToolUse",
    "TopicAdherence",
    # Safety metrics
    "Bias",
    "Toxicity",
    "NonAdvice",
    "Misuse",
    "PIILeakage",
    "RoleViolation",
    # General metrics
    "Hallucination",
    "Summarization",
    "JsonCorrectness",
    "PromptAlignment",
    # Deterministic metrics
    "ExactMatch",
    "PatternMatch",
]

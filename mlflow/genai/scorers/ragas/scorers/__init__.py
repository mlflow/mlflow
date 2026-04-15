from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.genai.scorers.ragas.scorers.agentic_metrics import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
    ToolCallAccuracy,
    ToolCallF1,
    TopicAdherence,
)
from mlflow.genai.scorers.ragas.scorers.comparison_metrics import (
    BleuScore,
    ChrfScore,
    ExactMatch,
    FactualCorrectness,
    NonLLMStringSimilarity,
    RougeScore,
    StringPresence,
)
from mlflow.genai.scorers.ragas.scorers.rag_metrics import (
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    SemanticSimilarity,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class AspectCritic(RagasScorer):
    """
    Evaluates the output based on specific aspects or criteria.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters (e.g., name, definition)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import AspectCritic

            scorer = AspectCritic(
                model="openai:/gpt-4",
                name="helpfulness",
                definition="Does the response help answer the question?",
            )
            feedback = scorer(inputs="What is MLflow?", outputs="MLflow is a platform...")
    """

    metric_name: ClassVar[str] = "AspectCritic"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class DiscreteMetric(RagasScorer):
    """
    Evaluates the output based on a custom prompt with discrete scoring.

    This metric allows you to define a custom evaluation prompt that returns
    discrete values (e.g., "pass"/"fail", scores 0-10, etc.).

    Args:
        name: Name for this metric instance.
        prompt: Custom prompt template for evaluation (required). Should contain
            placeholders for evaluation inputs that will be formatted at runtime.
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters.

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import DiscreteMetric

            scorer = DiscreteMetric(
                name="clarity",
                prompt='''Rate the clarity of the response on a scale of 0-10.
                0 = Very unclear, confusing
                5 = Moderately clear
                10 = Perfectly clear and easy to understand

                Response: {response}

                Respond with only the number (0-10).''',
                allowed_values=[num for num in range(10)],
            )
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "DiscreteMetric"

    # override to have name and prompt as required parameters
    def __init__(
        self,
        name: str,
        prompt: str,
        **metric_kwargs,
    ):
        super().__init__(
            metric_name=self.metric_name,
            name=name,
            prompt=prompt,
            **metric_kwargs,
        )


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class RubricsScore(RagasScorer):
    """
    Evaluates the output based on a predefined rubric.

    This metric uses a rubric (set of criteria with descriptions and scores) to evaluate
    the output in a structured way.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters (e.g., rubrics)

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import RubricsScore

            rubrics = {
                "1": "The response is entirely incorrect.",
                "2": "The response contains partial accuracy.",
                "3": "The response is mostly accurate but lacks clarity.",
                "4": "The response is accurate and clear with minor omissions.",
                "5": "The response is completely accurate and clear.",
            }
            scorer = RubricsScore(rubrics=rubrics)
            feedback = scorer(inputs="What is AI?", outputs="AI is artificial intelligence")
    """

    metric_name: ClassVar[str] = "RubricsScore"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class InstanceSpecificRubrics(RagasScorer):
    """
    Evaluates the output based on instance-specific rubrics.

    Unlike RubricsScore which uses one rubric for all evaluations, InstanceSpecificRubrics
    allows you to define different rubrics for each evaluation instance.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import InstanceSpecificRubrics

            scorer = InstanceSpecificRubrics(model="openai:/gpt-4")

            # Evaluate relevance with custom rubric
            feedback1 = scorer(
                inputs="How do I handle exceptions in Python?",
                outputs="To handle exceptions in Python, use try and except blocks.",
                expectations={
                    "expected_output": "Use try, except, and optionally else blocks.",
                    "rubrics": {
                        "0": "The response is off-topic or irrelevant.",
                        "1": "The response is fully relevant and focused.",
                    },
                },
            )

            # Evaluate code efficiency with different rubric
            feedback2 = scorer(
                inputs="Create a list of squares for numbers 1 through 5",
                outputs="squares = []\\nfor i in range(1, 6):\\n    squares.append(i**2)",
                expectations={
                    "expected_output": "squares = [i**2 for i in range(1, 6)]",
                    "rubrics": {
                        "0": "Inefficient code with performance issues.",
                        "1": "Efficient and optimized code.",
                    },
                },
            )
    """

    metric_name: ClassVar[str] = "InstanceSpecificRubrics"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class SummarizationScore(RagasScorer):
    """
    Evaluates the quality and accuracy of text summarization.

    This metric assesses whether the summary captures the key points of the source text
    while being concise and coherent.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import SummarizationScore

            scorer = SummarizationScore(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "SummarizationScore"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class AnswerAccuracy(RagasScorer):
    """
    Evaluates the accuracy of the answer compared to the expectations.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import AnswerAccuracy

            scorer = AnswerAccuracy(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "AnswerAccuracy"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class ContextRelevance(RagasScorer):
    """
    Evaluates the relevance of retrieved contexts to the user's question.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ContextRelevance

            scorer = ContextRelevance(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ContextRelevance"


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class ResponseGroundedness(RagasScorer):
    """
    Evaluates whether the response is grounded in the retrieved contexts.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ResponseGroundedness

            scorer = ResponseGroundedness(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ResponseGroundedness"


__all__ = [
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

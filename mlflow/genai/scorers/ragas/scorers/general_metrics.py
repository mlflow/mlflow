from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


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
                "score1_description": "The response is entirely incorrect.",
                "score2_description": "The response contains partial accuracy.",
                "score3_description": "The response is mostly accurate but lacks clarity.",
                "score4_description": "The response is accurate and clear with minor omissions.",
                "score5_description": "The response is completely accurate and clear.",
            }
            scorer = RubricsScore(rubrics=rubrics)
            feedback = scorer(inputs="What is AI?", outputs="AI is artificial intelligence")
    """

    metric_name: ClassVar[str] = "RubricsScore"

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class InstanceRubrics(RagasScorer):
    """
    Evaluates the output based on instance-specific rubrics.

    Unlike RubricsScore which uses one rubric for all evaluations, InstanceRubrics allows
    you to define different rubrics for each evaluation instance.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import InstanceRubrics

            scorer = InstanceRubrics(model="openai:/gpt-4")

            # Evaluate relevance with custom rubric
            feedback1 = scorer(
                inputs="How do I handle exceptions in Python?",
                outputs="To handle exceptions in Python, use try and except blocks.",
                expectations={
                    "expected_output": "Use try, except, and optionally else blocks.",
                    "rubrics": {
                        "score0_description": "The response is off-topic or irrelevant.",
                        "score1_description": "The response is fully relevant and focused.",
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
                        "score0_description": "Inefficient code with performance issues.",
                        "score1_description": "Efficient and optimized code.",
                    },
                },
            )
    """

    metric_name: ClassVar[str] = "InstanceRubrics"

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)

from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)

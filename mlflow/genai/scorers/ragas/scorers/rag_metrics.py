from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextPrecision(RagasScorer):
    """
    Evaluates whether relevant nodes in the retrieval context are ranked higher than
    irrelevant ones.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ContextPrecision

            scorer = ContextPrecision(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ContextPrecision"


@experimental(version="3.8.0")
class NonLLMContextPrecisionWithReference(RagasScorer):
    """
    Deterministic metric that evaluates context precision using non-LLM methods using expectations.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import NonLLMContextPrecisionWithReference

            scorer = NonLLMContextPrecisionWithReference()
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "NonLLMContextPrecisionWithReference"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextRecall(RagasScorer):
    """
    Evaluates whether the retrieval context contains all necessary information.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ContextRecall

            scorer = ContextRecall(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ContextRecall"


@experimental(version="3.8.0")
class NonLLMContextRecall(RagasScorer):
    """
    Deterministic metric that evaluates context recall without using an LLM.

    Args:
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import NonLLMContextRecall

            scorer = NonLLMContextRecall()
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "NonLLMContextRecall"

    def __init__(self, **metric_kwargs):
        self._validate_kwargs(**metric_kwargs)
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextEntityRecall(RagasScorer):
    """
    Evaluates entity recall in the retrieval context.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import ContextEntityRecall

            scorer = ContextEntityRecall(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "ContextEntityRecall"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class NoiseSensitivity(RagasScorer):
    """
    Evaluates how sensitive the model is to noise in the retrieval context.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import NoiseSensitivity

            scorer = NoiseSensitivity(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "NoiseSensitivity"


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class Faithfulness(RagasScorer):
    """
    Evaluates whether the output is factually consistent with the retrieval context.

    Args:
        model: {{ model }}
        **metric_kwargs: Additional metric-specific parameters

    Examples:
        .. code-block:: python

            from mlflow.genai.scorers.ragas import Faithfulness

            scorer = Faithfulness(model="openai:/gpt-4")
            feedback = scorer(trace=trace)
    """

    metric_name: ClassVar[str] = "Faithfulness"

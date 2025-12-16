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

    This metric assesses the quality of your retriever by checking if the most relevant
    retrieved context are ranked higher than less relevant ones.

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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


@experimental(version="3.8.0")
class NonLLMContextPrecisionWithReference(RagasScorer):
    """
    Evaluates context precision using non-LLM methods with reference.

    This is a deterministic metric that evaluates context precision without using an LLM,
    requiring a reference for comparison.

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
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextRecall(RagasScorer):
    """
    Evaluates whether the retrieval context contains all necessary information.

    This metric measures how much of the expected output can be attributed to the nodes in
    the retrieval context. It assesses the quality of the retriever by checking if all
    required information is present in the retrieved documents.

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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


@experimental(version="3.8.0")
class NonLLMContextRecall(RagasScorer):
    """
    Evaluates context recall using non-LLM methods.

    This is a deterministic metric that evaluates context recall without using an LLM.

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
        super().__init__(metric_name=self.metric_name, model=None, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ContextEntityRecall(RagasScorer):
    """
    Evaluates entity recall in the retrieval context.

    This metric measures how many entities from the ground truth are present in the
    retrieved context.

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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class NoiseSensitivity(RagasScorer):
    """
    Evaluates how sensitive the model is to noise in the retrieval context.

    This metric assesses whether the model is distracted by irrelevant information in
    the retrieved documents.

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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)


@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class Faithfulness(RagasScorer):
    """
    Evaluates whether the output is factually consistent with the retrieval context.

    This metric determines if claims in the output can be inferred from the provided context.
    It helps detect hallucinations by checking if the generated content is grounded in the
    retrieved documents.

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

    def __init__(self, model: str | None = None, **metric_kwargs):
        super().__init__(metric_name=self.metric_name, model=model, **metric_kwargs)

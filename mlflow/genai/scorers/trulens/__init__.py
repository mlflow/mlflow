"""
TruLens evaluation framework integration for MLflow GenAI scorers.

This module wraps TruLens feedback functions as MLflow scorers.

Available scorers:
    - Groundedness: Evaluates if outputs are grounded in context
    - ContextRelevance: Evaluates context relevance to query
    - AnswerRelevance: Evaluates answer relevance to query
    - Coherence: Evaluates logical flow of outputs

Installation:
    pip install trulens trulens-providers-openai

For LiteLLM provider support:
    pip install trulens trulens-providers-litellm

Example:
    >>> from mlflow.genai.scorers.trulens import Groundedness, get_scorer
    >>>
    >>> scorer = Groundedness(model="openai:/gpt-4")
    >>> feedback = scorer(
    ...     outputs="Paris is the capital.",
    ...     expectations={"context": "France's capital is Paris."},
    ... )
    >>>
    >>> # Or use get_scorer for dynamic metric selection
    >>> scorer = get_scorer("ContextRelevance", model="databricks")
"""

from mlflow.genai.scorers.trulens.trulens import (
    AnswerRelevance,
    Coherence,
    ContextRelevance,
    Groundedness,
    TruLensScorer,
    get_scorer,
)

__all__ = [
    # Core classes
    "TruLensScorer",
    "get_scorer",
    # Metric scorers
    "Groundedness",
    "ContextRelevance",
    "AnswerRelevance",
    "Coherence",
]

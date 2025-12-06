"""
TruLens evaluation framework integration for MLflow GenAI scorers.

This module wraps TruLens feedback functions as MLflow scorers, enabling use of
TruLens' groundedness, context relevance, answer relevance, and coherence metrics
within the MLflow evaluation framework.

**Available Scorers:**

- ``TruLensGroundednessScorer``: Evaluates if outputs are grounded in context
- ``TruLensContextRelevanceScorer``: Evaluates context relevance to query
- ``TruLensAnswerRelevanceScorer``: Evaluates answer relevance to query
- ``TruLensCoherenceScorer``: Evaluates logical flow of outputs

**Installation:**
    pip install trulens trulens-providers-openai

For LiteLLM provider support:
    pip install trulens trulens-providers-litellm

**Example:**

.. code-block:: python

    from mlflow.genai.scorers import TruLensGroundednessScorer

    scorer = TruLensGroundednessScorer()
    result = scorer(
        outputs="Paris is the capital.",
        context="France's capital is Paris.",
    )
    print(result.value)  # Score between 0.0 and 1.0

For more information on TruLens, see:
https://www.trulens.org/
"""

from mlflow.genai.scorers.trulens.trulens import (
    TruLensAnswerRelevanceScorer,
    TruLensCoherenceScorer,
    TruLensContextRelevanceScorer,
    TruLensGroundednessScorer,
)

__all__ = [
    "TruLensGroundednessScorer",
    "TruLensContextRelevanceScorer",
    "TruLensAnswerRelevanceScorer",
    "TruLensCoherenceScorer",
]

"""
Phoenix (Arize) evaluation framework integration for MLflow GenAI scorers.

This module wraps Phoenix evaluators as MLflow scorers, enabling use of Phoenix's
hallucination detection, relevance, toxicity, QA, and summarization metrics
within the MLflow evaluation framework.

**Available Scorers:**

- ``PhoenixHallucinationScorer``: Detects hallucinations (1.0=factual, 0.0=hallucinated)
- ``PhoenixRelevanceScorer``: Evaluates context relevance (1.0=relevant, 0.0=irrelevant)
- ``PhoenixToxicityScorer``: Detects toxic content (1.0=safe, 0.0=toxic)
- ``PhoenixQAScorer``: Evaluates QA correctness (1.0=correct, 0.0=incorrect)
- ``PhoenixSummarizationScorer``: Evaluates summary quality (1.0=good, 0.0=poor)

**Installation:**
    pip install arize-phoenix-evals

**Example:**

.. code-block:: python

    from mlflow.genai.scorers import PhoenixHallucinationScorer

    scorer = PhoenixHallucinationScorer()
    result = scorer(
        inputs={"query": "What is Python?"},
        outputs="Python is a programming language.",
        context="Python is a high-level programming language.",
    )
    print(result.value)  # 1.0 (factual) or 0.0 (hallucinated)

For more information on Phoenix evaluators, see:
https://docs.arize.com/phoenix/evaluation/evals
"""

from mlflow.genai.scorers.phoenix.phoenix import (
    PhoenixHallucinationScorer,
    PhoenixQAScorer,
    PhoenixRelevanceScorer,
    PhoenixSummarizationScorer,
    PhoenixToxicityScorer,
)

__all__ = [
    "PhoenixHallucinationScorer",
    "PhoenixRelevanceScorer",
    "PhoenixToxicityScorer",
    "PhoenixQAScorer",
    "PhoenixSummarizationScorer",
]

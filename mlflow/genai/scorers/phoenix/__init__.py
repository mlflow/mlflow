"""
Phoenix (Arize) integration for MLflow.

This module provides integration with Phoenix evaluators, allowing them to be used
with MLflow's scorer interface.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.phoenix import get_scorer

    scorer = get_scorer("Hallucination", model="openai:/gpt-4")
    feedback = scorer(
        inputs="What is MLflow?", outputs="MLflow is a platform...", trace=trace
    )
"""

from mlflow.genai.scorers.phoenix.phoenix import (
    QA,
    Hallucination,
    PhoenixScorer,
    Relevance,
    Summarization,
    Toxicity,
    get_scorer,
)

__all__ = [
    # Core classes
    "PhoenixScorer",
    "get_scorer",
    # Metric scorers
    "Hallucination",
    "Relevance",
    "Toxicity",
    "QA",
    "Summarization",
]

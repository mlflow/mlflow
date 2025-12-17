"""
Phoenix (Arize) evaluation framework integration for MLflow GenAI scorers.

This module wraps Phoenix evaluators as MLflow scorers.

Available scorers:
    - Hallucination: Detects hallucinations
    - Relevance: Evaluates context relevance
    - Toxicity: Detects toxic content
    - QA: Evaluates QA correctness
    - Summarization: Evaluates summary quality

Installation:
    pip install arize-phoenix-evals

Example:
    >>> from mlflow.genai.scorers.phoenix import Hallucination, get_scorer
    >>>
    >>> scorer = Hallucination(model="openai:/gpt-4")
    >>> feedback = scorer(
    ...     inputs="What is Python?",
    ...     outputs="Python is a programming language.",
    ...     expectations={"context": "Python is a high-level programming language."},
    ... )
    >>>
    >>> # Or use get_scorer for dynamic metric selection
    >>> scorer = get_scorer("Relevance", model="databricks")
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

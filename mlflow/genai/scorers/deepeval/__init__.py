"""
DeepEval integration for MLflow.

This module provides integration with DeepEval metrics, allowing them to be used
with MLflow's judge interface.

Example usage:
    >>> from mlflow.genai.scorers.deepeval import get_judge
    >>> judge = get_judge("AnswerRelevancy", threshold=0.7, model="openai:/gpt-4")
    >>> feedback = judge(inputs="What is MLflow?", outputs="MLflow is a platform...")
"""

from mlflow.genai.scorers.deepeval.adapter import DeepEvalAdapter, get_judge

__all__ = [
    "DeepEvalAdapter",
    "get_judge",
]

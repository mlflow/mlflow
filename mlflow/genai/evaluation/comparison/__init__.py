"""Paired statistical comparison of two GenAI evaluation runs."""

from mlflow.genai.evaluation.comparison.comparison import compare_evaluations
from mlflow.genai.evaluation.comparison.entities import ComparisonResult, ScorerComparison

__all__ = ["ComparisonResult", "ScorerComparison", "compare_evaluations"]

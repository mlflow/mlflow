"""Paired statistical comparison of two GenAI evaluation runs."""

from mlflow.genai.evaluation.comparison.comparison import (
    ComparisonResult,
    ScorerComparison,
    compare_evaluations,
)

__all__ = ["ComparisonResult", "ScorerComparison", "compare_evaluations"]

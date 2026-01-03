"""
TruLens agent trace scorers for goal-plan-action alignment evaluation.

This module provides TruLens agent trace evaluators as MLflow scorers, enabling
trace-aware evaluation of LLM agents. See: https://arxiv.org/abs/2510.08847

Example usage:
    >>> from mlflow.genai.scorers import TruLensLogicalConsistencyScorer
    >>> scorer = TruLensLogicalConsistencyScorer()
    >>> feedback = scorer(trace=trace)
"""

from mlflow.genai.scorers.trulens.agent_trace import (
    TruLensExecutionEfficiencyScorer,
    TruLensLogicalConsistencyScorer,
    TruLensPlanAdherenceScorer,
    TruLensPlanQualityScorer,
    TruLensToolCallingScorer,
    TruLensToolSelectionScorer,
)

__all__ = [
    "TruLensLogicalConsistencyScorer",
    "TruLensExecutionEfficiencyScorer",
    "TruLensPlanAdherenceScorer",
    "TruLensPlanQualityScorer",
    "TruLensToolSelectionScorer",
    "TruLensToolCallingScorer",
]

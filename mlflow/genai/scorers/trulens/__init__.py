"""
TruLens evaluation framework integration for MLflow.

This module provides TruLens agent trace evaluators as MLflow scorers, enabling
trace-aware evaluation of LLM agents.

Example usage:

.. code-block:: python

    from mlflow.genai.scorers.trulens import LogicalConsistencyScorer

    scorer = LogicalConsistencyScorer(model="openai:/gpt-4")
    feedback = scorer(trace=trace)
"""

from mlflow.genai.scorers.trulens.agent_trace import (
    ExecutionEfficiencyScorer,
    LogicalConsistencyScorer,
    PlanAdherenceScorer,
    PlanQualityScorer,
    ToolCallingScorer,
    ToolSelectionScorer,
)

__all__ = [
    "LogicalConsistencyScorer",
    "ExecutionEfficiencyScorer",
    "PlanAdherenceScorer",
    "PlanQualityScorer",
    "ToolSelectionScorer",
    "ToolCallingScorer",
]

"""
Base interface for MLflow Insights Store.
"""

from abc import ABC, abstractmethod

from mlflow.insights.models.entities import Census


class InsightsStore(ABC):
    """Abstract base class for insights analytics operations."""

    @abstractmethod
    def generate_census(self, experiment_id: str) -> Census:
        """
        Generate a comprehensive census of trace data.

        Analyzes traces to provide statistical distributions and patterns
        including operational metrics, quality assessments, and dimensional analysis.

        Args:
            experiment_id: The experiment ID to analyze

        Returns:
            Census object containing statistics and distributions
        """

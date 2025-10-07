"""
MLflow Insights Store module.

This module provides the abstract base interface for insights analytics operations.
"""

from mlflow.insights.store.base import InsightsStore
from mlflow.insights.store.sql_insights_store import SqlInsightsStore

__all__ = [
    "InsightsStore",
    "SqlInsightsStore",
]

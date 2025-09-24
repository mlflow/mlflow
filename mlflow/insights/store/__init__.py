"""
MLflow Insights Store for analytics and census operations.
"""

from mlflow.insights.store.base import InsightsStore
from mlflow.insights.store.sql_insights_store import SqlInsightsStore

__all__ = ["InsightsStore", "SqlInsightsStore"]
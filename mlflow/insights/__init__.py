"""
MLflow Insights - Models and utilities for ML investigation and analysis.

This module provides structured models for tracking analyses, hypotheses,
and issues discovered during ML model experimentation and investigation.
"""

from mlflow.insights.constants import (
    INSIGHTS_ANALYSIS_FILE_NAME,
    INSIGHTS_RUN_TAG_NAME_KEY,
    AnalysisStatus,
    HypothesisStatus,
    IssueSeverity,
    IssueStatus,
)
from mlflow.insights.models import (
    Analysis,
    AnalysisSummary,
    EvidenceEntry,
    Hypothesis,
    HypothesisSummary,
    Issue,
    IssueSummary,
)
from mlflow.insights.utils import extract_trace_ids, normalize_evidence

__all__ = [
    # Core models
    "Analysis",
    "Hypothesis",
    "Issue",
    "EvidenceEntry",
    # Summary models
    "AnalysisSummary",
    "HypothesisSummary",
    "IssueSummary",
    # Status enums
    "AnalysisStatus",
    "HypothesisStatus",
    "IssueSeverity",
    "IssueStatus",
    # Constants
    "INSIGHTS_ANALYSIS_FILE_NAME",
    "INSIGHTS_RUN_TAG_NAME_KEY",
    # Utilities
    "extract_trace_ids",
    "normalize_evidence",
]

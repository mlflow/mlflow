"""
Analysis utilities for MLflow tracing.
"""

from dataclasses import dataclass
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject


@dataclass
class TraceFilterCorrelationResult(_MlflowObject):
    """
    Result of calculating correlation between two trace filter conditions.
    
    This class represents the correlation analysis between two trace filters,
    using Normalized Pointwise Mutual Information (NPMI) as the correlation metric.
    
    Args:
        npmi: Normalized Pointwise Mutual Information score (-1 to 1).
              -1 indicates filters never co-occur, 0 indicates independence,
              1 indicates filters always co-occur together.
        confidence_lower: Lower bound of the confidence interval for NPMI.
        confidence_upper: Upper bound of the confidence interval for NPMI.
        filter_string1_count: Number of traces matching the first filter.
        filter_string2_count: Number of traces matching the second filter.
        joint_count: Number of traces matching both filters.
        total_count: Total number of traces in the experiment(s).
    """

    npmi: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    filter_string1_count: int = 0
    filter_string2_count: int = 0
    joint_count: int = 0
    total_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            "npmi": self.npmi,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "filter_string1_count": self.filter_string1_count,
            "filter_string2_count": self.filter_string2_count,
            "joint_count": self.joint_count,
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceFilterCorrelationResult":
        """Create a TraceFilterCorrelationResult from a dictionary representation."""
        return cls(
            npmi=data["npmi"],
            confidence_lower=data.get("confidence_lower"),
            confidence_upper=data.get("confidence_upper"),
            filter_string1_count=data.get("filter_string1_count", 0),
            filter_string2_count=data.get("filter_string2_count", 0),
            joint_count=data.get("joint_count", 0),
            total_count=data.get("total_count", 0),
        )
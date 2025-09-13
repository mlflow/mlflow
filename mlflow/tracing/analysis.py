import math
from dataclasses import dataclass

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import CalculateTraceFilterCorrelation


@dataclass
class TraceFilterCorrelationResult(_MlflowObject):
    """
    Result of calculating correlation between two trace filter conditions.

    This class represents the correlation analysis between two trace filters,
    using Normalized Pointwise Mutual Information (NPMI) as the correlation metric.

    NPMI ranges from -1 to 1:
    - -1: Perfect negative correlation (filters never co-occur)
    - 0: Independence (filters occur independently)
    - 1: Perfect positive correlation (filters always co-occur)
    - NaN: Undefined (when one or both filters have zero matches)

    Attributes:
        npmi: Normalized Pointwise Mutual Information score (unsmoothed).
              Returns NaN when undefined (e.g., when filter1_count=0 or filter2_count=0).
              Returns -1.0 when filters never co-occur but both have support.
              Otherwise returns a value in [-1, 1].
        npmi_smoothed: NPMI calculated with Jeffreys prior smoothing (alpha=0.5).
                      More robust for small sample sizes and confidence interval estimation.
                      Returns NaN when undefined.
        filter1_count: Number of traces matching the first filter.
        filter2_count: Number of traces matching the second filter.
        joint_count: Number of traces matching both filters.
        total_count: Total number of traces in the experiment(s).
        confidence_lower: Lower bound of the confidence interval for NPMI (optional).
        confidence_upper: Upper bound of the confidence interval for NPMI (optional).
    """

    npmi: float
    filter1_count: int
    filter2_count: int
    joint_count: int
    total_count: int
    npmi_smoothed: float | None = None
    confidence_lower: float | None = None
    confidence_upper: float | None = None

    @classmethod
    def from_proto(cls, proto):
        """
        Create a TraceFilterCorrelationResult from a protobuf response.

        Args:
            proto: CalculateTraceFilterCorrelation.Response protobuf message

        Returns:
            TraceFilterCorrelationResult instance
        """
        return cls(
            npmi=proto.npmi if proto.HasField("npmi") else float("nan"),
            npmi_smoothed=proto.npmi_smoothed if proto.HasField("npmi_smoothed") else None,
            filter1_count=proto.filter1_count,
            filter2_count=proto.filter2_count,
            joint_count=proto.joint_count,
            total_count=proto.total_count,
        )

    def to_proto(self):
        """
        Convert this result to a protobuf response message.

        Returns:
            CalculateTraceFilterCorrelation.Response protobuf message
        """

        response = CalculateTraceFilterCorrelation.Response()

        if self.npmi is not None and not math.isnan(self.npmi):
            response.npmi = self.npmi

        if self.npmi_smoothed is not None and not math.isnan(self.npmi_smoothed):
            response.npmi_smoothed = self.npmi_smoothed

        response.filter1_count = self.filter1_count
        response.filter2_count = self.filter2_count
        response.joint_count = self.joint_count
        response.total_count = self.total_count

        return response

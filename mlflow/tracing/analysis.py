from dataclasses import dataclass


@dataclass
class TraceFilterCorrelationResult:
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

import math
from dataclasses import dataclass

# Recommended smoothing parameter for NPMI calculation
# Using Jeffreys prior (alpha=0.5) to minimize bias while providing robust estimates
JEFFREYS_PRIOR = 0.5


@dataclass
class TraceCorrelationCounts:
    """
    Count statistics for trace correlation analysis.

    This dataclass encapsulates the four fundamental counts needed
    for correlation analysis between two trace filters.

    Attributes:
        total_count: Total number of traces in the experiment(s)
        filter1_count: Number of traces matching filter 1
        filter2_count: Number of traces matching filter 2
        joint_count: Number of traces matching both filters
    """

    total_count: int
    filter1_count: int
    filter2_count: int
    joint_count: int


@dataclass
class NPMIResult:
    """
    Result of NPMI calculation containing both unsmoothed and smoothed values.

    Attributes:
        npmi: Unsmoothed NPMI value with explicit -1.0 rule for zero joint count.
              Returns NaN when undefined (e.g., when filter1_count=0 or filter2_count=0).
        npmi_smoothed: NPMI calculated with Jeffreys prior smoothing (alpha=0.5).
                      More robust for small sample sizes and confidence interval estimation.
    """

    npmi: float
    npmi_smoothed: float | None


def calculate_npmi_from_counts(
    joint_count: int,
    filter1_count: int,
    filter2_count: int,
    total_count: int,
) -> NPMIResult:
    """
    Calculate both unsmoothed and smoothed NPMI from count data.

    Implements the recommended policy for NPMI calculation:
    - Returns NaN (undefined) when either filter has zero support (n1=0 or n2=0)
    - Returns -1.0 for unsmoothed when filters never co-occur despite both having support
    - Calculates smoothed version using Jeffreys prior for robustness

    NPMI measures the association between two events, normalized to [-1, 1]:
    - -1: Perfect negative correlation (events never co-occur)
    - 0: Independence (events occur independently)
    - 1: Perfect positive correlation (events always co-occur)
    - NaN: Undefined (when one or both events have zero count)

    Args:
        joint_count: Number of times both events occur together
        filter1_count: Number of times event 1 occurs
        filter2_count: Number of times event 2 occurs
        total_count: Total number of observations

    Returns:
        NPMIResult containing both unsmoothed and smoothed NPMI values.

    Examples:
        >>> result = calculate_npmi_from_counts(10, 20, 15, 100)
        >>> result.npmi  # Unsmoothed value
        >>> result.npmi_smoothed  # Smoothed value
    """
    # No population
    if total_count <= 0:
        return NPMIResult(npmi=float("nan"), npmi_smoothed=float("nan"))

    # Return NaN if either filter has zero support
    if filter1_count == 0 or filter2_count == 0:
        return NPMIResult(npmi=float("nan"), npmi_smoothed=float("nan"))

    n11 = joint_count  # Both occur
    n10 = filter1_count - joint_count  # Only filter1
    n01 = filter2_count - joint_count  # Only filter2
    n00 = total_count - filter1_count - filter2_count + joint_count  # Neither

    if min(n11, n10, n01, n00) < 0:
        # Inconsistent counts, return undefined
        return NPMIResult(npmi=float("nan"), npmi_smoothed=float("nan"))

    # Calculate unsmoothed NPMI with explicit -1.0 rule
    if joint_count == 0 and filter1_count > 0 and filter2_count > 0:
        npmi_unsmoothed = -1.0
    else:
        npmi_unsmoothed = _calculate_npmi_core(n11, n10, n01, n00, smoothing=0)

    # Calculate smoothed NPMI for robustness
    npmi_smoothed = _calculate_npmi_core(n11, n10, n01, n00, smoothing=JEFFREYS_PRIOR)

    return NPMIResult(npmi=npmi_unsmoothed, npmi_smoothed=npmi_smoothed)


def _calculate_npmi_core(
    n11: float,
    n10: float,
    n01: float,
    n00: float,
    smoothing: float = 0,
) -> float:
    """
    Core NPMI calculation with optional smoothing.

    Internal function that performs the actual NPMI calculation
    on a 2x2 contingency table with optional additive smoothing.

    Args:
        n11: Count of both events occurring
        n10: Count of only event 1 occurring
        n01: Count of only event 2 occurring
        n00: Count of neither event occurring
        smoothing: Additive smoothing parameter (0 for no smoothing)

    Returns:
        NPMI value in [-1, 1], or NaN if undefined.
    """
    n11_s = n11 + smoothing
    n10_s = n10 + smoothing
    n01_s = n01 + smoothing
    n00_s = n00 + smoothing

    N = n11_s + n10_s + n01_s + n00_s
    n1 = n11_s + n10_s  # Total event 1 count
    n2 = n11_s + n01_s  # Total event 2 count

    # NB: When marginals are zero (degenerate cases where no events occur), we return NaN
    # rather than forcing a sentinel value like -1. This is mathematically correct since
    # PMI is undefined when P(x)=0 or P(y)=0 (division by zero). NaN properly represents
    # this undefined state and can be handled by our RPC layer, providing a more accurate
    # signal than an arbitrary sentinel value.
    if n1 <= 0 or n2 <= 0 or n11_s <= 0:
        return float("nan")

    # Handle perfect co-occurrence - check pre-smoothing values
    # With smoothing, n11_s == N is never true since smoothing adds mass to other cells
    if n10 == 0 and n01 == 0 and n00 == 0:
        # Perfect co-occurrence: both events always occur together
        return 1.0

    # Calculate PMI using log-space arithmetic for numerical stability
    # PMI = log(P(x,y) / (P(x) * P(y))) = log(n11*N / (n1*n2))
    log_n11 = math.log(n11_s)
    log_N = math.log(N)
    log_n1 = math.log(n1)
    log_n2 = math.log(n2)

    pmi = (log_n11 + log_N) - (log_n1 + log_n2)

    # Normalize by -log(P(x,y)) to get NPMI
    denominator = -(log_n11 - log_N)  # -log(n11/N)

    npmi = pmi / denominator

    # Clamp to [-1, 1] to handle floating point errors
    return max(-1.0, min(1.0, npmi))


def calculate_smoothed_npmi(
    joint_count: int,
    filter1_count: int,
    filter2_count: int,
    total_count: int,
    smoothing: float = JEFFREYS_PRIOR,
) -> float:
    """
    Calculate smoothed NPMI for confidence interval estimation.

    This function applies additive smoothing (Jeffreys prior by default) to all cells
    of the contingency table. Used for uncertainty quantification via Dirichlet sampling.

    Args:
        joint_count: Number of times both events occur together
        filter1_count: Number of times event 1 occurs
        filter2_count: Number of times event 2 occurs
        total_count: Total number of observations
        smoothing: Additive smoothing parameter (default: JEFFREYS_PRIOR = 0.5)

    Returns:
        Smoothed NPMI value in [-1, 1], or NaN if undefined.
    """
    if total_count <= 0:
        return float("nan")

    n11 = joint_count
    n10 = filter1_count - joint_count
    n01 = filter2_count - joint_count
    n00 = total_count - filter1_count - filter2_count + joint_count

    if min(n11, n10, n01, n00) < 0:
        return float("nan")

    return _calculate_npmi_core(n11, n10, n01, n00, smoothing)

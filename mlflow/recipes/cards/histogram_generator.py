"""
Generates facets_overview histogram message for numeric features.
"""

from mlflow.protos.facet_feature_statistics_pb2 import Histogram


def generate_equal_height_histogram(quantiles, num_buckets: int) -> Histogram:
    """
    Generates the equal height histogram from the input quantiles. The quantiles are assumed to
    be ordered and corresponding to equal distant percentiles.

    Args:
        quantiles: The quantiles that capture the frequency distribution.
        num_buckets: The number of buckets in the generated equal height histogram.

    Returns:
        An equal height histogram or None if inputs are invalid.

    """
    if (len(quantiles) < 3) or ((len(quantiles) - 1) % num_buckets != 0):
        return None

    histogram = Histogram()
    histogram.type = Histogram.HistogramType.QUANTILES
    step = (len(quantiles) - 1) // num_buckets
    for low_index in range(0, len(quantiles) - step, step):
        high_index = low_index + step
        histogram.buckets.append(
            Histogram.Bucket(low_value=quantiles[low_index], high_value=quantiles[high_index])
        )

    return histogram


def generate_equal_width_histogram(quantiles, num_buckets: int, total_freq: float) -> Histogram:
    """
    Generates the equal width histogram from the input quantiles and total frequency. The
    quantiles are assumed to be ordered and corresponding to equal distant percentiles.

    Args:
        quantiles: The quantiles that capture the frequency distribution.
        num_buckets: The number of buckets in the generated histogram.
        total_freq: The total frequency (=count of rows).

    Returns:
        Equal width histogram or None if inputs are invalid.
    """
    if len(quantiles) < 2 or num_buckets <= 0 or total_freq <= 0:
        return None

    min_val = quantiles[0]
    max_val = quantiles[-1]

    # If all values are the same, the width of all buckets will be 1 as fixed,
    # except the bucket that contains the real value. The width of that will be 0.
    histogram = Histogram()
    histogram.type = Histogram.HistogramType.STANDARD
    if min_val == max_val:
        half_buckets = num_buckets // 2
        bucket_left = min_val - half_buckets
        for i in range(num_buckets):
            if i == half_buckets:
                histogram.buckets.append(
                    Histogram.Bucket(
                        low_value=bucket_left, high_value=bucket_left, sample_count=total_freq
                    )
                )
            else:
                histogram.buckets.append(
                    Histogram.Bucket(
                        low_value=bucket_left, high_value=bucket_left + 1, sample_count=0
                    )
                )
                bucket_left += 1
    else:
        bucket_width = (max_val - min_val) / num_buckets
        for i in range(num_buckets):
            bucket_left = min_val + i * bucket_width
            bucket_right = bucket_left + bucket_width
            histogram.buckets.append(
                _generate_equal_width_histogram_internal(
                    bucket_left=bucket_left,
                    bucket_right=bucket_right,
                    quantiles=quantiles,
                    total_freq=total_freq,
                )
            )

    return histogram


def _generate_equal_width_histogram_internal(
    bucket_left: float,
    bucket_right: float,
    quantiles,
    total_freq: float,
) -> Histogram.Bucket:
    """
    Generates a histogram bucket given the bucket range, the quantiles and the total frequency.

    Args:
        bucket_left: Bucket left boundary.
        bucket_right: Bucket right boundary.
        quantiles: The quantiles that capture the frequency distribution.
        total_freq: The total frequency (=count of rows).

    Returns:
        The histogram bucket corresponding to the inputs.
    """
    max_val = quantiles[-1]
    bucket_freq = 0.0
    quantile_freq = total_freq / (len(quantiles) - 1)
    for i in range(len(quantiles) - 1):
        quantile_low = quantiles[i]
        quantile_high = quantiles[i + 1]
        overlap_low = max(quantile_low, bucket_left)
        overlap_high = min(quantile_high, bucket_right)
        overlap_length = overlap_high - overlap_low
        quantile_contribution_ratio = 0.0
        if quantile_low == quantile_high:
            if (bucket_left <= quantile_low < bucket_right) or (
                quantile_low == bucket_right == max_val
            ):
                quantile_contribution_ratio = 1.0
        elif overlap_length > 0:
            quantile_contribution_ratio = overlap_length / (quantile_high - quantile_low)
        bucket_freq += quantile_freq * quantile_contribution_ratio

    return Histogram.Bucket(
        low_value=bucket_left, high_value=bucket_right, sample_count=bucket_freq
    )

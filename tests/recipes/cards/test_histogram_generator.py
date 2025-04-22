"""
Unit tests for histogram_generator.py
"""

import pytest
from google.protobuf import text_format

from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator


def assert_histogram(
    expected: facet_feature_statistics_pb2.Histogram,
    actual: facet_feature_statistics_pb2.Histogram,
):
    """
    Helper function to assert the actual histogram is almost equal to the expected histogram.
    """
    assert expected.type == actual.type
    assert len(expected.buckets) == len(actual.buckets)
    for act_bucket, exp_bucket in zip(actual.buckets, expected.buckets):
        assert act_bucket.low_value == pytest.approx(exp_bucket.low_value, rel=1e-2)
        assert act_bucket.high_value == pytest.approx(exp_bucket.high_value, rel=1e-2)
        assert act_bucket.sample_count == pytest.approx(exp_bucket.sample_count, rel=1e-2)


def test_invalid_quantiles():
    """
    Tests generate_equal_height_histogram returns None when quantiles is invalid.
    """
    # Empty quantiles is invalid.
    assert histogram_generator.generate_equal_height_histogram(quantiles=[], num_buckets=5) is None
    # Quantiles which has less than 3 elements is invalid.
    assert (
        histogram_generator.generate_equal_height_histogram(quantiles=[1, 2], num_buckets=1) is None
    )
    # When (len(quantiles) - 1) % num_buckets != 0, quantiles is considered as invalid.
    assert (
        histogram_generator.generate_equal_height_histogram(
            quantiles=[1, 2, 3, 4, 5], num_buckets=3
        )
        is None
    )


def test_valid_quantiles():
    """
    Tests generate_equal_height_histogram returns correct histogram.
    """
    quantiles = list(range(11))
    actual_histogram = histogram_generator.generate_equal_height_histogram(
        quantiles=quantiles, num_buckets=5
    )
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 0.0
            high_value: 2.0
        }
        buckets {
            low_value: 2.0
            high_value: 4.0
        }
        buckets {
            low_value: 4.0
            high_value: 6.0
        }
        buckets {
            low_value: 6.0
            high_value: 8.0
        }
        buckets {
            low_value: 8.0
            high_value: 10.0
        }
        type: QUANTILES
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_uniform():
    """
    Tests the scenario where the single quantile bucket contains all histogram buckets.

    Setup:
      - the value range is [1.0, 2.0] and we request 5 buckets, so the bucket boundaries are
      [1.0, 1.2), [1.2, 1.4), [1.4, 1.6), [1.6, 1.8), [1.8, 2.0].
      - A single quantile bucket contains all histogram buckets. Hence, each histogram bucket
      gets assigned 1/5 of the total quantile bucket.
    """
    quantiles = [1.0, 2.0]
    total_freq = 100
    num_buckets = 5
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 1.2
            sample_count: 20.0
        }
        buckets {
            low_value: 1.2
            high_value: 1.4
            sample_count: 20.0
        }
        buckets {
            low_value: 1.4
            high_value: 1.6
            sample_count: 20.0
        }
        buckets {
            low_value: 1.6
            high_value: 1.8
            sample_count: 20.0
        }
        buckets {
            low_value: 1.8
            high_value: 2.0
            sample_count: 20.0
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_slightly_non_uniform():
    """
    Tests multiple quantile buckets with non-uniform distribution.

    Setup:
      - Histogram buckets: [1.0, 1.2), [1.2, 1.4), [1.4, 1.6), [1.6, 1.8), [1.8, 2.0).
      - There are two quantile buckets. The first contains the first two histogram buckets and
      the second contains the last three histogram buckets respectively.
    """
    quantiles = [1.0, 1.4, 2.0]
    total_freq = 100
    num_buckets = 5
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 1.2
            sample_count: 25.0
        }
        buckets {
            low_value: 1.2
            high_value: 1.4
            sample_count: 25.0
        }
        buckets {
            low_value: 1.4
            high_value: 1.6
            sample_count: 16.667
        }
        buckets {
            low_value: 1.6
            high_value: 1.8
            sample_count: 16.667
        }
        buckets {
            low_value: 1.8
            high_value: 2.0
            sample_count: 16.667
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_extremely_skewed_left():
    """
    Tests multiple quantile buckets with distribution heavily skewed to the left side.

    Setup:
      - Histogram buckets: [1.0, 1.2), [1.2, 1.4), [1.4, 1.6), [1.6, 1.8), [1.8, 2.0).
      - The distribution is "left-skewed" so that the first histogram bucket fully contains the
      first four quantile buckets and 1/9 of the last quantile bucket. The remaining 8/9 of the
      last quantile bucket overlap uniformly with the last four histogram buckets.
    """
    quantiles = [1.0, 1.01, 1.02, 1.03, 1.1, 2.0]
    total_freq = 100
    num_buckets = 5
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 1.2
            sample_count: 82.22
        }
        buckets {
            low_value: 1.2
            high_value: 1.4
            sample_count: 4.44
        }
        buckets {
            low_value: 1.4
            high_value: 1.6
            sample_count: 4.44
        }
        buckets {
            low_value: 1.6
            high_value: 1.8
            sample_count: 4.44
        }
        buckets {
            low_value: 1.8
            high_value: 2.0
            sample_count: 4.44
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_extremely_skewed_right():
    """
    Tests multiple quantile buckets with distribution heavily skewed to the right side.

    Setup:
      - Histogram buckets: [1.0, 1.2), [1.2, 1.4), [1.4, 1.6), [1.6, 1.8), [1.8, 2.0).
      - The distribution is "right-skewed" so that the last histogram bucket fully contains the
      last four quantile buckets and 1/9 of the first quantile bucket. The remaining 8/9 of the
      first quantile bucket overlap uniformly with the first four histogram buckets.
    """
    quantiles = [1.0, 1.9, 1.97, 1.98, 1.99, 2.0]
    total_freq = 100
    num_buckets = 5
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 1.2
            sample_count: 4.44
        }
        buckets {
            low_value: 1.2
            high_value: 1.4
            sample_count: 4.44
        }
        buckets {
            low_value: 1.4
            high_value: 1.6
            sample_count: 4.44
        }
        buckets {
            low_value: 1.6
            high_value: 1.8
            sample_count: 4.44
        }
        buckets {
            low_value: 1.8
            high_value: 2.0
            sample_count: 82.22
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_single_histogram_bucket():
    """
    Tests the single histogram bucket.
    """
    quantiles = [1.0, 1.9, 1.97, 1.98, 1.99, 2.0]
    total_freq = 100
    num_buckets = 1
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 100
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_repeated_quantiles():
    """
    Tests the repeated quantiles.

    Setup:
      - Histogram buckets boundaries are [1.0, 2.0) and [2.0, 3.0]
      - The first histogram bucket contains the first three quantile buckets, and 1/2 of the
      last quantile bucket: 25 * 3 + 25 / 2 = 87.5
      - The second histogram bucket contains 1/2 of the last quantile bucket: 25 / 2 = 12.5
    """
    quantiles = [1.0, 1.0, 1.0, 1.0, 3.0]
    total_freq = 100
    num_buckets = 2
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 87.5
        }
        buckets {
            low_value: 2.0
            high_value: 3.0
            sample_count: 12.5
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_edge_overlapping_ranges():
    """
    Tests the case where histogram buckets are the same as the quantile buckets.
    """
    quantiles = [1.0, 2.0, 3.0]
    total_freq = 200
    num_buckets = 2
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 100
        }
        buckets {
            low_value: 2.0
            high_value: 3.0
            sample_count: 100
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)


def test_same_value():
    """
    Tests the case where all quantiles are the same.
    """
    quantiles = [1.0, 1.0, 1.0, 1.0, 1.0]
    total_freq = 100
    num_buckets = 5
    expected_histogram = text_format.Parse(
        """
        buckets {
            low_value: -1.0
            high_value: 0
            sample_count: 0
        }
        buckets {
            low_value: 0
            high_value: 1.0
            sample_count: 0
        }
        buckets {
            low_value: 1.0
            high_value: 1.0
            sample_count: 100
        }
        buckets {
            low_value: 1.0
            high_value: 2.0
            sample_count: 0
        }
        buckets {
            low_value: 2.0
            high_value: 3.0
            sample_count: 0
        }
        type: STANDARD
        """,
        facet_feature_statistics_pb2.Histogram(),
    )
    actual_histogram = histogram_generator.generate_equal_width_histogram(
        quantiles=quantiles, num_buckets=num_buckets, total_freq=total_freq
    )
    assert_histogram(expected_histogram, actual_histogram)

"""
Test to convert data from tabular format to DatasetFeatureStatisticsList proto.
"""
import numpy as np
import pandas as pd
from google.protobuf import text_format

from mlflow.protos import facet_feature_statistics_pb2
from mlflow.pipelines.cards import pandas_renderer


def test_convert_to_html():
    proto = facet_feature_statistics_pb2.DatasetFeatureStatisticsList()
    html = pandas_renderer.construct_facets_html(proto)
    assert len(html) != 0


def testDTypeToType():
    fs_proto = facet_feature_statistics_pb2.FeatureNameStatistics
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.int32))
    # Boolean and time types treated as int
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(bool))
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.datetime64))
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.timedelta64))
    assert fs_proto.FLOAT == pandas_renderer.DtypeToType(np.dtype(np.float32))
    assert fs_proto.STRING == pandas_renderer.DtypeToType(np.dtype(str))
    # Unsupported types treated as string for now
    assert fs_proto.STRING == pandas_renderer.DtypeToType(np.dtype(np.void))


def testNdarrayToEntryTimeTypes():
    arr = np.array([np.datetime64("2005-02-25"), np.datetime64("2006-02-25")], dtype=np.datetime64)
    convertor = pandas_renderer.DtypeToNumberConverter(arr.dtype)
    assert np.array_equal([1109289600000000000, 1140825600000000000], convertor(arr))

    arr = np.array(
        [np.datetime64("2009-01-01") - np.datetime64("2008-01-01")], dtype=np.timedelta64
    )
    convertor = pandas_renderer.DtypeToNumberConverter(arr.dtype)
    assert np.array_equal([31622400000000000], convertor(arr))


def testCommonStats():
    data = {
        "Symbol": ["MSFT", "GOOG", "TSLA", "AAPL", "NFLX"],
        "Shares": [100, 50, 150, 200, None],
    }
    df = pd.DataFrame(data)

    common_stats = pandas_renderer.compute_common_stats(df["Symbol"])
    assert common_stats == text_format.Parse(
        """
        num_non_missing: 5
        num_missing: 0
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        """,
        facet_feature_statistics_pb2.CommonStatistics(),
    )

    common_stats = pandas_renderer.compute_common_stats(df["Shares"])
    assert common_stats == text_format.Parse(
        """
        num_missing: 1
        num_non_missing: 4
        min_num_values: 1
        max_num_values: 1
        avg_num_values: 1.0
        """,
        facet_feature_statistics_pb2.CommonStatistics(),
    )


def testConvertToProto():
    data = {
        "Symbol": ["MSFT", "AAPL", "MSFT", "AAPL", "NFLX"],
        "Shares": [100, 170, 150, 200, None],
    }
    df = pd.DataFrame(data)

    expected_proto = text_format.Parse(
        """
        datasets {
          num_examples: 5
          features {
            name: "Symbol"
            type: STRING
            string_stats {
              common_stats {
                num_non_missing: 5
                num_missing: 0
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
              }
              unique: 3
              top_values {
                value: "MSFT"
                frequency: 2.0
              }
              top_values {
                value: "AAPL"
                frequency: 2.0
              }
              avg_length: 4.0
              rank_histogram {
                buckets {
                  low_rank: 0
                  high_rank: 0
                  label: "MSFT"
                  sample_count: 2.0
                }
                buckets {
                  low_rank: 1
                  high_rank: 1
                  label: "AAPL"
                  sample_count: 2.0
                }
                buckets {
                  low_rank: 2
                  high_rank: 2
                  label: "NFLX"
                  sample_count: 1.0
                }
              }
            }
          }
          features {
            name: "Shares"
            type: FLOAT
            num_stats {
              common_stats {
                num_non_missing: 4
                num_missing: 1
                min_num_values: 1
                max_num_values: 1
                avg_num_values: 1.0
              }
              mean: 155.0
              std_dev: 42.03173404306164
              num_zeros: 0
              min: 100.0
              median: 160.0
              max: 200.0
              histograms {
                buckets {
                  low_value: 100.0
                  high_value: 110.0
                  sample_count: 0.26666666666666666
                }
                buckets {
                  low_value: 110.0
                  high_value: 120.0
                  sample_count: 0.26666666666666666
                }
                buckets {
                  low_value: 120.0
                  high_value: 130.0
                  sample_count: 0.26666666666666666
                }
                buckets {
                  low_value: 130.0
                  high_value: 140.0
                  sample_count: 0.26666666666666666
                }
                buckets {
                  low_value: 140.0
                  high_value: 150.0
                  sample_count: 0.35555555555555557
                }
                buckets {
                  low_value: 150.0
                  high_value: 160.0
                  sample_count: 0.5777777777777778
                }
                buckets {
                  low_value: 160.0
                  high_value: 170.0
                  sample_count: 0.6285714285714286
                }
                buckets {
                  low_value: 170.0
                  high_value: 180.0
                  sample_count: 0.48253968253968255
                }
                buckets {
                  low_value: 180.0
                  high_value: 190.0
                  sample_count: 0.4444444444444445
                }
                buckets {
                  low_value: 190.0
                  high_value: 200.0
                  sample_count: 0.4444444444444445
                }
                type: STANDARD
              }
              histograms {
                buckets {
                  low_value: 100.0
                  high_value: 115.0
                }
                buckets {
                  low_value: 115.0
                  high_value: 130.0
                }
                buckets {
                  low_value: 130.0
                  high_value: 145.0
                }
                buckets {
                  low_value: 145.0
                  high_value: 154.0
                }
                buckets {
                  low_value: 154.0
                  high_value: 160.0
                }
                buckets {
                  low_value: 160.0
                  high_value: 166.0
                }
                buckets {
                  low_value: 166.0
                  high_value: 173.0
                }
                buckets {
                  low_value: 173.0
                  high_value: 182.0
                }
                buckets {
                  low_value: 182.0
                  high_value: 191.0
                }
                buckets {
                  low_value: 191.0
                  high_value: 200.0
                }
                type: QUANTILES
              }
            }
          }
        }
        """,
        facet_feature_statistics_pb2.DatasetFeatureStatisticsList(),
    )

    converted_proto = pandas_renderer.convert_to_proto(df)
    assert converted_proto == expected_proto

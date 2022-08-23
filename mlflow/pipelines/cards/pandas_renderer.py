"""
Renders the statistics of logged data in a HTML format.
"""
import base64
import numpy as np
import pandas as pd

from typing import Union, Iterable, Tuple
from facets_overview import feature_statistics_pb2
from mlflow.pipelines.cards import histogram_generator


def DtypeToType(dtype):
    """Converts a Numpy dtype to the FeatureNameStatistics.Type proto enum."""
    fs_proto = feature_statistics_pb2.FeatureNameStatistics
    if dtype.char in np.typecodes["AllFloat"]:
        return fs_proto.FLOAT
    elif (
        dtype.char in np.typecodes["AllInteger"]
        or dtype == np.bool
        or np.issubdtype(dtype, np.datetime64)
        or np.issubdtype(dtype, np.timedelta64)
    ):
        return fs_proto.INT
    else:
        return fs_proto.STRING


def convert_to_dataset_feature_statistics(
    df: pd.DataFrame,
) -> feature_statistics_pb2.DatasetFeatureStatistics:
    """
    Converts the data statistics from DataFrame format to DatasetFeatureStatistics proto.

    :param df: The DataFrame that contains the statistics. Each column of the DataFrame is a statistic.
               We only support the statistics defined in `aggregate_metric.py` now.
    :return: A DatasetFeatureStatistics proto.
    """
    fs_proto = feature_statistics_pb2.FeatureNameStatistics
    feature_stats = feature_statistics_pb2.DatasetFeatureStatistics()
    pandas_describe = df.describe()
    num_examples = len(df)
    quantiles_to_get = [x * 10 / 100 for x in range(10 + 1)]
    quantiles = df.quantile(quantiles_to_get, numeric_only=False)

    for key in df:
        feat = feature_stats.features.add(type=DtypeToType(df[key].dtype), name=key.encode("utf-8"))
        if feat.type in (fs_proto.INT, fs_proto.FLOAT):
            featstats = feat.num_stats
            commonstats = featstats.common_stats

            featstats.std_dev = pandas_describe[key]["std"]
            featstats.mean = pandas_describe[key]["mean"]
            featstats.min = pandas_describe[key]["min"]
            featstats.max = pandas_describe[key]["max"]
            featstats.median = df[key].median()
            featstats.num_zeros = df[key].value_counts()[0]
            commonstats.num_missing = df[key].isnull().sum()
            commonstats.num_non_missing = num_examples - commonstats.num_missing

            equal_width_hist = histogram_generator.generate_equal_width_histogram(
                quantiles=quantiles[key].to_numpy(),
                num_buckets=10,
                total_freq=commonstats.num_non_missing,
            )
            if equal_width_hist:
                featstats.histograms.append(equal_width_hist)
            equal_height_hist = histogram_generator.generate_equal_height_histogram(
                quantiles=quantiles[key].to_numpy(), num_buckets=10
            )
            if equal_height_hist:
                featstats.histograms.append(equal_height_hist)

    return feature_stats


def convert_to_proto(df: pd.DataFrame) -> feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts the data statistics from DataFrame format to DatasetFeatureStatisticsList proto.

    :param df: The DataFrame that contains the statistics. Each column of the DataFrame is a statistic.
               We only support the statistics defined in `aggregate_metric.py` now.
    :return: A DatasetFeatureStatisticsList proto.
    """
    feature_stats = convert_to_dataset_feature_statistics(df)
    feature_stats_list = feature_statistics_pb2.DatasetFeatureStatisticsList()
    feature_stats_list.datasets.append(feature_stats)
    return feature_stats_list


def convert_to_comparison_proto(
    dfs: Iterable[Tuple[str, pd.DataFrame]]
) -> feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts a collection of named stats DataFrames to a single DatasetFeatureStatisticsList proto.
    :param dfs: The named "glimpses" that contain the statistics. Each "glimpse" DataFrame has the same properties
        as the input to `convert_to_proto()`.
    :return: A DatasetFeatureStatisticsList proto which contains a translation of the glimpses with the given names.
    """
    feature_stats_list = feature_statistics_pb2.DatasetFeatureStatisticsList()
    for (name, df) in dfs:
        proto = convert_to_dataset_feature_statistics(df)
        proto.name = name
        feature_stats_list.datasets.append(proto)
    return feature_stats_list


def construct_facets_html(
    proto: feature_statistics_pb2.DatasetFeatureStatisticsList, compare: bool = False
):
    """
    Constructs the facets HTML to visualize the serialized FeatureStatisticsList proto.
    :param protostr: A string serialization of a FeatureStatisticsList proto
    :param compare: If True, then the returned visualization switches on the comparison mode for several stats.
    :return: the HTML for Facets visualization
    """
    # facets_html_bundle = _get_facets_html_bundle()
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    html_template = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="facets" proto-input="{protostr}" compare-mode="{compare}"></facets-overview>
    """
    html = html_template.format(protostr=protostr, compare=compare)
    return html


def render_html(inputs: pd.DataFrame) -> None:
    """Rendering the data statistics in a HTML format.

    :param inputs: Either a single "glimpse" DataFrame that contains the statistics, or a collection of
        (name, DataFrame) pairs where each pair names a separate "glimpse" and they are all visualized in comparison
        mode.
    :return: None
    """
    from IPython.display import display as ip_display, HTML

    if isinstance(inputs, pd.DataFrame):
        df: pd.DataFrame = inputs
        proto = convert_to_proto(df)
        compare = False
    else:
        proto = convert_to_comparison_proto(inputs)
        compare = True

    html = construct_facets_html(proto, compare=compare)
    ip_display(HTML(data=html))

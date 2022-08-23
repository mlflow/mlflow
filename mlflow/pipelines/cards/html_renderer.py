"""
Renders the statistics of logged data in a HTML format.
"""
import base64
import re
import functools

import gzip
import importlib.resources as pkg_resources
from typing import Union, Iterable, Tuple

import mlflow
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F

from mlflow.exceptions import MlflowException
from facets_overview import feature_statistics_pb2
from mlflow.pipelines.cards import histogram_generator


def _has_column(row: Row, column: str) -> bool:
    """Checks if `column` exists in a row and that its value is non-null."""
    return column in row and row[column] is not None


def _convert_to_common_statistics(row: Row):
    """Converts the row to feature_statistics_pb2.CommonStatistics."""
    common_stats = feature_statistics_pb2.CommonStatistics()
    if _has_column(row, "num_nulls"):
        common_stats.num_missing = row["num_nulls"]
    if _has_column(row, "count"):
        common_stats.num_non_missing = row["count"]

    # TODO(ML-19762): Once we computes the size of array, populate meaningful value here for array, similar to
    # https://github.com/databricks/universe/blob/HEAD/spark/driver/dbutils_impl/data/rendering/SummaryProtoRenderer.scala#L19-L34
    common_stats.min_num_values = 1
    common_stats.max_num_values = 1
    common_stats.avg_num_values = 1
    return common_stats


# A map from tabular field to numeric proto field.
_NUMERIC_STATS_FIELDS_MAP = {
    "min": "min",
    "max": "max",
    "avg": "mean",
    "median": "median",
    "stddev": "std_dev",
    "num_zeros": "num_zeros",
}

# Number of buckets in the equal width histograms
_NUM_EQUAL_WIDTH_HISTOGRAM_BUCKETS = 10

# Number of buckets in the equal height histograms
_NUM_EQUAL_HEIGHT_HISTOGRAM_BUCKETS = 10


def get_num_equal_width_histogram_buckets():
    """Returns the number of buckets used in equal width histograms"""
    return _NUM_EQUAL_WIDTH_HISTOGRAM_BUCKETS


def get_num_equal_height_histogram_buckets():
    """Returns the number of buckets used in equal height histograms"""
    return _NUM_EQUAL_HEIGHT_HISTOGRAM_BUCKETS


def _convert_numeric_statistics(
    row: Row, feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the numeric row to feature_statistics_pb2.FeatureNameStatistics."""
    # TODO(ML-19756): Also add the histograms to the proto.
    common_stats = _convert_to_common_statistics(row)
    num_stats = feature_statistics_pb2.NumericStatistics()
    num_stats.common_stats.CopyFrom(common_stats)
    for source_field, target_field in _NUMERIC_STATS_FIELDS_MAP.items():
        if _has_column(row, source_field):
            setattr(num_stats, target_field, row[source_field])

    if row["data_type"] in {"tinyint", "smallint", "int", "bigint"}:
        feature_name_stats.type = feature_statistics_pb2.FeatureNameStatistics.INT
    else:
        feature_name_stats.type = feature_statistics_pb2.FeatureNameStatistics.FLOAT
    if _has_column(row, "quantiles"):
        equal_width_hist = histogram_generator.generate_equal_width_histogram(
            quantiles=row["quantiles"],
            num_buckets=get_num_equal_width_histogram_buckets(),
            total_freq=common_stats.num_non_missing,
        )
        if equal_width_hist:
            num_stats.histograms.append(equal_width_hist)
        equal_height_hist = histogram_generator.generate_equal_height_histogram(
            quantiles=row["quantiles"], num_buckets=get_num_equal_height_histogram_buckets()
        )
        if equal_height_hist:
            num_stats.histograms.append(equal_height_hist)

    feature_name_stats.num_stats.CopyFrom(num_stats)


def _convert_boolean_statistics(
    row: Row, feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the boolean row to feature_statistics_pb2.FeatureNameStatistics."""
    return _convert_string_statistics(row, feature_name_stats)


def _convert_timestamp_statistics(
    row: Row, feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the timestamp/date row to feature_statistics_pb2.FeatureNameStatistics."""
    return _convert_numeric_statistics(row, feature_name_stats)


def _convert_string_statistics(
    row: Row, feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the string row to feature_statistics_pb2.FeatureNameStatistics."""
    common_stats = _convert_to_common_statistics(row)
    string_stats = feature_statistics_pb2.StringStatistics()
    string_stats.common_stats.CopyFrom(common_stats)
    if _has_column(row, "distinct_count"):
        string_stats.unique = row["distinct_count"]
    # row["frequent_items"] can be an empty list if majority values are distinct.
    if _has_column(row, "frequent_items"):
        # Fill the first non-None value to the top_values.
        for frequent_item in row["frequent_items"]:
            if frequent_item["item"] is not None:
                string_stats.top_values.append(
                    feature_statistics_pb2.StringStatistics.FreqAndValue(
                        value=frequent_item["item"], frequency=frequent_item["count"]
                    )
                )
                break
        string_stats.rank_histogram.CopyFrom(_gen_rank_histogram(row))

    if _has_column(row, "avg_length"):
        string_stats.avg_length = row["avg_length"]

    feature_name_stats.type = feature_statistics_pb2.FeatureNameStatistics.STRING
    feature_name_stats.string_stats.CopyFrom(string_stats)


# A map from tabular field to bytes proto field.
_BYTES_STATS_FIELDS_MAP = {
    "distinct_count": "unique",
    "min_length": "min_num_bytes",
    "max_length": "max_num_bytes",
    "avg_length": "avg_num_bytes",
}


def _convert_bytes_statistics(
    row: Row, feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the binary row to feature_statistics_pb2.FeatureNameStatistics."""
    common_stats = _convert_to_common_statistics(row)
    bytes_stats = feature_statistics_pb2.BytesStatistics()
    bytes_stats.common_stats.CopyFrom(common_stats)
    for source_field, target_field in _BYTES_STATS_FIELDS_MAP.items():
        if _has_column(row, source_field):
            setattr(bytes_stats, target_field, row[source_field])

    feature_name_stats.type = feature_statistics_pb2.FeatureNameStatistics.BYTES
    feature_name_stats.bytes_stats.CopyFrom(bytes_stats)


def _convert_struct_statistics(
    unused_row: Row, unused_feature_name_stats: feature_statistics_pb2.FeatureNameStatistics
):
    """Converts the array/map/struct row to feature_statistics_pb2.FeatureNameStatistics."""
    # TODO(ML-19763): Use the latest proto file for the model monitoring package.
    pass


def _gen_rank_histogram(row: Row):
    """Converts the the output of F.expr(approx_top_k(...)) to feature_statistics_pb2.RankHistogram."""
    buckets = [
        feature_statistics_pb2.RankHistogram.Bucket(
            low_rank=id + 1,
            high_rank=id + 1,
            label=frequent_item["item"],
            sample_count=frequent_item["count"],
        )
        for id, frequent_item in enumerate(row["frequent_items"])
        if frequent_item["item"] is not None
    ]
    histogram = feature_statistics_pb2.RankHistogram()
    histogram.buckets.extend(buckets)
    return histogram


_CONVERTER_BY_DATA_TYPES = {
    # Numeric
    "tinyint": _convert_numeric_statistics,
    "smallint": _convert_numeric_statistics,
    "int": _convert_numeric_statistics,
    "bigint": _convert_numeric_statistics,
    "float": _convert_numeric_statistics,
    "double": _convert_numeric_statistics,
    "decimal": _convert_numeric_statistics,
    # Boolean
    "boolean": _convert_boolean_statistics,
    # Time
    "timestamp": _convert_timestamp_statistics,
    "date": _convert_timestamp_statistics,
    # String/Binary
    "string": _convert_string_statistics,
    "binary": _convert_bytes_statistics,
    # Complex types
    "array": _convert_struct_statistics,
    "map": _convert_struct_statistics,
    "struct": _convert_struct_statistics,
}


def _compute_data_profiles(df: DataFrame) -> DataFrame:
    """
    This internal function computes the data profiles given the DataFrame.

    Uses the provided timestamp keys and profile spec to compute a profile
    DataFrame for every granularity, and unions the resulting DataFrames
    into a single output.

    :param df: The data to compute the profiles.
    :return: The computed data profiles in a DataFrame with columns:
                - granularity
                - window
                - column_name
                - <all computed statistics>
    """
    features_to_compute = df.columns

    # TODO(ML-21787): Refactor so that we can reuse the metrics definition here.
    # Generates a struct that contains an expression for every aggregate metric we want to compute.
    aggregates_expressions = [
        F.struct(
            [
                F.count(column_name).alias("count"),
                F.sum(F.col(column_name).isNull().cast(T.LongType())).alias("num_nulls"),
            ]
        ).alias(column_name)
        for column_name in features_to_compute
    ]
    # For each feature, we map the column name to itself and explode it so we can compute
    # the same set of aggregate expressions on every input feature
    map_args_for_explode = []
    for column_name in features_to_compute:
        map_args_for_explode.extend([F.lit(column_name), F.col(column_name)])
    transposed_columns = F.explode(F.create_map(*map_args_for_explode))

    # We will compute an aggregate DataFrame for every granularity from the spec and union them at the end
    aggregate_dfs = []
    aggregate_dfs.append(
        df.agg(*aggregates_expressions)
        .select([transposed_columns])
        .select(
            [
                F.col("key").alias("column_name"),
                F.col("value.count"),
                F.col("value.num_nulls"),
            ]
        )
    )

    return functools.reduce(DataFrame.unionByName, aggregate_dfs)


def DtypeToType(self, dtype):
    """Converts a Numpy dtype to the FeatureNameStatistics.Type proto enum."""
    if dtype.char in np.typecodes["AllFloat"]:
        return self.fs_proto.FLOAT
    elif (
        dtype.char in np.typecodes["AllInteger"]
        or dtype == np.bool
        or np.issubdtype(dtype, np.datetime64)
        or np.issubdtype(dtype, np.timedelta64)
    ):
        return self.fs_proto.INT
    else:
        return self.fs_proto.STRING


def convert_to_dataset_feature_statistics(
    df: DataFrame,
) -> feature_statistics_pb2.DatasetFeatureStatistics:
    """
    Converts the data statistics from DataFrame format to DatasetFeatureStatistics proto.

    :param df: The DataFrame that contains the statistics. Each column of the DataFrame is a statistic.
               We only support the statistics defined in `aggregate_metric.py` now.
    :return: A DatasetFeatureStatistics proto.
    """
    feature_stats = feature_statistics_pb2.DatasetFeatureStatistics()
    # computed_df = _compute_data_profiles(df)
    rows = df.itertuples()
    print(feature_stats)
    # feature_stats.features.append(name=df["name"], num_examples=df["size"])

    for row in rows:
        # To remove the element type or type parameter (e.g. `array<bigint>` -> `array`, `decimal(10,0)` -> `decimal`).
        data_type = re.sub(r"(<|\().*(>|\))", "", row[type])
        if data_type in _CONVERTER_BY_DATA_TYPES:
            feature_name_stats = feature_statistics_pb2.FeatureNameStatistics(name=row["name"])
            convert_func = _CONVERTER_BY_DATA_TYPES[data_type]
            convert_func(row, feature_name_stats)
            feature_stats.features.append(feature_name_stats)
        else:
            raise MlflowException(
                f"Failed to convert the data for data_type {data_type}",
                error_code=INVALID_PARAMETER_VALUE,
            )
    return feature_stats


def convert_to_proto(df: DataFrame) -> feature_statistics_pb2.DatasetFeatureStatisticsList:
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
    dfs: Iterable[Tuple[str, DataFrame]]
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


def _get_facets_html_bundle() -> str:
    """
    Lazily loads (and caches) the HTML bundle for the facets visualization.
    :return: The HTML source for the Facets bundle
    """
    if not hasattr(_get_facets_html_bundle, "_bundle"):
        with pkg_resources.open_binary(
            __package__, "facets_bundle.html.gz"
        ) as compressed_facets_html_bundle, gzip.GzipFile(
            fileobj=compressed_facets_html_bundle, mode="rb"
        ) as f:
            setattr(_get_facets_html_bundle, "_bundle", f.read().decode())
    return getattr(_get_facets_html_bundle, "_bundle")


def construct_facets_html(
    proto: feature_statistics_pb2.DatasetFeatureStatisticsList, compare: bool = False
):
    """
    Constructs the facets HTML to visualize the serialized FeatureStatisticsList proto.
    :param protostr: A string serialization of a FeatureStatisticsList proto
    :param compare: If True, then the returned visualization switches on the comparison mode for several stats.
    :return: the HTML for Facets visualization
    """
    facets_html_bundle = _get_facets_html_bundle()
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    html_template = """
        {facets_html_bundle}
        <facets-overview id="facets" proto-input="{protostr}" compare-mode="{compare}"></facets-overview>
    """
    html = html_template.format(
        facets_html_bundle=facets_html_bundle, protostr=protostr, compare=compare
    )
    return html


def render_html(inputs: Union[DataFrame, Iterable[Tuple[str, DataFrame]]]) -> None:
    """Rendering the data statistics in a HTML format.

    :param inputs: Either a single "glimpse" DataFrame that contains the statistics, or a collection of
        (name, DataFrame) pairs where each pair names a separate "glimpse" and they are all visualized in comparison
        mode.
    :return: None
    """
    if isinstance(inputs, DataFrame):
        df: DataFrame = inputs
        proto = convert_to_proto(df)
        compare = False
    else:
        proto = convert_to_comparison_proto(inputs)
        compare = True

    html = construct_facets_html(proto, compare=compare)
    mlflow.utils.databricks_utils._get_dbutils().notebook.displayHTML(
        html
    )  # pylint: disable=protected-access

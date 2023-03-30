"""
Renders the statistics of logged data in a HTML format.
"""
import base64
import numpy as np
import pandas as pd
import sys
from packaging.version import Version

from typing import Union, Iterable, Tuple
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
from mlflow.exceptions import MlflowException

# Number of categorical strings values to be rendered as part of the histogram
HISTOGRAM_CATEGORICAL_LEVELS_COUNT = 100


def get_facet_type_from_numpy_type(dtype):
    """Converts a Numpy dtype to the FeatureNameStatistics.Type proto enum."""
    fs_proto = facet_feature_statistics_pb2.FeatureNameStatistics
    if dtype.char in np.typecodes["Complex"]:
        raise MlflowException(
            "Found type complex, but expected one of: int, long, float, string, bool"
        )
    elif dtype.char in np.typecodes["AllFloat"]:
        return fs_proto.FLOAT
    elif (
        dtype.char in np.typecodes["AllInteger"]
        or np.issubdtype(dtype, np.datetime64)
        or np.issubdtype(dtype, np.timedelta64)
    ):
        return fs_proto.INT
    else:
        return fs_proto.STRING


def datetime_and_timedelta_converter(dtype):
    """
    Converts a Numpy dtype to a converter method if applicable.
    The converter method takes in a numpy array of objects of the provided
    dtype and returns a numpy array of the numbers backing that object for
    statistical analysis. Returns None if no converter is necessary.
    :param dtype: The numpy dtype to make a converter for.
    :return: The converter method or None.
    """
    if np.issubdtype(dtype, np.datetime64):

        def datetime_converter(dt_list):
            return np.array([pd.Timestamp(dt).value for dt in dt_list])

        return datetime_converter
    elif np.issubdtype(dtype, np.timedelta64):

        def timedelta_converter(td_list):
            return np.array([pd.Timedelta(td).value for td in td_list])

        return timedelta_converter
    else:
        return None


def compute_common_stats(column) -> facet_feature_statistics_pb2.CommonStatistics:
    """
    Computes common statistics for a given column in the DataFrame.

    :param column: A column from a DataFrame.
    :return: A CommonStatistics proto.
    """
    common_stats = facet_feature_statistics_pb2.CommonStatistics()
    common_stats.num_missing = column.isnull().sum()
    common_stats.num_non_missing = len(column) - common_stats.num_missing
    # TODO: Add support to multi dimensional columns similar to
    # https://github.com/PAIR-code/facets/blob/4742b8b93c2dacf22fc8ace2cee42dd06382c48e/facets_overview/facets_overview/base_generic_feature_statistics_generator.py#L106-L117
    common_stats.min_num_values = 1
    common_stats.max_num_values = 1
    common_stats.avg_num_values = 1.0

    return common_stats


def convert_to_dataset_feature_statistics(
    df: pd.DataFrame,
) -> facet_feature_statistics_pb2.DatasetFeatureStatistics:
    """
    Converts the data statistics from DataFrame format to DatasetFeatureStatistics proto.

    :param df: The DataFrame for which feature statistics need to be computed.
    :return: A DatasetFeatureStatistics proto.
    """
    fs_proto = facet_feature_statistics_pb2.FeatureNameStatistics
    feature_stats = facet_feature_statistics_pb2.DatasetFeatureStatistics()
    data_type_custom_stat = facet_feature_statistics_pb2.CustomStatistic()
    kwargs = {} if Version(pd.__version__) >= Version("2.0.0rc0") else {"datetime_is_numeric": True}
    pandas_describe = df.describe(include="all", **kwargs)
    feature_stats.num_examples = len(df)
    quantiles_to_get = [x * 10 / 100 for x in range(10 + 1)]
    try:
        quantiles = df.select_dtypes(include="number").quantile(quantiles_to_get)
    except:
        raise MlflowException("Error in generating quantiles")

    for key in df:
        pandas_describe_key = pandas_describe[key]
        current_column_value = df[key]
        data_type = current_column_value.dtype
        data_type_custom_stat.name = "data type"
        data_type_custom_stat.str = str(data_type)
        feat = feature_stats.features.add(
            type=get_facet_type_from_numpy_type(data_type),
            name=key.encode("utf-8"),
            custom_stats=[data_type_custom_stat],
        )
        if feat.type in (fs_proto.INT, fs_proto.FLOAT):
            feat_stats = feat.num_stats

            converter = datetime_and_timedelta_converter(current_column_value.dtype)
            if converter:
                date_time_converted = converter(current_column_value)
                current_column_value = pd.DataFrame(date_time_converted)[0]
                kwargs = (
                    {}
                    if Version(pd.__version__) >= Version("2.0.0rc0")
                    else {"datetime_is_numeric": True}
                )
                pandas_describe_key = current_column_value.describe(include="all", **kwargs)
                quantiles[key] = current_column_value.quantile(quantiles_to_get)

            default_value = 0
            feat_stats.std_dev = pandas_describe_key.get("std", default_value)
            feat_stats.mean = pandas_describe_key.get("mean", default_value)
            feat_stats.min = pandas_describe_key.get("min", default_value)
            feat_stats.max = pandas_describe_key.get("max", default_value)
            feat_stats.median = current_column_value.median()
            feat_stats.num_zeros = (current_column_value == 0).sum()
            feat_stats.common_stats.CopyFrom(compute_common_stats(current_column_value))

            if key in quantiles:
                equal_width_hist = histogram_generator.generate_equal_width_histogram(
                    quantiles=quantiles[key].to_numpy(),
                    num_buckets=10,
                    total_freq=feat_stats.common_stats.num_non_missing,
                )
                if equal_width_hist:
                    feat_stats.histograms.append(equal_width_hist)
                equal_height_hist = histogram_generator.generate_equal_height_histogram(
                    quantiles=quantiles[key].to_numpy(), num_buckets=10
                )
                if equal_height_hist:
                    feat_stats.histograms.append(equal_height_hist)
        elif feat.type == fs_proto.STRING:
            is_current_column_boolean_type = False
            if current_column_value.dtype == bool:
                current_column_value = current_column_value.replace({True: "True", False: "False"})
                is_current_column_boolean_type = True
            feat_stats = feat.string_stats
            strs = current_column_value.dropna()

            feat_stats.avg_length = (
                np.mean(np.vectorize(len)(strs)) if not is_current_column_boolean_type else 0
            )
            vals, counts = np.unique(strs, return_counts=True)
            feat_stats.unique = pandas_describe_key.get("unique", len(vals))
            sorted_vals = sorted(zip(counts, vals), reverse=True)
            sorted_vals = sorted_vals[:HISTOGRAM_CATEGORICAL_LEVELS_COUNT]
            for val_index, val in enumerate(sorted_vals):
                try:
                    if sys.version_info.major < 3 or isinstance(val[1], (bytes, bytearray)):
                        printable_val = val[1].decode("UTF-8", "strict")
                    else:
                        printable_val = val[1]
                except (UnicodeDecodeError, UnicodeEncodeError):
                    printable_val = "__BYTES_VALUE__"
                bucket = feat_stats.rank_histogram.buckets.add(
                    low_rank=val_index,
                    high_rank=val_index,
                    sample_count=val[0].item(),
                    label=printable_val,
                )
                if val_index < 2:
                    feat_stats.top_values.add(value=bucket.label, frequency=bucket.sample_count)

            feat_stats.common_stats.CopyFrom(compute_common_stats(current_column_value))

    return feature_stats


def convert_to_proto(df: pd.DataFrame) -> facet_feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts the data from DataFrame format to DatasetFeatureStatisticsList proto.

    :param df: The DataFrame for which feature statistics need to be computed.
    :return: A DatasetFeatureStatisticsList proto.
    """
    feature_stats = convert_to_dataset_feature_statistics(df)
    feature_stats_list = facet_feature_statistics_pb2.DatasetFeatureStatisticsList()
    feature_stats_list.datasets.append(feature_stats)
    return feature_stats_list


def convert_to_comparison_proto(
    dfs: Iterable[Tuple[str, pd.DataFrame]]
) -> facet_feature_statistics_pb2.DatasetFeatureStatisticsList:
    """
    Converts a collection of named stats DataFrames to a single DatasetFeatureStatisticsList proto.
    :param dfs: The named "glimpses" that contain the DataFrame. Each "glimpse"
        DataFrame has the same properties as the input to `convert_to_proto()`.
    :return: A DatasetFeatureStatisticsList proto which contains a translation
        of the glimpses with the given names.
    """
    feature_stats_list = facet_feature_statistics_pb2.DatasetFeatureStatisticsList()
    for name, df in dfs:
        if not df.empty:
            proto = convert_to_dataset_feature_statistics(df)
            proto.name = name
            feature_stats_list.datasets.append(proto)
    return feature_stats_list


def get_facets_polyfills() -> str:
    """
    A JS polyfill/monkey-patching function that fixes issue where objectURL passed as a
    "base" argument to the URL constructor ends up in a "invalid URL" exception.

    Polymer is using parent's URL in its internal asset URL resolution system, while MLFLow
    artifact rendering engine uses object URLs to display iframed artifacts code. This ends up
    in object URL being used in `new URL()` constructor which needs to be patched.

    Original function code:

    (function patchURLConstructor() {
        const _originalURLConstructor = window.URL;
        window.URL = function (url, base) {
            if (typeof base === "string" && base.startsWith("blob:")) {
                return new URL(base);
            }
            return new _originalURLConstructor(url, base);
        };
    })();
    """
    return '!function(){let t=window.URL;window.URL=function(n,e){return"string"==typeof e&&e.startsWith("blob:")?new URL(e):new t(n,e)}}();'  # pylint: disable=line-too-long


def construct_facets_html(
    proto: facet_feature_statistics_pb2.DatasetFeatureStatisticsList, compare: bool = False
) -> str:
    """
    Constructs the facets HTML to visualize the serialized FeatureStatisticsList proto.
    :param proto: A DatasetFeatureStatisticsList proto which contains the statistics for a DataFrame
    :param compare: If True, then the returned visualization switches on the comparison
        mode for several stats.
    :return: the HTML for Facets visualization
    """
    # facets_html_bundle = _get_facets_html_bundle()
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")
    polyfills_code = get_facets_polyfills()

    html_template = """
        <div style="background-color: white">
        <script>{polyfills_code}</script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
        <facets-overview id="facets" proto-input="{protostr}" compare-mode="{compare}"></facets-overview>
        </div>
    """
    html = html_template.format(protostr=protostr, compare=compare, polyfills_code=polyfills_code)
    return html


def get_html(inputs: Union[pd.DataFrame, Iterable[Tuple[str, pd.DataFrame]]]) -> str:
    """Rendering the data statistics in a HTML format.

    :param inputs: Either a single "glimpse" DataFrame that contains the statistics, or a
        collection of (name, DataFrame) pairs where each pair names a separate "glimpse"
        and they are all visualized in comparison mode.
    :return: None
    """
    if isinstance(inputs, pd.DataFrame):
        if not inputs.empty:
            proto = convert_to_proto(inputs)
            compare = False
    else:
        proto = convert_to_comparison_proto(inputs)
        compare = True

    html = construct_facets_html(proto, compare=compare)
    return html

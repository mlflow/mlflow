"""
Test to convert data from tabular format to DatasetFeatureStatisticsList proto.
"""
import numpy as np

from facets_overview import feature_statistics_pb2
from mlflow.pipelines.cards import pandas_renderer

# _NUMERIC_TYPES = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal(10,0)")
# _TIME_TYPES = ("timestamp", "date")
# _STRUCT_TYPES = ("array<bigint>", "map<string,bigint>", "struct<col1:int,col2:int>")


def test_convert_to_html():
    proto = feature_statistics_pb2.DatasetFeatureStatisticsList()
    html = pandas_renderer.construct_facets_html(proto)
    assert len(html) != 0


def testDTypeToType():
    fs_proto = feature_statistics_pb2.FeatureNameStatistics
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.int32))
    # Boolean and time types treated as int
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(bool))
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.datetime64))
    assert fs_proto.INT == pandas_renderer.DtypeToType(np.dtype(np.timedelta64))
    assert fs_proto.FLOAT == pandas_renderer.DtypeToType(np.dtype(np.float32))
    assert fs_proto.STRING == pandas_renderer.DtypeToType(np.dtype(np.str))
    # Unsupported types treated as string for now
    assert fs_proto.STRING == pandas_renderer.DtypeToType(np.dtype(np.void))

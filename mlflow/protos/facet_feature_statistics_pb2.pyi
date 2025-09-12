from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DatasetFeatureStatisticsList(_message.Message):
    __slots__ = ("datasets",)
    DATASETS_FIELD_NUMBER: _ClassVar[int]
    datasets: _containers.RepeatedCompositeFieldContainer[DatasetFeatureStatistics]
    def __init__(self, datasets: _Optional[_Iterable[_Union[DatasetFeatureStatistics, _Mapping]]] = ...) -> None: ...

class Path(_message.Message):
    __slots__ = ("step",)
    STEP_FIELD_NUMBER: _ClassVar[int]
    step: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, step: _Optional[_Iterable[str]] = ...) -> None: ...

class DatasetFeatureStatistics(_message.Message):
    __slots__ = ("name", "num_examples", "weighted_num_examples", "features")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_NUM_EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    name: str
    num_examples: int
    weighted_num_examples: float
    features: _containers.RepeatedCompositeFieldContainer[FeatureNameStatistics]
    def __init__(self, name: _Optional[str] = ..., num_examples: _Optional[int] = ..., weighted_num_examples: _Optional[float] = ..., features: _Optional[_Iterable[_Union[FeatureNameStatistics, _Mapping]]] = ...) -> None: ...

class FeatureNameStatistics(_message.Message):
    __slots__ = ("name", "path", "type", "num_stats", "string_stats", "bytes_stats", "struct_stats", "custom_stats")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        INT: _ClassVar[FeatureNameStatistics.Type]
        FLOAT: _ClassVar[FeatureNameStatistics.Type]
        STRING: _ClassVar[FeatureNameStatistics.Type]
        BYTES: _ClassVar[FeatureNameStatistics.Type]
        STRUCT: _ClassVar[FeatureNameStatistics.Type]
    INT: FeatureNameStatistics.Type
    FLOAT: FeatureNameStatistics.Type
    STRING: FeatureNameStatistics.Type
    BYTES: FeatureNameStatistics.Type
    STRUCT: FeatureNameStatistics.Type
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NUM_STATS_FIELD_NUMBER: _ClassVar[int]
    STRING_STATS_FIELD_NUMBER: _ClassVar[int]
    BYTES_STATS_FIELD_NUMBER: _ClassVar[int]
    STRUCT_STATS_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_STATS_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: Path
    type: FeatureNameStatistics.Type
    num_stats: NumericStatistics
    string_stats: StringStatistics
    bytes_stats: BytesStatistics
    struct_stats: StructStatistics
    custom_stats: _containers.RepeatedCompositeFieldContainer[CustomStatistic]
    def __init__(self, name: _Optional[str] = ..., path: _Optional[_Union[Path, _Mapping]] = ..., type: _Optional[_Union[FeatureNameStatistics.Type, str]] = ..., num_stats: _Optional[_Union[NumericStatistics, _Mapping]] = ..., string_stats: _Optional[_Union[StringStatistics, _Mapping]] = ..., bytes_stats: _Optional[_Union[BytesStatistics, _Mapping]] = ..., struct_stats: _Optional[_Union[StructStatistics, _Mapping]] = ..., custom_stats: _Optional[_Iterable[_Union[CustomStatistic, _Mapping]]] = ...) -> None: ...

class WeightedCommonStatistics(_message.Message):
    __slots__ = ("num_non_missing", "num_missing", "avg_num_values", "tot_num_values")
    NUM_NON_MISSING_FIELD_NUMBER: _ClassVar[int]
    NUM_MISSING_FIELD_NUMBER: _ClassVar[int]
    AVG_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    TOT_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    num_non_missing: float
    num_missing: float
    avg_num_values: float
    tot_num_values: float
    def __init__(self, num_non_missing: _Optional[float] = ..., num_missing: _Optional[float] = ..., avg_num_values: _Optional[float] = ..., tot_num_values: _Optional[float] = ...) -> None: ...

class CustomStatistic(_message.Message):
    __slots__ = ("name", "num", "str", "histogram", "rank_histogram")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_FIELD_NUMBER: _ClassVar[int]
    STR_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    RANK_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    name: str
    num: float
    str: str
    histogram: Histogram
    rank_histogram: RankHistogram
    def __init__(self, name: _Optional[str] = ..., num: _Optional[float] = ..., str: _Optional[str] = ..., histogram: _Optional[_Union[Histogram, _Mapping]] = ..., rank_histogram: _Optional[_Union[RankHistogram, _Mapping]] = ...) -> None: ...

class NumericStatistics(_message.Message):
    __slots__ = ("common_stats", "mean", "std_dev", "num_zeros", "min", "median", "max", "histograms", "weighted_numeric_stats")
    COMMON_STATS_FIELD_NUMBER: _ClassVar[int]
    MEAN_FIELD_NUMBER: _ClassVar[int]
    STD_DEV_FIELD_NUMBER: _ClassVar[int]
    NUM_ZEROS_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_NUMERIC_STATS_FIELD_NUMBER: _ClassVar[int]
    common_stats: CommonStatistics
    mean: float
    std_dev: float
    num_zeros: int
    min: float
    median: float
    max: float
    histograms: _containers.RepeatedCompositeFieldContainer[Histogram]
    weighted_numeric_stats: WeightedNumericStatistics
    def __init__(self, common_stats: _Optional[_Union[CommonStatistics, _Mapping]] = ..., mean: _Optional[float] = ..., std_dev: _Optional[float] = ..., num_zeros: _Optional[int] = ..., min: _Optional[float] = ..., median: _Optional[float] = ..., max: _Optional[float] = ..., histograms: _Optional[_Iterable[_Union[Histogram, _Mapping]]] = ..., weighted_numeric_stats: _Optional[_Union[WeightedNumericStatistics, _Mapping]] = ...) -> None: ...

class StringStatistics(_message.Message):
    __slots__ = ("common_stats", "unique", "top_values", "avg_length", "rank_histogram", "weighted_string_stats")
    class FreqAndValue(_message.Message):
        __slots__ = ("deprecated_freq", "value", "frequency")
        DEPRECATED_FREQ_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        FREQUENCY_FIELD_NUMBER: _ClassVar[int]
        deprecated_freq: int
        value: str
        frequency: float
        def __init__(self, deprecated_freq: _Optional[int] = ..., value: _Optional[str] = ..., frequency: _Optional[float] = ...) -> None: ...
    COMMON_STATS_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELD_NUMBER: _ClassVar[int]
    TOP_VALUES_FIELD_NUMBER: _ClassVar[int]
    AVG_LENGTH_FIELD_NUMBER: _ClassVar[int]
    RANK_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_STRING_STATS_FIELD_NUMBER: _ClassVar[int]
    common_stats: CommonStatistics
    unique: int
    top_values: _containers.RepeatedCompositeFieldContainer[StringStatistics.FreqAndValue]
    avg_length: float
    rank_histogram: RankHistogram
    weighted_string_stats: WeightedStringStatistics
    def __init__(self, common_stats: _Optional[_Union[CommonStatistics, _Mapping]] = ..., unique: _Optional[int] = ..., top_values: _Optional[_Iterable[_Union[StringStatistics.FreqAndValue, _Mapping]]] = ..., avg_length: _Optional[float] = ..., rank_histogram: _Optional[_Union[RankHistogram, _Mapping]] = ..., weighted_string_stats: _Optional[_Union[WeightedStringStatistics, _Mapping]] = ...) -> None: ...

class WeightedNumericStatistics(_message.Message):
    __slots__ = ("mean", "std_dev", "median", "histograms")
    MEAN_FIELD_NUMBER: _ClassVar[int]
    STD_DEV_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMS_FIELD_NUMBER: _ClassVar[int]
    mean: float
    std_dev: float
    median: float
    histograms: _containers.RepeatedCompositeFieldContainer[Histogram]
    def __init__(self, mean: _Optional[float] = ..., std_dev: _Optional[float] = ..., median: _Optional[float] = ..., histograms: _Optional[_Iterable[_Union[Histogram, _Mapping]]] = ...) -> None: ...

class WeightedStringStatistics(_message.Message):
    __slots__ = ("top_values", "rank_histogram")
    TOP_VALUES_FIELD_NUMBER: _ClassVar[int]
    RANK_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    top_values: _containers.RepeatedCompositeFieldContainer[StringStatistics.FreqAndValue]
    rank_histogram: RankHistogram
    def __init__(self, top_values: _Optional[_Iterable[_Union[StringStatistics.FreqAndValue, _Mapping]]] = ..., rank_histogram: _Optional[_Union[RankHistogram, _Mapping]] = ...) -> None: ...

class BytesStatistics(_message.Message):
    __slots__ = ("common_stats", "unique", "avg_num_bytes", "min_num_bytes", "max_num_bytes")
    COMMON_STATS_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELD_NUMBER: _ClassVar[int]
    AVG_NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    MIN_NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    common_stats: CommonStatistics
    unique: int
    avg_num_bytes: float
    min_num_bytes: float
    max_num_bytes: float
    def __init__(self, common_stats: _Optional[_Union[CommonStatistics, _Mapping]] = ..., unique: _Optional[int] = ..., avg_num_bytes: _Optional[float] = ..., min_num_bytes: _Optional[float] = ..., max_num_bytes: _Optional[float] = ...) -> None: ...

class StructStatistics(_message.Message):
    __slots__ = ("common_stats",)
    COMMON_STATS_FIELD_NUMBER: _ClassVar[int]
    common_stats: CommonStatistics
    def __init__(self, common_stats: _Optional[_Union[CommonStatistics, _Mapping]] = ...) -> None: ...

class CommonStatistics(_message.Message):
    __slots__ = ("num_non_missing", "num_missing", "min_num_values", "max_num_values", "avg_num_values", "tot_num_values", "num_values_histogram", "weighted_common_stats", "feature_list_length_histogram")
    NUM_NON_MISSING_FIELD_NUMBER: _ClassVar[int]
    NUM_MISSING_FIELD_NUMBER: _ClassVar[int]
    MIN_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    AVG_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    TOT_NUM_VALUES_FIELD_NUMBER: _ClassVar[int]
    NUM_VALUES_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_COMMON_STATS_FIELD_NUMBER: _ClassVar[int]
    FEATURE_LIST_LENGTH_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    num_non_missing: int
    num_missing: int
    min_num_values: int
    max_num_values: int
    avg_num_values: float
    tot_num_values: int
    num_values_histogram: Histogram
    weighted_common_stats: WeightedCommonStatistics
    feature_list_length_histogram: Histogram
    def __init__(self, num_non_missing: _Optional[int] = ..., num_missing: _Optional[int] = ..., min_num_values: _Optional[int] = ..., max_num_values: _Optional[int] = ..., avg_num_values: _Optional[float] = ..., tot_num_values: _Optional[int] = ..., num_values_histogram: _Optional[_Union[Histogram, _Mapping]] = ..., weighted_common_stats: _Optional[_Union[WeightedCommonStatistics, _Mapping]] = ..., feature_list_length_histogram: _Optional[_Union[Histogram, _Mapping]] = ...) -> None: ...

class Histogram(_message.Message):
    __slots__ = ("num_nan", "num_undefined", "buckets", "type", "name")
    class HistogramType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STANDARD: _ClassVar[Histogram.HistogramType]
        QUANTILES: _ClassVar[Histogram.HistogramType]
    STANDARD: Histogram.HistogramType
    QUANTILES: Histogram.HistogramType
    class Bucket(_message.Message):
        __slots__ = ("low_value", "high_value", "deprecated_count", "sample_count")
        LOW_VALUE_FIELD_NUMBER: _ClassVar[int]
        HIGH_VALUE_FIELD_NUMBER: _ClassVar[int]
        DEPRECATED_COUNT_FIELD_NUMBER: _ClassVar[int]
        SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
        low_value: float
        high_value: float
        deprecated_count: int
        sample_count: float
        def __init__(self, low_value: _Optional[float] = ..., high_value: _Optional[float] = ..., deprecated_count: _Optional[int] = ..., sample_count: _Optional[float] = ...) -> None: ...
    NUM_NAN_FIELD_NUMBER: _ClassVar[int]
    NUM_UNDEFINED_FIELD_NUMBER: _ClassVar[int]
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    num_nan: int
    num_undefined: int
    buckets: _containers.RepeatedCompositeFieldContainer[Histogram.Bucket]
    type: Histogram.HistogramType
    name: str
    def __init__(self, num_nan: _Optional[int] = ..., num_undefined: _Optional[int] = ..., buckets: _Optional[_Iterable[_Union[Histogram.Bucket, _Mapping]]] = ..., type: _Optional[_Union[Histogram.HistogramType, str]] = ..., name: _Optional[str] = ...) -> None: ...

class RankHistogram(_message.Message):
    __slots__ = ("buckets", "name")
    class Bucket(_message.Message):
        __slots__ = ("low_rank", "high_rank", "deprecated_count", "label", "sample_count")
        LOW_RANK_FIELD_NUMBER: _ClassVar[int]
        HIGH_RANK_FIELD_NUMBER: _ClassVar[int]
        DEPRECATED_COUNT_FIELD_NUMBER: _ClassVar[int]
        LABEL_FIELD_NUMBER: _ClassVar[int]
        SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
        low_rank: int
        high_rank: int
        deprecated_count: int
        label: str
        sample_count: float
        def __init__(self, low_rank: _Optional[int] = ..., high_rank: _Optional[int] = ..., deprecated_count: _Optional[int] = ..., label: _Optional[str] = ..., sample_count: _Optional[float] = ...) -> None: ...
    BUCKETS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    buckets: _containers.RepeatedCompositeFieldContainer[RankHistogram.Bucket]
    name: str
    def __init__(self, buckets: _Optional[_Iterable[_Union[RankHistogram.Bucket, _Mapping]]] = ..., name: _Optional[str] = ...) -> None: ...

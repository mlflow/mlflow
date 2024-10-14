import re
from functools import reduce
from typing import Set, Union

try:
    # For spark >= 4.0
    from pyspark.errors.exceptions.base import IllegalArgumentException
except ModuleNotFoundError:
    from pyspark.sql.utils import IllegalArgumentException
from pyspark.ml.base import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import types as t


def cast_spark_df_with_vector_to_array(input_spark_df):  # noqa: D417
    """
    Finds columns of vector type in a spark dataframe and
    casts them to array<double> type.

    Args:
        input_spark_df:

    Returns:
        A spark dataframe with vector columns transformed to array<double> type
    """
    vector_type_columns = [
        _field.name for _field in input_spark_df.schema if isinstance(_field.dataType, VectorUDT)
    ]
    return reduce(
        lambda df, vector_col: df.withColumn(vector_col, vector_to_array(vector_col)),
        vector_type_columns,
        input_spark_df,
    )


def _do_pipeline_transform(df: DataFrame, transformer: Union[Transformer, PipelineModel]):  # noqa: D417
    """
    A util method that runs transform on a pipeline model/transformer

    Args:
        df: a spark dataframe

    Returns:
        output transformed dataframe using pipeline model/transformer
    """
    return transformer.transform(df)


def _get_struct_type_by_cols(input_fields: Set[str], df_schema: t.StructType) -> t.StructType:
    """
    Args:
        input_fields: A set of input columns to be
                    intersected with the input dataset's columns.
        df_schema: A Spark dataframe schema to compare input_fields

    Returns:
        A StructType from the intersection of given columns and
        the columns present in the training dataset
    """
    if len(input_fields) > 0:
        return t.StructType([_field for _field in df_schema.fields if _field.name in input_fields])
    return []


def get_feature_cols(
    df: DataFrame,
    transformer: Union[Transformer, PipelineModel],
) -> Set[str]:
    """
    Finds feature columns from an input dataset. If a dataset
    contains non-feature columns, those columns are not returned, but
    if `input_fields` is set to include non-feature columns those
    will be included in the return set of column names.

    Args:
        df: An input spark dataframe.
        transformer: A pipeline/transformer to get the required feature columns

    Returns:
        A set of all the feature columns that are required
        for the pipeline/transformer plus any initial columns passed in.
    """
    feature_cols = set()
    df_subset = df.limit(1).cache()
    for column in df.columns:
        try:
            transformer.transform(df_subset.drop(column))
        except IllegalArgumentException as iae:
            if re.search("does not exist|no such struct field", str(iae), re.IGNORECASE):
                feature_cols.add(column)
                continue
            raise
    df_subset.unpersist()
    return feature_cols

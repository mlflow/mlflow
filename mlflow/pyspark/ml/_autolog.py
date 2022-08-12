from pyspark.sql.utils import IllegalArgumentException
from pyspark.sql import SparkSession, DataFrame
import re
from functools import reduce
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import types as t
from pyspark.ml.base import Transformer
from pyspark.ml.pipeline import PipelineModel
from typing import Union, Set


def cast_spark_df_with_vector_to_array(input_spark_df):
    """
    Finds columns of vector type in a spark dataframe and
    casts them to array<double> type.

    :param input_spark_df:
    :return: a spark dataframe with vector columns transformed to array<double> type
    """
    vector_type_columns = [
        _field.name for _field in input_spark_df.schema if isinstance(_field.dataType, VectorUDT)
    ]
    return reduce(
        lambda df, vector_col: df.withColumn(vector_col, vector_to_array(vector_col)),
        vector_type_columns,
        input_spark_df,
    )


def _do_pipeline_transform(df: DataFrame, transformer: Union[Transformer, PipelineModel]):
    """
    A util method that runs transform on a pipeline model/transformer

    :param df:a spark dataframe
    :return: output transformed dataframe using pipeline model/transformer
    """
    return transformer.transform(df)


def _get_struct_type_by_cols(input_fields: Set[str], df_schema: t.StructType) -> t.StructType:
    """

    :param input_fields: A set of input columns to be
                 intersected with the input dataset's columns.
    :param df_schema: A Spark dataframe schema to compare input_fields
    :return:A StructType from the intersection of given columns and
            the columns present in the training dataset
    """
    if len(input_fields) > 0:
        return t.StructType([_field for _field in df_schema.fields if _field.name in input_fields])
    return []


def get_feature_cols(
    df_schema: t.StructType,
    transformer: Union[Transformer, PipelineModel],
    input_fields: Set[str] = None,
    spark: SparkSession = None,
) -> Set[str]:
    """
    Finds feature columns from an input dataset. If a dataset
    contains non-feature columns, those columns are not returned, but
    if `input_fields` is set to include non-feature columns those
    will be included in the return set of column names.

    :param df_schema: An input spark schema to look for the feature columns
    :param transformer: A pipeline/transformer to get the required feature columns
    :param input_fields: Initial columns to keep in the returned list
    :param spark: A spark session
    :return: A set of all the feature columns that are required
             for the pipeline/transformer plus any initial columns passed in.
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    if input_fields is None:
        input_fields = set()
        # Using an empty dataset doesn't work for all estimators
        # such as: pyspark.ml.classification.OneVsRest, so set
        # a single row and column of double type
        df = spark.createDataFrame([(1.0,)])
    else:
        df = spark.createDataFrame(
            spark.sparkContext.emptyRDD(),
            _get_struct_type_by_cols(input_fields, df_schema),
        )
    try:
        _do_pipeline_transform(df, transformer)
    except IllegalArgumentException as iae:
        col_name_search = re.search("(.*) does not exist.", iae.desc, re.IGNORECASE)
        if col_name_search:
            col_name = col_name_search.group(1)
            input_fields.add(col_name)
            get_feature_cols(df_schema, transformer, input_fields, spark)
    return input_fields

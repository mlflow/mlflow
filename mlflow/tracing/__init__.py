import json

from typing import List, Optional, Dict, Union 
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

import pandas as pd


def select_from_traces_df(df: Union[pd.DataFrame, 'pyspark.sql.DataFrame'], col_name: str, fields: List[str]) -> pd.DataFrame:
    try:
        from pyspark.sql import DataFrame as SparkDataFrame

        if isinstance(df, SparkDataFrame):
            return _select_from_traces_spark_df(df=df, col_name=col_name, fields=fields)
    except ImportError:
        pass

    if isinstance(df, pd.DataFrame):
        return _select_from_traces_pandas_df(df=df, col_name=col_name, fields=fields)

    raise MlflowException(
        message=(
            "`df` must be a pandas DataFrame or a Spark DataFrame. Got: {type(df)}"
        ),
        error_code=INVALID_PARAMETER_VALUE,
    )


def select_from_traces(traces: List['mlflow.entities.Trace'], fields: List[str]) -> pd.DataFrame:
    df = _traces_to_df(traces)
    return select_from_traces_df(df=df, col_name="spans", fields=fields)


def _traces_to_df(traces: List['mlflow.entities.Trace']) -> pd.DataFrame:
    """
    Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces"
    containing string representations of each Trace.
    """
    from mlflow.tracing.types.wrapper import Span

    rows = [
        TraceRow(
            request_id=trace.info.request_id,
            timestamp_ms=trace.info.timestamp_ms,
            status=trace.info.status,
            execution_time_ms=trace.info.execution_time_ms,
            request=trace.data.request,
            response=trace.data.response,
            request_metadata=trace.info.request_metadata,
            spans=trace.data.spans,
        )
        for trace in traces
    ]
    return pd.DataFrame.from_records([row.model_dump() for row in rows])


def _select_from_traces_pandas_df(df: pd.DataFrame, col_name: str, fields: List[str]) -> pd.DataFrame:
    spans = _extract_spans(df=df, col_name=col_name)
    # DO SOME SCHEMA MATCHING TO FIGURE OUT WHICH SCHEMA TO USE!

    if "spans" not in df.columns:
        raise MlflowException(
            message=(
                "Traces dataframe must contain a column named 'spans'. Got: {df.columns}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )


    pass


def _select_from_traces_spark_df():
    pass


def _extract_spans(df: pd.DataFrame, col_name: str) -> List['mlflow.tracing.types.wrapper.Span']:
    # Check various schemas and extract spans from the matching schema
    # df.select(...)
    pass


def _databricks_inference_table_to_traces_df(df: 'pyspark.sql.DataFrame') -> 'pyspark.sql.DataFrame':
    # df.withColumn(...)
    pass


@dataclass
class TraceRow:
    request_id: str
    timestamp_ms: int
    status: str
    execution_time_ms: int
    request: str
    response: Optional[str] = None
    request_metadata: Dict[str, str]
    spans: List[Span]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp_ms": self.timestamp_ms,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "request": self.request,
            "response": self.response,
            "request_metadata": self.request_metadata,
            "spans": [span.to_dict() for span in self.spans],
        }

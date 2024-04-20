import json
from collections import defaultdict
from dataclasses import dataclass

from typing import List, Optional, Dict, Union, Any, NamedTuple, Literal
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

import pandas as pd


def select_from_traces_df(df: Union[pd.DataFrame, 'pyspark.sql.DataFrame'], col_name: str, fields: List[str]) -> Union[pd.DataFrame, 'pyspark.sql.DataFrame']:
    parsed_fields = _parse_fields(fields)

    try:
        from pyspark.sql import DataFrame as SparkDataFrame

        if isinstance(df, SparkDataFrame):
            return _select_from_traces_spark_df(df=df, col_name=col_name, fields=parsed_fields)
    except ImportError:
        pass

    if isinstance(df, pd.DataFrame):
        return _select_from_traces_pandas_df(df=df, col_name=col_name, fields=parsed_fields)

    raise MlflowException(
        message=(
            "`df` must be a pandas DataFrame or a Spark DataFrame. Got: {type(df)}"
        ),
        error_code=INVALID_PARAMETER_VALUE,
    )


def select_from_traces(traces: List['mlflow.entities.Trace'], fields: List[str]) -> pd.DataFrame:
    df = _traces_to_df(traces)
    return select_from_traces_df(df=df, col_name="spans", fields=fields)


class _ParsedField(NamedTuple):
    span_name: str
    field_type: Literal["inputs", "outputs"]
    field_name: str

    @classmethod
    def from_string(cls, s: str) -> '_ParsedField':
        components = s.split(".")
        if len(components) != 3:
            raise MlflowException(
                message=f"Field must be of the form 'span_name.field_type.field_name'. Got: {s}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        span_name, field_type, field_name = components 

        if field_type not in ["inputs", "outputs"]:
            raise MlflowException(
                message=f"Field type must be 'inputs' or 'outputs'. Got: {field_type}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        
        return cls(span_name=span_name, field_type=field_type, field_name=field_name)

    def __str__(self) -> str:
        return f"{self.span_name}.{self.field_type}.{self.field_name}"


def _parse_fields(fields: List[str]) -> List['_ParsedField']:
    return [_ParsedField.from_string(field) for field in fields]


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
    return pd.DataFrame.from_records([row.to_dict() for row in rows])


def _select_from_traces_pandas_df(df: pd.DataFrame, col_name: str, fields: List[_ParsedField]) -> pd.DataFrame:
    from mlflow.tracing.types.wrapper import Span

    new_columns: Dict[str, List[Any]] = defaultdict(list)
    for _, row in df.iterrows():
        spans_dict: Dict[str, List[Span]] = defaultdict(list)
        for span in [Span.from_dict(span_dict) for span_dict in row["spans"]]:
            spans_dict[span.name].append(span)

        for field in fields:
            matching_spans = spans_dict.get(field.span_name, [])
            matching_values = []
            if field.field_type == "inputs" and isinstance(span.inputs, dict):
                matching_values = [span.inputs.get(field.field_name) for span in matching_spans]
            elif field.field_type == "outputs" and isinstance(span.outputs, dict):
                matching_values = [span.outputs.get(field.field_name) for span in matching_spans]
            new_columns[str(field)].append(matching_values[0] if matching_values else None)

    df_with_new_fields = df.copy()
    for field in fields:
        df_with_new_fields[str(field)] = new_columns[str(field)]

    return df_with_new_fields


def _select_from_traces_spark_df(df: 'pyspark.sql.DataFrame', col_name: str, fields: List[_ParsedField]) -> 'pyspark.sql.DataFrame':
    pass


def _extract_spans(df: pd.DataFrame, col_name: str) -> List[List['mlflow.tracing.types.wrapper.Span']]:

    return [
        [Span.from_dict(span_dict) for span_dict in row]
        for row in df["spans"].tolist()
    ]


def _databricks_inference_table_to_traces_df(df: 'pyspark.sql.DataFrame') -> 'pyspark.sql.DataFrame':
    # df.withColumn(...)
    pass


@dataclass
class TraceRow:
    from mlflow.tracing.types.wrapper import Span

    request_id: str
    timestamp_ms: int
    status: str
    execution_time_ms: int
    request: str
    request_metadata: Dict[str, str]
    spans: List[Span]
    response: Optional[str] = None

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

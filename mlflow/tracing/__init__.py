from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.utils import SPANS_COLUMN_NAME, traces_to_df

if TYPE_CHECKING:
    import pandas as pd

    import mlflow.entities


def extract(
    traces: Union[List["mlflow.entities.Trace"], "pd.DataFrame"],
    fields: List[str],
    col_name: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Extracts the specified fields from the spans contained in the specified traces.

    Args:
        traces: A list of MLflow Traces or a pandas DataFrame containing traces.
        fields: A list of field strings of the form 'span_name.[inputs|outputs]' or
            'span_name.[inputs|outputs].field_name'.
        col_name: The name of the column in the traces DataFrame containing the spans. If `traces`
            is a list of MLflow Traces, this argument should not be provided.
    """
    import pandas as pd

    parsed_fields = _parse_fields(fields)

    if isinstance(traces, list):
        if col_name is not None:
            raise MlflowException(
                message=(
                    "If `traces` is a list of MLflow Traces, `col_name` should not be provided."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        traces = traces_to_df(traces)
        col_name = SPANS_COLUMN_NAME

    if isinstance(traces, pd.DataFrame):
        return _extract_from_traces_pandas_df(df=traces, col_name=col_name, fields=parsed_fields)

    raise MlflowException(
        message=(
            "`traces` must be a list of MLflow Traces or a pandas DataFrame. Got: {type(traces)}"
        ),
        error_code=INVALID_PARAMETER_VALUE,
    )


class _ParsedField(NamedTuple):
    """
    Represents a parsed field from a string of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name'.
    """

    span_name: str
    field_type: Literal["inputs", "outputs"]
    field_name: Optional[str]

    @classmethod
    def from_string(cls, s: str) -> "_ParsedField":
        components = s.split(".")
        if len(components) not in [2, 3] or components[1] not in ["inputs", "outputs"]:
            raise MlflowException(
                message=(
                    f"Field must be of the form 'span_name.[inputs|outputs]' or"
                    f" 'span_name.[inputs|outputs].field_name'. Got: {s}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            span_name=components[0],
            field_type=components[1],
            field_name=components[2] if len(components) == 3 else None,
        )

    def __str__(self) -> str:
        return (
            f"{self.span_name}.{self.field_type}.{self.field_name}"
            if self.field_name is not None
            else f"{self.span_name}.{self.field_type}"
        )


def _parse_fields(fields: List[str]) -> List["_ParsedField"]:
    """
    Parses the specified field strings of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name' into _ParsedField objects.
    """
    return [_ParsedField.from_string(field) for field in fields]


def _extract_from_traces_pandas_df(
    df: "pd.DataFrame", col_name: str, fields: List[_ParsedField]
) -> "pd.DataFrame":
    """
    Extracts the specified fields from the spans contained in the specified column of the
    specified traces DataFrame.
    """

    from mlflow.tracing.types.wrapper import Span

    if col_name not in df.columns:
        raise MlflowException(
            message=(
                f"Column '{col_name}' not found in traces DataFrame."
                f" Available columns: {df.columns}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    new_columns: Dict[str, List[Any]] = defaultdict(list)
    for _, row in df.iterrows():
        spans_dict: Dict[str, List[Span]] = defaultdict(list)
        for span in _extract_spans_from_row(row[col_name]):
            spans_dict[span.name].append(span)

        for field in fields:
            matching_spans = spans_dict.get(field.span_name, [])
            matching_value = _find_matching_value(field, matching_spans)
            new_columns[str(field)].append(matching_value)

    df_with_new_fields = df.copy()
    for field in fields:
        df_with_new_fields[str(field)] = new_columns[str(field)]

    return df_with_new_fields


def _find_matching_value(
    field: _ParsedField, spans: List["mlflow.tracing.types.wrapper.Span"]
) -> Optional[Any]:
    """
    Find the value of the field in the list of spans. If the field is not found, return None.
    """
    for span in spans:
        span_inputs_or_outputs = getattr(span, field.field_type)
        if (
            isinstance(span_inputs_or_outputs, dict)
            and field.field_name is not None
            and field.field_name in span_inputs_or_outputs
        ):
            return span_inputs_or_outputs.get(field.field_name)
        elif field.field_name is None:
            return span_inputs_or_outputs


def _extract_spans_from_row(
    row_content: Optional[List[Dict[str, Any]]],
) -> List["mlflow.tracing.types.wrapper.Span"]:
    """
    Parses and extracts MLflow Spans from the row content of a traces pandas DataFrame.
    """
    from mlflow.tracing.types.wrapper import Span

    if row_content is None:
        return []

    try:
        return [Span.from_dict(span_dict) for span_dict in row_content]
    except Exception as e:
        raise MlflowException(
            message=(
                f"Failed to extract spans from traces DataFrame row content: {row_content}."
                f" Error: {e}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

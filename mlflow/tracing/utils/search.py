from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Optional, Union

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

SPANS_COLUMN_NAME = "spans"

if TYPE_CHECKING:
    import pandas

    import mlflow.entities
    from mlflow.entities import Trace


def traces_to_df(traces: List[Trace]) -> "pandas.DataFrame":
    """
    Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces"
    containing string representations of each Trace.
    """
    import pandas as pd

    from mlflow.entities.trace import Trace  # import here to avoid circular import

    rows = [trace.to_pandas_dataframe_row() for trace in traces]
    return pd.DataFrame.from_records(data=rows, columns=Trace.pandas_dataframe_columns())


def extract_span_inputs_outputs(
    traces: Union[List["mlflow.entities.Trace"], "pandas.DataFrame"],
    fields: List[str],
    col_name: Optional[str] = None,
) -> "pandas.DataFrame":
    """
    Extracts the specified input and output fields from the spans contained in the specified traces.

    Args:
        traces: A list of :py:class:`mlflow.entities.Trace` or a pandas DataFrame containing traces.
        fields: A list of field strings of the form 'span_name.[inputs|outputs]' or
            'span_name.[inputs|outputs].field_name'.
        col_name: The name of the column in the traces DataFrame containing the spans. If `traces`
            is a list of MLflow Traces, this argument should not be provided.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise MlflowException(
            message=(
                "The `pandas` library is not installed. Please install `pandas` to use the"
                f"`mlflow.tracing.extract` function. Error: {e}"
            ),
        )

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


class _PeekableIterator:
    """
    Wraps an iterator and allows peeking at the next element without consuming it.
    """

    def __init__(self, it):
        self.it = iter(it)
        self._next = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._next is not None:
            next_value = self._next
            self._next = None
            return next_value
        return next(self.it)

    def peek(self):
        if self._next is None:
            try:
                self._next = next(self.it)
            except StopIteration:
                return None
        return self._next


class _ParsedField(NamedTuple):
    """
    Represents a parsed field from a string of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name'.
    """

    span_name: str
    field_type: Literal["inputs", "outputs"]
    field_name: Optional[str]

    def __str__(self) -> str:
        return (
            f"{self.span_name}.{self.field_type}.{self.field_name}"
            if self.field_name is not None
            else f"{self.span_name}.{self.field_type}"
        )


_BACKTICK = "`"


class _FieldParser:
    def __init__(self, field: str) -> None:
        self.field = field
        self.chars = _PeekableIterator(field)

    def peek(self) -> str:
        return self.chars.peek()

    def next(self) -> str:
        return next(self.chars)

    def has_next(self) -> bool:
        return self.peek() is not None

    def consume_until_char_or_end(self, stop_char: Optional[str] = None) -> str:
        """
        Consume characters until the specified character is encountered or the end of the
        string. If char is None, consume until the end of the string.
        """
        consumed = ""
        while (c := self.peek()) and c != stop_char:
            consumed += self.next()
        return consumed

    def _parse_span_name(self) -> str:
        if self.peek() == _BACKTICK:
            self.next()
            span_name = self.consume_until_char_or_end(_BACKTICK)
            if self.peek() != _BACKTICK:
                raise MlflowException.invalid_parameter_value(
                    f"Expected closing backtick: {self.field!r}"
                )
            self.next()
        else:
            span_name = self.consume_until_char_or_end(".")

        if self.peek() != ".":
            raise MlflowException.invalid_parameter_value(
                f"Expected dot after span name: {self.field!r}"
            )
        self.next()
        return span_name

    def _parse_field_type(self) -> str:
        field_type = self.consume_until_char_or_end(".")
        if field_type not in ("inputs", "outputs"):
            raise MlflowException.invalid_parameter_value(
                f"Invalid field type: {field_type!r}. Expected 'inputs' or 'outputs'."
            )

        if self.has_next():
            self.next()  # Consume the dot
        return field_type

    def _parse_field_name(self) -> str:
        if self.peek() == _BACKTICK:
            self.next()
            field_name = self.consume_until_char_or_end(_BACKTICK)
            if self.peek() != _BACKTICK:
                raise MlflowException.invalid_parameter_value(
                    f"Expected closing backtick: {self.field!r}"
                )
            self.next()

            # There should be no more characters after the closing backtick
            if self.has_next():
                raise MlflowException.invalid_parameter_value(
                    f"Unexpected characters after closing backtick: {self.field!r}"
                )

        else:
            field_name = self.consume_until_char_or_end()

        return field_name

    def parse(self) -> _ParsedField:
        span_name = self._parse_span_name()
        field_type = self._parse_field_type()
        field_name = self._parse_field_name() if self.has_next() else None
        return _ParsedField(span_name=span_name, field_type=field_type, field_name=field_name)


def _parse_fields(fields: List[str]) -> List[_ParsedField]:
    """
    Parses the specified field strings of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name' into _ParsedField objects.
    """
    return [_FieldParser(field).parse() for field in fields]


def _extract_from_traces_pandas_df(
    df: "pandas.DataFrame", col_name: str, fields: List[_ParsedField]
) -> "pandas.DataFrame":
    """
    Extracts the specified fields from the spans contained in the specified column of the
    specified traces DataFrame.
    """

    from mlflow.entities import Span

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


def _find_matching_value(field: _ParsedField, spans: List["mlflow.entities.Span"]) -> Optional[Any]:
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
) -> List["mlflow.entities.Span"]:
    """
    Parses and extracts MLflow Spans from the row content of a traces pandas DataFrame.
    """
    from mlflow.entities import Span

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

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from mlflow.exceptions import MlflowException

SPANS_COLUMN_NAME = "spans"

if TYPE_CHECKING:
    import pandas

    import mlflow.entities
    from mlflow.entities import Trace


# sqlparse 0.4.3 and newer releases differ on paired backslashes before a quote.
_TRACE_FILTER_QUOTED_TOKENS = (
    r"""'(?:''|\\'|[^'])*'|"(?:""|\\"|[^"])*"|`(?:``|[^`])*`""",
    r"""'(?:''|\\\\|\\'|[^'])*'|"(?:""|\\\\|\\"|[^"])*"|`(?:``|[^`])*`""",
)
_TRACE_METADATA_FILTER_FIELDS = {
    "request_metadata.`mlflow.modelId`",
    "metadata.`mlflow.trace.session`",
}


def format_trace_metadata_filter(field: str, value: str) -> str:
    """Frame fixed metadata IDs as one exact literal under supported server tokenizers."""
    value = str(value)
    if field not in _TRACE_METADATA_FILTER_FIELDS or any(
        (literal := re.match(pattern, f"'{value}'")) is None or literal.end() != len(value) + 2
        for pattern in _TRACE_FILTER_QUOTED_TOKENS
    ):
        raise MlflowException.invalid_parameter_value(
            f"Value cannot be represented as a single equality filter for {field}"
        )
    # Metadata literals retain their raw contents, including doubled quotes and backslashes.
    return f"{field} = '{value}'"


def combine_trace_filters(*filters: str | None) -> str:
    """Conjoin filters, grouping top-level OR without rewriting the supplied expressions."""
    filters = [expression for expression in filters if expression]
    if len(filters) < 2:
        return "".join(filters)
    clauses = []
    for expression in filters:
        has_or = False
        for quoted_tokens in _TRACE_FILTER_QUOTED_TOKENS:
            depth = 0
            for match in re.finditer(
                quoted_tokens + r"|[()]|(?<![\w.$#])OR(?![\w.$#])", expression, re.IGNORECASE
            ):
                token = match.group()
                if token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                elif token.upper() == "OR" and depth == 0:
                    has_or = True
                    break
            if has_or:
                break
        clauses.append(f"({expression})" if has_or else expression)
    return " AND ".join(clauses)


def traces_to_df(
    traces: list[Trace], extract_fields: list[str] | None = None
) -> "pandas.DataFrame":
    """
    Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces"
    containing string representations of each Trace.
    """
    import pandas as pd

    from mlflow.entities.trace import Trace  # import here to avoid circular import

    rows = []
    columns = Trace.pandas_dataframe_columns()
    parsed_fields = []
    # update columns to ensure they exist in result dataframe in case traces are empty
    if extract_fields is not None:
        parsed_fields = _parse_fields(extract_fields)
        columns.extend([str(field) for field in parsed_fields])

    for trace in traces:
        row = trace.to_pandas_dataframe_row()
        for field in parsed_fields:
            row[str(field)] = None
            for span in trace.data.spans:
                if field.span_name == span.name:
                    row[str(field)] = _extract_field_from_span(span, field)
        rows.append(row)

    return pd.DataFrame.from_records(
        data=rows,
        columns=columns,
    )


def _extract_field_from_span(span: "mlflow.entities.Span", field: _ParsedField) -> Any | None:
    span_inputs_or_outputs = getattr(span, field.field_type)
    if (
        isinstance(span_inputs_or_outputs, dict)
        and field.field_name is not None
        and field.field_name in span_inputs_or_outputs
    ):
        return span_inputs_or_outputs.get(field.field_name)
    elif field.field_name is None:
        return span_inputs_or_outputs


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
    field_name: str | None

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

    def consume_until_char_or_end(self, stop_char: str | None = None) -> str:
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


def _parse_fields(fields: list[str]) -> list[_ParsedField]:
    """
    Parses the specified field strings of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name' into _ParsedField objects.
    """
    return [_FieldParser(field).parse() for field in fields]

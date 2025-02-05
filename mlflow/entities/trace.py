from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


@dataclass
class Trace(_MlflowObject):
    """A trace object.

    Args:
        info: A lightweight object that contains the metadata of a trace.
        data: A container object that holds the spans data of a trace.
    """

    info: TraceInfo
    data: TraceData

    def __repr__(self) -> str:
        return f"Trace(request_id={self.info.request_id})"

    def to_dict(self) -> dict[str, Any]:
        return {"info": self.info.to_dict(), "data": self.data.to_dict()}

    def to_json(self, pretty=False) -> str:
        from mlflow.tracing.utils import TraceJSONEncoder

        return json.dumps(self.to_dict(), cls=TraceJSONEncoder, indent=2 if pretty else None)

    @classmethod
    def from_dict(cls, trace_dict: dict[str, Any]) -> Trace:
        info = trace_dict.get("info")
        data = trace_dict.get("data")
        if info is None or data is None:
            raise MlflowException(
                "Unable to parse Trace from dictionary. Expected keys: 'info' and 'data'. "
                f"Received keys: {list(trace_dict.keys())}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            info=TraceInfo.from_dict(info),
            data=TraceData.from_dict(data),
        )

    @classmethod
    def from_json(cls, trace_json: str) -> Trace:
        try:
            trace_dict = json.loads(trace_json)
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Unable to parse trace JSON: {trace_json}. Error: {e}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls.from_dict(trace_dict)

    def _serialize_for_mimebundle(self):
        # databricks notebooks will use the request ID to
        # fetch the trace from the backend. including the
        # full JSON can cause notebooks to exceed size limits
        return json.dumps(self.info.request_id)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """
        This method is used to trigger custom display logic in IPython notebooks.
        See https://ipython.readthedocs.io/en/stable/config/integrating.html#MyObject
        for more details.

        At the moment, the only supported MIME type is "application/databricks.mlflow.trace",
        which contains a JSON representation of the Trace object. This object is deserialized
        in Databricks notebooks to display the Trace object in a nicer UI.
        """
        from mlflow.tracing.display import (
            get_display_handler,
            get_notebook_iframe_html,
            is_using_tracking_server,
        )
        from mlflow.utils.databricks_utils import is_in_databricks_runtime

        bundle = {"text/plain": repr(self)}

        if not get_display_handler().disabled:
            if is_in_databricks_runtime():
                bundle["application/databricks.mlflow.trace"] = self._serialize_for_mimebundle()
            elif is_using_tracking_server():
                bundle["text/html"] = get_notebook_iframe_html([self])

        return bundle

    def to_pandas_dataframe_row(self) -> dict[str, Any]:
        return {
            "request_id": self.info.request_id,
            "trace": self,
            "timestamp_ms": self.info.timestamp_ms,
            "status": self.info.status,
            "execution_time_ms": self.info.execution_time_ms,
            "request": self._deserialize_json_attr(self.data.request),
            "response": self._deserialize_json_attr(self.data.response),
            "request_metadata": self.info.request_metadata,
            "spans": [span.to_dict() for span in self.data.spans],
            "tags": self.info.tags,
        }

    def _deserialize_json_attr(self, value: str):
        try:
            return json.loads(value)
        except Exception:
            _logger.debug(f"Failed to deserialize JSON attribute: {value}", exc_info=True)
            return value

    def search_spans(
        self, span_type: Optional[SpanType] = None, name: Optional[Union[str, re.Pattern]] = None
    ) -> list[Span]:
        """
        Search for spans that match the given criteria within the trace.

        Args:
            span_type: The type of the span to search for.
            name: The name of the span to search for. This can be a string or a regular expression.

        Returns:
            A list of spans that match the given criteria.
            If there is no match, an empty list is returned.

        .. code-block:: python

            import mlflow
            import re
            from mlflow.entities import SpanType


            @mlflow.trace(span_type=SpanType.CHAIN)
            def run(x: int) -> int:
                x = add_one(x)
                x = add_two(x)
                x = multiply_by_two(x)
                return x


            @mlflow.trace(span_type=SpanType.TOOL)
            def add_one(x: int) -> int:
                return x + 1


            @mlflow.trace(span_type=SpanType.TOOL)
            def add_two(x: int) -> int:
                return x + 2


            @mlflow.trace(span_type=SpanType.TOOL)
            def multiply_by_two(x: int) -> int:
                return x * 2


            # Run the function and get the trace
            y = run(2)
            trace = mlflow.get_last_active_trace()

            # 1. Search spans by name (exact match)
            spans = trace.search_spans(name="add_one")
            print(spans)
            # Output: [Span(name='add_one', ...)]

            # 2. Search spans by name (regular expression)
            pattern = re.compile(r"add.*")
            spans = trace.search_spans(name=pattern)
            print(spans)
            # Output: [Span(name='add_one', ...), Span(name='add_two', ...)]

            # 3. Search spans by type
            spans = trace.search_spans(span_type=SpanType.LLM)
            print(spans)
            # Output: [Span(name='run', ...)]

            # 4. Search spans by name and type
            spans = trace.search_spans(name="add_one", span_type=SpanType.TOOL)
            print(spans)
            # Output: [Span(name='add_one', ...)]
        """

        def _match_name(span: Span) -> bool:
            if isinstance(name, str):
                return span.name == name
            elif isinstance(name, re.Pattern):
                return name.search(span.name) is not None
            elif name is None:
                return True
            else:
                raise MlflowException(
                    f"Invalid type for 'name'. Expected str or re.Pattern. Got: {type(name)}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        def _match_type(span: Span) -> bool:
            if isinstance(span_type, str):
                return span.span_type == span_type
            elif span_type is None:
                return True
            else:
                raise MlflowException(
                    "Invalid type for 'span_type'. Expected str or mlflow.entities.SpanType. "
                    f"Got: {type(span_type)}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        return [span for span in self.data.spans if _match_name(span) and _match_type(span)]

    @staticmethod
    def pandas_dataframe_columns() -> list[str]:
        return [
            "request_id",
            "trace",
            "timestamp_ms",
            "status",
            "execution_time_ms",
            "request",
            "response",
            "request_metadata",
            "spans",
            "tags",
        ]

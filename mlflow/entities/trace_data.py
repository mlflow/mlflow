from dataclasses import dataclass, field
from typing import Any, Optional

from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey


@dataclass
class TraceData:
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
        request: Input data for the entire trace. Equivalent to the input of the root span
            but added for ease of access. Stored as a JSON string.
        response: Output data for the entire trace. Equivalent to the output of the root span.
            Stored as a JSON string.
    """

    spans: list[Span] = field(default_factory=list)
    request: Optional[str] = None
    response: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise TypeError(f"TraceData.from_dict() expects a dictionary. Got: {type(d).__name__}")
        return cls(
            request=d.get("request"),
            response=d.get("response"),
            spans=[Span.from_dict(span) for span in d.get("spans", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spans": [span.to_dict() for span in self.spans],
            "request": self.request,
            "response": self.response,
        }

    @property
    def intermediate_outputs(self) -> Optional[dict[str, Any]]:
        """
        Returns intermediate outputs produced by the model or agent while handling the request.
        There are mainly two flows to return intermediate outputs:
        1. When a trace is generate by the `mlflow.log_trace` API,
        return `intermediate_outputs` attribute of the span.
        2. When a trace is created normally with a tree of spans,
        aggregate the outputs of non-root spans.
        """
        root_span = self._get_root_span()
        if root_span and root_span.get_attribute(SpanAttributeKey.INTERMEDIATE_OUTPUTS):
            return root_span.get_attribute(SpanAttributeKey.INTERMEDIATE_OUTPUTS)

        if len(self.spans) > 1:
            return {
                span.name: span.outputs
                for span in self.spans
                if span.parent_id and span.outputs is not None
            }

    def _get_root_span(self) -> Optional[Span]:
        for span in self.spans:
            if span.parent_id is None:
                return span

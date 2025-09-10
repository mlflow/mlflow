import json
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey


@dataclass
class TraceData:
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
    """

    spans: list[Span] = field(default_factory=list)

    # NB: Custom constructor to allow passing additional kwargs for backward compatibility for
    # DBX agent evaluator. Once they migrates to trace V3 schema, we can remove this.
    def __init__(self, spans: list[Span] | None = None, **kwargs):
        self.spans = spans or []

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise TypeError(f"TraceData.from_dict() expects a dictionary. Got: {type(d).__name__}")
        return cls(spans=[Span.from_dict(span) for span in d.get("spans", [])])

    def to_dict(self) -> dict[str, Any]:
        return {"spans": [span.to_dict() for span in self.spans]}

    def to_dict_for_export(self) -> dict[str, Any]:
        """
        Convert to dict for external export (e.g., trace artifacts).
        This method deserializes JSON-stringified attributes to prevent double stringification.
        """
        result = {"spans": []}
        for span in self.spans:
            span_dict = span.to_dict()

            if "attributes" in span_dict:
                keys_to_deserialize = [
                    SpanAttributeKey.INPUTS,
                    SpanAttributeKey.OUTPUTS,
                    SpanAttributeKey.SPAN_TYPE,
                ]
                for key in keys_to_deserialize:
                    if key in span_dict["attributes"] and isinstance(
                        span_dict["attributes"][key], str
                    ):
                        try:
                            span_dict["attributes"][key] = json.loads(span_dict["attributes"][key])
                        except (json.JSONDecodeError, TypeError):
                            pass

            result["spans"].append(span_dict)

        return result

    @property
    def intermediate_outputs(self) -> dict[str, Any] | None:
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

    def _get_root_span(self) -> Span | None:
        for span in self.spans:
            if span.parent_id is None:
                return span

    # `request` and `response` are preserved for backward compatibility with v2
    @property
    def request(self) -> str | None:
        if span := self._get_root_span():
            # Accessing the OTel span directly get serialized value directly.
            return span._span.attributes.get(SpanAttributeKey.INPUTS)
        return None

    @property
    def response(self) -> str | None:
        if span := self._get_root_span():
            # Accessing the OTel span directly get serialized value directly.
            return span._span.attributes.get(SpanAttributeKey.OUTPUTS)
        return None

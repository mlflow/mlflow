from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import Span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.annotations import deprecated


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

    # TODO: remove this property in 3.7.0
    @property
    @deprecated(since="3.6.0", alternative="trace.search_spans(name=...)")
    def intermediate_outputs(self) -> dict[str, Any] | None:
        """
        .. deprecated:: 3.6.0
            Use `trace.search_spans(name=...)` to search for spans and get the outputs.

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
            result = {}
            # spans may have duplicate names, so deduplicate the names by appending an index number.
            span_name_counter = Counter(span.name for span in self.spans)
            span_name_counter = {name: 1 for name, count in span_name_counter.items() if count > 1}
            for span in self.spans:
                span_name = span.name
                if count := span_name_counter.get(span_name):
                    span_name_counter[span_name] += 1
                    span_name = f"{span_name}_{count}"
                if span.parent_id and span.outputs is not None:
                    result[span_name] = span.outputs
            return result

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

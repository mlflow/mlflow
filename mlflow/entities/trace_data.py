from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlflow.entities import Span


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

    spans: List[Span] = field(default_factory=list)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spans": [span.to_dict() for span in self.spans],
            "request": self.request,
            "response": self.response,
        }

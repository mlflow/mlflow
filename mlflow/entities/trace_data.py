from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from mlflow.entities.span import Span


@dataclass
class TraceData:
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
        request: Input data for the entire trace. Equivalent to the input of the root span
            but added for ease of access.
        response: Output data for the entire trace. Equivalent to the output of the root span.
    """

    spans: List[Span] = field(default_factory=list)
    request: Optional[Any] = None
    response: Optional[Any] = None

    @classmethod
    def from_dict(cls, d):
        return cls(spans=[Span(**span) for span in d["spans"]])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

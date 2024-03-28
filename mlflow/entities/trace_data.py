from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.span import Span


@dataclass
class TraceData(_MLflowObject):
    """A container object that holds the spans data of a trace.

    Args:
        spans: List of spans that are part of the trace.
    """

    spans: List[Span] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        return cls(spans=[Span(**span) for span in d["spans"]])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

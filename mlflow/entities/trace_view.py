from __future__ import annotations

import json
from dataclasses import dataclass, field

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class SpanSelector:
    span_name: str | None = None
    span_type: str | None = None
    span_id: str | None = None
    attribute_key: str | None = None
    attribute_value: str | None = None

    def to_dict(self) -> dict:
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "span_id": self.span_id,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpanSelector:
        return cls(
            span_name=d.get("span_name"),
            span_type=d.get("span_type"),
            span_id=d.get("span_id"),
            attribute_key=d.get("attribute_key"),
            attribute_value=d.get("attribute_value"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SpanSelector:
        return cls.from_dict(json.loads(s))


@dataclass
class PathSelection:
    span_selector: SpanSelector
    path: str

    def to_dict(self) -> dict:
        return {
            "span_selector": self.span_selector.to_dict(),
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PathSelection:
        return cls(
            span_selector=SpanSelector.from_dict(d["span_selector"]),
            path=d["path"],
        )


@dataclass
class SpanRange:
    from_selector: SpanSelector
    to_selector: SpanSelector | None = None
    label: str = ""
    description: str = ""
    input_path: str | None = None
    output_path: str | None = None
    input_selections: list[PathSelection] = field(default_factory=list)
    output_selections: list[PathSelection] = field(default_factory=list)
    position: int = 0
    range_id: str | None = None

    def to_dict(self) -> dict:
        d = {
            "from_selector": self.from_selector.to_dict(),
            "to_selector": self.to_selector.to_dict() if self.to_selector else None,
            "label": self.label,
            "description": self.description,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "position": self.position,
            "range_id": self.range_id,
        }
        if self.input_selections:
            d["input_selections"] = [s.to_dict() for s in self.input_selections]
        if self.output_selections:
            d["output_selections"] = [s.to_dict() for s in self.output_selections]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SpanRange:
        to_sel = d.get("to_selector")
        return cls(
            from_selector=SpanSelector.from_dict(d["from_selector"]),
            to_selector=SpanSelector.from_dict(to_sel) if to_sel else None,
            label=d.get("label", ""),
            description=d.get("description", ""),
            input_path=d.get("input_path"),
            output_path=d.get("output_path"),
            input_selections=[
                PathSelection.from_dict(s) for s in d.get("input_selections", [])
            ],
            output_selections=[
                PathSelection.from_dict(s) for s in d.get("output_selections", [])
            ],
            position=d.get("position", 0),
            range_id=d.get("range_id"),
        )


@dataclass
class TraceView:
    name: str
    trace_id: str | None = None
    experiment_id: str | None = None
    ranges: list[SpanRange] = field(default_factory=list)
    created_by: str | None = None
    view_id: str | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None

    def validate_scope(self) -> None:
        has_trace = self.trace_id is not None
        has_experiment = self.experiment_id is not None
        if has_trace == has_experiment:
            raise MlflowException(
                "Exactly one of trace_id or experiment_id must be set.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @property
    def scope(self) -> str:
        if self.trace_id is not None:
            return "trace"
        return "experiment"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "experiment_id": self.experiment_id,
            "ranges": [r.to_dict() for r in self.ranges],
            "created_by": self.created_by,
            "view_id": self.view_id,
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceView:
        ranges_data = d.get("ranges", [])
        return cls(
            name=d["name"],
            trace_id=d.get("trace_id"),
            experiment_id=d.get("experiment_id"),
            ranges=[SpanRange.from_dict(r) for r in ranges_data],
            created_by=d.get("created_by"),
            view_id=d.get("view_id"),
            create_time_ms=d.get("create_time_ms"),
            last_update_time_ms=d.get("last_update_time_ms"),
        )

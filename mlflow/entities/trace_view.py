from __future__ import annotations

import json
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


@dataclass
class SpanFilter:
    span_name: str | None = None
    span_type: str | None = None
    attribute_key: str | None = None
    attribute_value: str | None = None

    def to_dict(self) -> dict:
        return {
            "span_name": self.span_name,
            "span_type": self.span_type,
            "attribute_key": self.attribute_key,
            "attribute_value": self.attribute_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SpanFilter:
        return cls(
            span_name=d.get("span_name"),
            span_type=d.get("span_type"),
            attribute_key=d.get("attribute_key"),
            attribute_value=d.get("attribute_value"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> SpanFilter:
        return cls.from_dict(json.loads(s))


@dataclass
class TraceView:
    name: str
    trace_id: str | None = None
    experiment_id: str | None = None
    span_filter: SpanFilter | None = None
    input_path: str | None = None
    output_path: str | None = None
    created_by: str | None = None
    description: str | None = None
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
            "span_filter": self.span_filter.to_dict() if self.span_filter else None,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "created_by": self.created_by,
            "description": self.description,
            "view_id": self.view_id,
            "create_time_ms": self.create_time_ms,
            "last_update_time_ms": self.last_update_time_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceView:
        span_filter_data = d.get("span_filter")
        return cls(
            name=d["name"],
            trace_id=d.get("trace_id"),
            experiment_id=d.get("experiment_id"),
            span_filter=SpanFilter.from_dict(span_filter_data) if span_filter_data else None,
            input_path=d.get("input_path"),
            output_path=d.get("output_path"),
            created_by=d.get("created_by"),
            description=d.get("description"),
            view_id=d.get("view_id"),
            create_time_ms=d.get("create_time_ms"),
            last_update_time_ms=d.get("last_update_time_ms"),
        )

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from google.protobuf.json_format import MessageToDict

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.protos.evaluation_datasets_pb2 import DatasetRecord as ProtoDatasetRecord
from mlflow.protos.evaluation_datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource
from mlflow.utils.time import get_current_time_millis


@dataclass
class DatasetRecord(_MlflowObject):
    dataset_id: str
    inputs: dict[str, Any]
    dataset_record_id: Optional[str] = None
    expectations: Optional[dict[str, Any]] = None
    tags: Optional[dict[str, str]] = None
    source: Optional[DatasetRecordSource] = None
    source_id: Optional[str] = None
    source_type: Optional[str] = None
    created_time: Optional[int] = None
    last_update_time: Optional[int] = None
    created_by: Optional[str] = None
    last_updated_by: Optional[str] = None

    def __post_init__(self):
        if self.dataset_record_id is None:
            self.dataset_record_id = str(uuid.uuid4())

        if self.created_time is None:
            self.created_time = get_current_time_millis()
        if self.last_update_time is None:
            self.last_update_time = get_current_time_millis()

        if self.tags is None:
            self.tags = {}

        if self.source and isinstance(self.source, DatasetRecordSource):
            if not self.source_id:
                if self.source.source_type == DatasetRecordSourceType.TRACE:
                    self.source_id = self.source.source_data.get("trace_id")
                else:
                    self.source_id = self.source.source_data.get("source_id")
            if not self.source_type:
                self.source_type = self.source.source_type.value

    def to_proto(self) -> ProtoDatasetRecord:
        proto = ProtoDatasetRecord()

        if self.dataset_record_id is not None:
            proto.dataset_record_id = self.dataset_record_id
        if self.dataset_id is not None:
            proto.dataset_id = self.dataset_id
        if self.inputs is not None:
            proto.inputs = json.dumps(self.inputs)
        if self.expectations is not None:
            proto.expectations = json.dumps(self.expectations)
        if self.tags is not None:
            proto.tags = json.dumps(self.tags)
        if self.source is not None:
            proto.source = json.dumps(self.source.to_dict())
        if self.source_id is not None:
            proto.source_id = self.source_id
        if self.source_type is not None:
            proto.source_type = ProtoDatasetRecordSource.SourceType.Value(self.source_type)
        if self.created_time is not None:
            proto.created_time = self.created_time
        if self.last_update_time is not None:
            proto.last_update_time = self.last_update_time
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by

        return proto

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecord) -> "DatasetRecord":
        inputs = json.loads(proto.inputs) if proto.HasField("inputs") else {}
        expectations = json.loads(proto.expectations) if proto.HasField("expectations") else None
        tags = json.loads(proto.tags) if proto.HasField("tags") else None

        source = None
        if proto.HasField("source"):
            source_dict = json.loads(proto.source)
            source = DatasetRecordSource.from_dict(source_dict)

        return cls(
            dataset_id=proto.dataset_id if proto.HasField("dataset_id") else "",
            inputs=inputs,
            dataset_record_id=proto.dataset_record_id
            if proto.HasField("dataset_record_id")
            else None,
            expectations=expectations,
            tags=tags,
            source=source,
            source_id=proto.source_id if proto.HasField("source_id") else None,
            source_type=DatasetRecordSourceType.from_proto(proto.source_type)
            if proto.HasField("source_type")
            else None,
            created_time=proto.created_time if proto.HasField("created_time") else None,
            last_update_time=proto.last_update_time if proto.HasField("last_update_time") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d = MessageToDict(
            self.to_proto(),
            preserving_proto_field_name=True,
        )
        d["inputs"] = json.loads(d["inputs"])
        if "expectations" in d:
            d["expectations"] = json.loads(d["expectations"])
        if "tags" in d:
            d["tags"] = json.loads(d["tags"])
        if "source" in d:
            d["source"] = json.loads(d["source"])
        d["created_time"] = self.created_time
        d["last_update_time"] = self.last_update_time
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        source = None
        if data.get("source"):
            source = DatasetRecordSource.from_dict(data["source"])

        return cls(
            dataset_id=data.get("dataset_id", ""),
            inputs=data.get("inputs", {}),
            dataset_record_id=data.get("dataset_record_id"),
            expectations=data.get("expectations"),
            tags=data.get("tags"),
            source=source,
            source_id=data.get("source_id"),
            source_type=data.get("source_type"),
            created_time=data.get("created_time"),
            last_update_time=data.get("last_update_time"),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetRecord):
            return False
        return (
            self.dataset_record_id == other.dataset_record_id
            and self.dataset_id == other.dataset_id
            and self.inputs == other.inputs
            and self.expectations == other.expectations
            and self.tags == other.tags
            and self.source == other.source
            and self.source_id == other.source_id
            and self.source_type == other.source_type
        )

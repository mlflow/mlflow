from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.protos.datasets_pb2 import DatasetRecord as ProtoDatasetRecord
from mlflow.protos.datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource

# Reserved key for wrapping non-dict outputs when storing in SQL database
DATASET_RECORD_WRAPPED_OUTPUT_KEY = "mlflow_wrapped"

# Reserved record tag holding the names of the scorers to run on that record alone.
# Record tags are ``dict[str, str]``, so the names are stored comma-separated. This
# piggybacks on tags rather than adding a dedicated field so that per-record scorers
# need no dataset schema change.
DATASET_RECORD_SCORERS_TAG = "mlflow.scorers"


def encode_record_scorers(scorer_names: list[str]) -> str:
    return ",".join(scorer_names)


def decode_record_scorers(tag_value: str | None) -> list[str]:
    if not tag_value:
        return []
    return [name for part in tag_value.split(",") if (name := part.strip())]


def validate_record_scorer_names(scorers: Any) -> list[str]:
    """Validate a record's ``scorers`` value and return it as a list of scorer names.

    Only names are accepted. Passing a ``Scorer`` object is rejected because a persisted
    record can only carry a reference, not a definition.
    """
    from mlflow.exceptions import MlflowException

    if scorers is None or isinstance(scorers, float):  # float covers a NaN cell from pandas
        return []

    if isinstance(scorers, str):
        raise MlflowException.invalid_parameter_value(
            f"The record `scorers` field must be a list of scorer names, got the string "
            f"{scorers!r}. Use `scorers=[{scorers!r}]`."
        )

    if not isinstance(scorers, (list, tuple)):
        raise MlflowException.invalid_parameter_value(
            f"The record `scorers` field must be a list of scorer names, got "
            f"{type(scorers).__name__}."
        )

    for name in scorers:
        if isinstance(name, str):
            continue
        # Scorer objects are the most likely mistake here, so name that case explicitly.
        hint = (
            " Pass the scorer's registered name instead, e.g. "
            f"`scorers=[{getattr(name, 'name', 'my_scorer')!r}]`."
            if hasattr(name, "name")
            else ""
        )
        raise MlflowException.invalid_parameter_value(
            f"The record `scorers` field must contain scorer names, got an item of type "
            f"{type(name).__name__}.{hint}"
        )

    return list(scorers)


def fold_record_scorers_into_tags(record_dicts: list[dict[str, Any]]) -> None:
    """Move each record's ``scorers`` list into its ``tags`` map, in place.

    Persisting via tags keeps per-record scorers off the dataset schema. Records without
    ``scorers`` are left untouched, so a dataset that never uses the feature is unchanged.
    """
    for record in record_dicts:
        if not isinstance(record, dict) or "scorers" not in record:
            continue

        scorer_names = validate_record_scorer_names(record.pop("scorers"))
        if not scorer_names:
            continue

        tags = record.get("tags")
        if tags is None or isinstance(tags, float):
            tags = {}
            record["tags"] = tags
        tags[DATASET_RECORD_SCORERS_TAG] = encode_record_scorers(scorer_names)


@dataclass
class DatasetRecord(_MlflowObject):
    """Represents a single record in an evaluation dataset.

    A DatasetRecord contains the input data, expected outputs (ground truth),
    and metadata for a single evaluation example. Records are immutable once
    created and are uniquely identified by their dataset_record_id.
    """

    dataset_id: str
    inputs: dict[str, Any]
    dataset_record_id: str
    created_time: int
    last_update_time: int
    outputs: dict[str, Any] | None = None
    expectations: dict[str, Any] | None = None
    tags: dict[str, str] | None = None
    source: DatasetRecordSource | None = None
    source_id: str | None = None
    source_type: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None

    def __post_init__(self):
        if self.inputs is None:
            raise ValueError("inputs must be provided")

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

    @property
    def scorers(self) -> list[str]:
        """Names of the scorers to run on this record alone, in addition to the run-level ones."""
        return decode_record_scorers((self.tags or {}).get(DATASET_RECORD_SCORERS_TAG))

    def to_proto(self) -> ProtoDatasetRecord:
        proto = ProtoDatasetRecord()

        proto.dataset_record_id = self.dataset_record_id
        proto.dataset_id = self.dataset_id
        proto.inputs = json.dumps(self.inputs)
        proto.created_time = self.created_time
        proto.last_update_time = self.last_update_time
        if self.outputs is not None:
            proto.outputs = json.dumps(self.outputs)
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
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by

        return proto

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecord) -> "DatasetRecord":
        inputs = json.loads(proto.inputs) if proto.HasField("inputs") else {}
        outputs = json.loads(proto.outputs) if proto.HasField("outputs") else None
        expectations = json.loads(proto.expectations) if proto.HasField("expectations") else None
        tags = json.loads(proto.tags) if proto.HasField("tags") else None

        source = None
        if proto.HasField("source"):
            source_dict = json.loads(proto.source)
            source = DatasetRecordSource.from_dict(source_dict)

        return cls(
            dataset_id=proto.dataset_id,
            inputs=inputs,
            dataset_record_id=proto.dataset_record_id,
            created_time=proto.created_time,
            last_update_time=proto.last_update_time,
            outputs=outputs,
            expectations=expectations,
            tags=tags,
            source=source,
            source_id=proto.source_id if proto.HasField("source_id") else None,
            source_type=DatasetRecordSourceType.from_proto(proto.source_type)
            if proto.HasField("source_type")
            else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d = MessageToDict(
            self.to_proto(),
            preserving_proto_field_name=True,
        )
        d["inputs"] = json.loads(d["inputs"])
        if "outputs" in d:
            d["outputs"] = json.loads(d["outputs"])
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
        # Validate required fields
        if "dataset_id" not in data:
            raise ValueError("dataset_id is required")
        if "dataset_record_id" not in data:
            raise ValueError("dataset_record_id is required")
        if "inputs" not in data:
            raise ValueError("inputs is required")
        if "created_time" not in data:
            raise ValueError("created_time is required")
        if "last_update_time" not in data:
            raise ValueError("last_update_time is required")

        source = None
        if data.get("source"):
            source = DatasetRecordSource.from_dict(data["source"])

        return cls(
            dataset_id=data["dataset_id"],
            inputs=data["inputs"],
            dataset_record_id=data["dataset_record_id"],
            created_time=data["created_time"],
            last_update_time=data["last_update_time"],
            outputs=data.get("outputs"),
            expectations=data.get("expectations"),
            tags=data.get("tags"),
            source=source,
            source_id=data.get("source_id"),
            source_type=data.get("source_type"),
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
            and self.outputs == other.outputs
            and self.expectations == other.expectations
            and self.tags == other.tags
            and self.source == other.source
            and self.source_id == other.source_id
            and self.source_type == other.source_type
        )

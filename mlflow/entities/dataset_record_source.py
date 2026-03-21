from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource


class DatasetRecordSourceType(str, Enum):
    """
    Enumeration for dataset record source types.

    Available source types:
        - UNSPECIFIED: Default when source type is not specified
        - TRACE: Record created from a trace/span
        - HUMAN: Record created from human annotation
        - DOCUMENT: Record created from a document
        - CODE: Record created from code/computation

    Example:
        Using enum values directly:

        .. code-block:: python

            from mlflow.entities import DatasetRecordSource, DatasetRecordSourceType

            # Direct enum usage
            source = DatasetRecordSource(
                source_type=DatasetRecordSourceType.TRACE, source_data={"trace_id": "trace123"}
            )

        String validation through instance creation:

        .. code-block:: python

            # String input - case insensitive
            source = DatasetRecordSource(
                source_type="trace",  # Will be standardized to "TRACE"
                source_data={"trace_id": "trace123"},
            )
    """

    UNSPECIFIED = "UNSPECIFIED"
    TRACE = "TRACE"
    HUMAN = "HUMAN"
    DOCUMENT = "DOCUMENT"
    CODE = "CODE"

    @staticmethod
    def _parse(source_type: str) -> str:
        source_type = source_type.upper()
        try:
            return DatasetRecordSourceType(source_type).value
        except ValueError:
            valid_types = [t.value for t in DatasetRecordSourceType]
            raise MlflowException(
                message=(
                    f"Invalid dataset record source type: {source_type}. "
                    f"Valid source types: {valid_types}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _standardize(source_type: str) -> "DatasetRecordSourceType":
        if isinstance(source_type, DatasetRecordSourceType):
            return source_type
        parsed = DatasetRecordSourceType._parse(source_type)
        return DatasetRecordSourceType(parsed)

    @classmethod
    def from_proto(cls, proto_source_type) -> str:
        return ProtoDatasetRecordSource.SourceType.Name(proto_source_type)


@dataclass
class DatasetRecordSource(_MlflowObject):
    """
    Source of a dataset record.

    Args:
        source_type: The type of the dataset record source. Must be one of the values in
            the DatasetRecordSourceType enum or a string that can be parsed to one.
        source_data: Additional source-specific data as a dictionary.
    """

    source_type: DatasetRecordSourceType
    source_data: dict[str, Any] | None = None

    def __post_init__(self):
        self.source_type = DatasetRecordSourceType._standardize(self.source_type)

        if self.source_data is None:
            self.source_data = {}

    def to_proto(self) -> ProtoDatasetRecordSource:
        proto = ProtoDatasetRecordSource()
        proto.source_type = ProtoDatasetRecordSource.SourceType.Value(self.source_type.value)
        if self.source_data:
            proto.source_data = json.dumps(self.source_data)
        return proto

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecordSource) -> "DatasetRecordSource":
        source_data = json.loads(proto.source_data) if proto.HasField("source_data") else {}
        source_type = (
            DatasetRecordSourceType.from_proto(proto.source_type)
            if proto.HasField("source_type")
            else None
        )

        return cls(source_type=source_type, source_data=source_data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecordSource":
        return cls(**data)

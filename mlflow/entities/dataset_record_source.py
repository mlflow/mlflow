from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.evaluation_datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource


class DatasetRecordSourceType:
    """
    Enumeration and validator for dataset record source types.

    This class provides constants for valid dataset record source types and handles validation
    and standardization of source type values. It supports both direct constant access and
    string validation.

    The class automatically handles:
    - Case-insensitive string inputs (converts to uppercase)
    - Validation of source type values

    Available source types:
        - SOURCE_TYPE_UNSPECIFIED: Default when source type is not specified
        - TRACE: Record created from a trace/span
        - HUMAN: Record created from human annotation
        - DOCUMENT: Record created from a document
        - CODE: Record created from code/computation

    Example:
        Using class constants directly:

        .. code-block:: python

            from mlflow.entities import DatasetRecordSource, DatasetRecordSourceType

            # Direct constant usage
            source = DatasetRecordSource(
                source_type=DatasetRecordSourceType.TRACE,
                source_data={"trace_id": "trace123"}
            )

        String validation through instance creation:

        .. code-block:: python

            # String input - case insensitive
            source = DatasetRecordSource(
                source_type="trace",  # Will be standardized to "TRACE"
                source_data={"trace_id": "trace123"}
            )
    """
    
    SOURCE_TYPE_UNSPECIFIED = "SOURCE_TYPE_UNSPECIFIED"
    TRACE = "TRACE"
    HUMAN = "HUMAN"
    DOCUMENT = "DOCUMENT"
    CODE = "CODE"
    _SOURCE_TYPES = [SOURCE_TYPE_UNSPECIFIED, TRACE, HUMAN, DOCUMENT, CODE]
    
    def __init__(self, source_type: str):
        self._source_type = DatasetRecordSourceType._parse(source_type)
    
    @staticmethod
    def _parse(source_type: str) -> str:
        source_type = source_type.upper()
        
        if source_type not in DatasetRecordSourceType._SOURCE_TYPES:
            raise MlflowException(
                message=(
                    f"Invalid dataset record source type: {source_type}. "
                    f"Valid source types: {DatasetRecordSourceType._SOURCE_TYPES}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return source_type
    
    def __str__(self):
        return self._source_type
    
    @staticmethod
    def _standardize(source_type: str) -> str:
        return str(DatasetRecordSourceType(source_type))
    
    @classmethod
    def from_proto(cls, proto_source_type) -> str:
        return ProtoDatasetRecordSource.SourceType.Name(proto_source_type)


@dataclass
class DatasetRecordSource(_MlflowObject):
    """
    Source of a dataset record (trace, human annotation, document, etc).
    
    Args:
        source_type: The type of the dataset record source. Must be one of the values in
            the DatasetRecordSourceType enum or an instance of the enumerator value.
        source_data: Additional source-specific data as a dictionary. For example:
            - For TRACE sources: {"trace_id": "...", "span_id": "..."}
            - For HUMAN sources: {"user_id": "...", "timestamp": "..."}
            - For DOCUMENT sources: {"doc_uri": "...", "content": "..."}
    """
    
    source_type: str
    source_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # Standardize source type using the same pattern as AssessmentSource
        self.source_type = DatasetRecordSourceType._standardize(self.source_type)
        
        # Initialize empty dict if None
        if self.source_data is None:
            self.source_data = {}
    
    def to_proto(self) -> ProtoDatasetRecordSource:
        """Convert to protobuf representation."""
        proto = ProtoDatasetRecordSource()
        proto.source_type = ProtoDatasetRecordSource.SourceType.Value(self.source_type)
        if self.source_data:
            proto.source_data = json.dumps(self.source_data)
        return proto
    
    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecordSource) -> "DatasetRecordSource":
        """Create instance from protobuf representation."""
        source_data = json.loads(proto.source_data) if proto.HasField("source_data") else {}
        source_type = DatasetRecordSourceType.from_proto(proto.source_type) if proto.HasField("source_type") else None
        
        return cls(
            source_type=source_type,
            source_data=source_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_type": self.source_type,
            "source_data": self.source_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecordSource":
        """Create instance from dictionary representation."""
        return cls(
            source_type=data.get("source_type"),
            source_data=data.get("source_data", {})
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on type and data."""
        if not isinstance(other, DatasetRecordSource):
            return False
        return self.source_type == other.source_type and self.source_data == other.source_data
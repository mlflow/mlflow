"""
Managed Datasets entities for MLflow GenAI evaluation.

This module provides the core entity classes for managed datasets functionality,
including ManagedDataset, DatasetRecord, and various source types. These entities
are designed to be compatible with the Databricks agents SDK interface while
providing a pure OSS implementation.
"""

from typing import List, Optional, Dict, Any, Union
import uuid
import json
import time
import hashlib

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.managed_datasets_pb2 import (
    ManagedDataset as ProtoManagedDataset,
    DatasetRecord as ProtoDatasetRecord,
    DatasetRecordSource as ProtoDatasetRecordSource,
    InputValue as ProtoInputValue,
    ExpectationValue as ProtoExpectationValue,
    HumanSource as ProtoHumanSource,
    DocumentSource as ProtoDocumentSource,
    TraceSource as ProtoTraceSource,
)
from google.protobuf.struct_pb2 import Value as ProtoValue


class InputValue(_MlflowObject):
    """Represents a single input field within a dataset record."""

    def __init__(self, key: str, value: Any) -> None:
        self._key = key
        self._value = value

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self) -> str:
        """The key/name of the input field (e.g., 'question', 'context', 'prompt')."""
        return self._key

    @property
    def value(self) -> Any:
        """The value of the input field."""
        return self._value

    def to_proto(self) -> ProtoInputValue:
        """Convert this entity to a protobuf message."""
        input_value = ProtoInputValue()
        input_value.key = self.key
        
        # Convert value to protobuf Value
        proto_value = ProtoValue()
        self._set_proto_value(proto_value, self.value)
        input_value.value.CopyFrom(proto_value)
        
        return input_value

    @classmethod
    def from_proto(cls, proto: ProtoInputValue) -> "InputValue":
        """Create an entity from a protobuf message."""
        value = cls._get_value_from_proto(proto.value)
        return cls(key=proto.key, value=value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputValue":
        """Create an entity from a dictionary."""
        return cls(key=data["key"], value=data["value"])

    @staticmethod
    def _set_proto_value(proto_value: ProtoValue, value: Any) -> None:
        """Set a protobuf Value from a Python value."""
        if value is None:
            proto_value.null_value = 0
        elif isinstance(value, bool):
            proto_value.bool_value = value
        elif isinstance(value, int):
            proto_value.number_value = float(value)
        elif isinstance(value, float):
            proto_value.number_value = value
        elif isinstance(value, str):
            proto_value.string_value = value
        elif isinstance(value, (list, tuple)):
            list_value = proto_value.list_value
            for item in value:
                item_value = list_value.values.add()
                InputValue._set_proto_value(item_value, item)
        elif isinstance(value, dict):
            struct_value = proto_value.struct_value
            for k, v in value.items():
                item_value = struct_value.fields[k]
                InputValue._set_proto_value(item_value, v)
        else:
            # Fallback to string representation
            proto_value.string_value = str(value)

    @staticmethod
    def _get_value_from_proto(proto_value: ProtoValue) -> Any:
        """Extract a Python value from a protobuf Value."""
        which = proto_value.WhichOneof("kind")
        if which == "null_value":
            return None
        elif which == "bool_value":
            return proto_value.bool_value
        elif which == "number_value":
            # Try to return as int if it's a whole number
            if proto_value.number_value.is_integer():
                return int(proto_value.number_value)
            return proto_value.number_value
        elif which == "string_value":
            return proto_value.string_value
        elif which == "list_value":
            return [InputValue._get_value_from_proto(item) for item in proto_value.list_value.values]
        elif which == "struct_value":
            return {k: InputValue._get_value_from_proto(v) for k, v in proto_value.struct_value.fields.items()}
        else:
            return None


class ExpectationValue(_MlflowObject):
    """Represents an expected output for evaluation scoring."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def value(self) -> Any:
        """The expected output value."""
        return self._value

    def to_proto(self) -> ProtoExpectationValue:
        """Convert this entity to a protobuf message."""
        expectation_value = ProtoExpectationValue()
        
        # Convert value to protobuf Value
        proto_value = ProtoValue()
        InputValue._set_proto_value(proto_value, self.value)
        expectation_value.value.CopyFrom(proto_value)
        
        return expectation_value

    @classmethod
    def from_proto(cls, proto: ProtoExpectationValue) -> "ExpectationValue":
        """Create an entity from a protobuf message."""
        value = InputValue._get_value_from_proto(proto.value)
        return cls(value=value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectationValue":
        """Create an entity from a dictionary."""
        return cls(value=data["value"])


class DatasetRecordSource(_MlflowObject):
    """
    Base class for dataset record sources indicating the origin of a dataset record.
    
    This is a union type that can represent human, document, or trace sources.
    """

    def __init__(self, source_type: str, source_data: Dict[str, Any]) -> None:
        self._source_type = source_type
        self._source_data = source_data

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def source_type(self) -> str:
        """The type of source ('human', 'document', or 'trace')."""
        return self._source_type

    @property
    def source_data(self) -> Dict[str, Any]:
        """The source-specific data."""
        return self._source_data

    def to_proto(self) -> ProtoDatasetRecordSource:
        """Convert this entity to a protobuf message."""
        source = ProtoDatasetRecordSource()
        
        if self.source_type == "human":
            human_source = ProtoHumanSource()
            if "user_id" in self.source_data:
                human_source.user_id = self.source_data["user_id"]
            source.human.CopyFrom(human_source)
        elif self.source_type == "document":
            doc_source = ProtoDocumentSource()
            if "doc_uri" in self.source_data:
                doc_source.doc_uri = self.source_data["doc_uri"]
            if "content" in self.source_data:
                doc_source.content = self.source_data["content"]
            source.document.CopyFrom(doc_source)
        elif self.source_type == "trace":
            trace_source = ProtoTraceSource()
            if "trace_id" in self.source_data:
                trace_source.trace_id = self.source_data["trace_id"]
            if "span_id" in self.source_data:
                trace_source.span_id = self.source_data["span_id"]
            source.trace.CopyFrom(trace_source)
        
        return source

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecordSource) -> "DatasetRecordSource":
        """Create an entity from a protobuf message."""
        which = proto.WhichOneof("source_type")
        
        if which == "human":
            source_data = {}
            if proto.human.HasField("user_id"):
                source_data["user_id"] = proto.human.user_id
            return cls("human", source_data)
        elif which == "document":
            source_data = {}
            if proto.document.HasField("doc_uri"):
                source_data["doc_uri"] = proto.document.doc_uri
            if proto.document.HasField("content"):
                source_data["content"] = proto.document.content
            return cls("document", source_data)
        elif which == "trace":
            source_data = {}
            if proto.trace.HasField("trace_id"):
                source_data["trace_id"] = proto.trace.trace_id
            if proto.trace.HasField("span_id"):
                source_data["span_id"] = proto.trace.span_id
            return cls("trace", source_data)
        else:
            raise ValueError(f"Unknown source type in proto: {which}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "source_type": self.source_type,
            "source_data": self.source_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecordSource":
        """Create an entity from a dictionary."""
        return cls(
            source_type=data["source_type"],
            source_data=data["source_data"],
        )


class HumanSource(DatasetRecordSource):
    """Records that were manually created or annotated by humans."""

    def __init__(self, user_id: str) -> None:
        super().__init__("human", {"user_id": user_id})

    @property
    def user_id(self) -> str:
        """Identifier for the human annotator (e.g., username, email, user_id)."""
        return self.source_data["user_id"]


class DocumentSource(DatasetRecordSource):
    """Records derived from processing documents or knowledge bases."""

    def __init__(self, doc_uri: str, content: Optional[str] = None) -> None:
        source_data = {"doc_uri": doc_uri}
        if content is not None:
            source_data["content"] = content
        super().__init__("document", source_data)

    @property
    def doc_uri(self) -> str:
        """URI or identifier of the source document."""
        return self.source_data["doc_uri"]

    @property
    def content(self) -> Optional[str]:
        """Optional document content or excerpt for reference."""
        return self.source_data.get("content")


class TraceSource(DatasetRecordSource):
    """
    Records derived from MLflow traces, typically from model inference
    or evaluation runs that can be converted into evaluation datasets.
    """

    def __init__(self, trace_id: str, span_id: Optional[str] = None) -> None:
        source_data = {"trace_id": trace_id}
        if span_id is not None:
            source_data["span_id"] = span_id
        super().__init__("trace", source_data)

    @property
    def trace_id(self) -> str:
        """ID of the trace from which this record was derived."""
        return self.source_data["trace_id"]

    @property
    def span_id(self) -> Optional[str]:
        """Optional: specific span ID within the trace that generated this record."""
        return self.source_data.get("span_id")


class DatasetRecord(_MlflowObject):
    """
    Dataset Record entity representing a single evaluation record within a managed dataset.
    
    Contains inputs, expected outputs, and metadata for evaluation.
    """

    def __init__(
        self,
        dataset_record_id: str,
        dataset_id: str,
        inputs: List[InputValue],
        expectations: Dict[str, ExpectationValue],
        tags: Optional[Dict[str, str]] = None,
        source: Optional[DatasetRecordSource] = None,
        created_time: Optional[int] = None,
        last_update_time: Optional[int] = None,
        created_by: Optional[str] = None,
        last_updated_by: Optional[str] = None,
    ) -> None:
        self._dataset_record_id = dataset_record_id
        self._dataset_id = dataset_id
        self._inputs = inputs or []
        self._expectations = expectations or {}
        self._tags = tags or {}
        self._source = source
        # Store timestamps as milliseconds to match MLflow patterns
        current_time = int(time.time() * 1000)
        self._created_time = created_time or current_time
        self._last_update_time = last_update_time or current_time
        self._created_by = created_by
        self._last_updated_by = last_updated_by

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_record_id(self) -> str:
        """Unique identifier for the dataset record."""
        return self._dataset_record_id

    @property
    def dataset_id(self) -> str:
        """The dataset ID this record belongs to."""
        return self._dataset_id

    @property
    def inputs(self) -> List[InputValue]:
        """Input values for the evaluation record (e.g., questions, prompts, context)."""
        return self._inputs

    @property
    def expectations(self) -> Dict[str, ExpectationValue]:
        """Expected outputs/answers for the given inputs, used for evaluation scoring."""
        return self._expectations

    @property
    def tags(self) -> Dict[str, str]:
        """User-defined tags for categorizing and filtering records."""
        return self._tags

    @property
    def source(self) -> Optional[DatasetRecordSource]:
        """Source information indicating how this record was created."""
        return self._source

    @property
    def created_time(self) -> int:
        """Unix timestamp of when the record was created in milliseconds."""
        return self._created_time

    @property
    def last_update_time(self) -> int:
        """Unix timestamp of when the record was last updated in milliseconds."""
        return self._last_update_time

    @property
    def created_by(self) -> Optional[str]:
        """User who created the record."""
        return self._created_by

    @property
    def last_updated_by(self) -> Optional[str]:
        """User who last updated the record."""
        return self._last_updated_by

    def add_input(self, key: str, value: Any) -> None:
        """Add an input field to this record."""
        self._inputs.append(InputValue(key, value))
        self._last_update_time = int(time.time() * 1000)

    def add_expectation(self, key: str, value: Any) -> None:
        """Add an expected output for this record."""
        self._expectations[key] = ExpectationValue(value)
        self._last_update_time = int(time.time() * 1000)

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to this record."""
        self._tags[key] = value
        self._last_update_time = int(time.time() * 1000)

    def get_input_value(self, key: str) -> Any:
        """Get the value of an input field by key."""
        for input_val in self._inputs:
            if input_val.key == key:
                return input_val.value
        return None

    def get_expectation_value(self, key: str) -> Any:
        """Get the value of an expectation by key."""
        expectation = self._expectations.get(key)
        return expectation.value if expectation else None

    def to_proto(self) -> ProtoDatasetRecord:
        """Convert this entity to a protobuf message."""
        record = ProtoDatasetRecord()
        record.dataset_record_id = self.dataset_record_id
        record.dataset_id = self.dataset_id
        
        # Add inputs
        for input_val in self.inputs:
            record.inputs.append(input_val.to_proto())
        
        # Add expectations
        for key, expectation in self.expectations.items():
            record.expectations[key].CopyFrom(expectation.to_proto())
        
        # Add tags
        for key, value in self.tags.items():
            record.tags[key] = value
        
        # Add source if present
        if self.source:
            record.source.CopyFrom(self.source.to_proto())
        
        # Add metadata
        if self.created_by:
            record.created_by = self.created_by
        if self.last_updated_by:
            record.last_updated_by = self.last_updated_by
        
        # Set timestamps (convert from milliseconds)
        if self.created_time:
            record.created_time.FromMilliseconds(self.created_time)
        if self.last_update_time:
            record.last_update_time.FromMilliseconds(self.last_update_time)
        
        return record

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecord) -> "DatasetRecord":
        """Create an entity from a protobuf message."""
        inputs = [InputValue.from_proto(input_proto) for input_proto in proto.inputs]
        expectations = {
            key: ExpectationValue.from_proto(exp_proto) 
            for key, exp_proto in proto.expectations.items()
        }
        tags = dict(proto.tags)
        
        source = None
        if proto.HasField("source"):
            source = DatasetRecordSource.from_proto(proto.source)
        
        created_time = None
        if proto.HasField("created_time"):
            created_time = proto.created_time.ToMilliseconds()
        
        last_update_time = None
        if proto.HasField("last_update_time"):
            last_update_time = proto.last_update_time.ToMilliseconds()
        
        return cls(
            dataset_record_id=proto.dataset_record_id,
            dataset_id=proto.dataset_id,
            inputs=inputs,
            expectations=expectations,
            tags=tags,
            source=source,
            created_time=created_time,
            last_update_time=last_update_time,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "dataset_record_id": self.dataset_record_id,
            "dataset_id": self.dataset_id,
            "inputs": [input_val.to_dict() for input_val in self.inputs],
            "expectations": {key: exp.to_dict() for key, exp in self.expectations.items()},
            "tags": self.tags,
            "source": self.source.to_dict() if self.source else None,
            "created_time": self.created_time,
            "last_update_time": self.last_update_time,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        """Create an entity from a dictionary."""
        inputs = [InputValue.from_dict(input_data) for input_data in data.get("inputs", [])]
        expectations = {
            key: ExpectationValue.from_dict(exp_data)
            for key, exp_data in data.get("expectations", {}).items()
        }
        
        source = None
        if data.get("source"):
            source = DatasetRecordSource.from_dict(data["source"])
        
        created_time = data.get("created_time")
        last_update_time = data.get("last_update_time")
        
        return cls(
            dataset_record_id=data["dataset_record_id"],
            dataset_id=data["dataset_id"],
            inputs=inputs,
            expectations=expectations,
            tags=data.get("tags", {}),
            source=source,
            created_time=created_time,
            last_update_time=last_update_time,
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
        )

    @classmethod
    def create_new(
        cls,
        dataset_id: str,
        inputs: Dict[str, Any],
        expectations: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        source: Optional[DatasetRecordSource] = None,
        created_by: Optional[str] = None,
    ) -> "DatasetRecord":
        """Create a new dataset record with a generated ID."""
        record_id = str(uuid.uuid4())
        
        input_values = [InputValue(key, value) for key, value in inputs.items()]
        expectation_values = {key: ExpectationValue(value) for key, value in expectations.items()}
        
        return cls(
            dataset_record_id=record_id,
            dataset_id=dataset_id,
            inputs=input_values,
            expectations=expectation_values,
            tags=tags or {},
            source=source,
            created_by=created_by,
            last_updated_by=created_by,
        )


class ManagedDataset(_MlflowObject):
    """
    Managed Dataset entity for storing GenAI evaluation records.
    
    Represents a collection of evaluation records (inputs and expectations)
    for GenAI model evaluation and training. Provides capabilities to merge 
    records from various sources including traces, human annotations, and documents.
    
    This implementation is compatible with the Databricks agents SDK interface.
    """

    def __init__(
        self,
        dataset_id: str,
        name: str,
        source: Optional[str] = None,
        source_type: Optional[str] = None,
        schema: Optional[str] = None,
        profile: Optional[str] = None,
        digest: Optional[str] = None,
        created_time: Optional[int] = None,
        last_update_time: Optional[int] = None,
        created_by: Optional[str] = None,
        last_updated_by: Optional[str] = None,
        experiment_ids: Optional[List[str]] = None,
        records: Optional[List[DatasetRecord]] = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._name = name
        self._source = source
        self._source_type = source_type
        self._schema = schema
        self._profile = profile
        self._digest = digest
        # Store timestamps as milliseconds to match MLflow patterns
        current_time = int(time.time() * 1000)
        self._created_time = created_time or current_time
        self._last_update_time = last_update_time or current_time
        self._created_by = created_by
        self._last_updated_by = last_updated_by
        self._experiment_ids = experiment_ids or []
        self._records = records or []

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def dataset_id(self) -> str:
        """Unique identifier for the dataset."""
        return self._dataset_id

    @property
    def name(self) -> str:
        """Human readable name that identifies the dataset."""
        return self._name

    @property
    def source(self) -> Optional[str]:
        """Source information for the dataset, e.g. table name, file path, or trace ID."""
        return self._source

    @property
    def source_type(self) -> Optional[str]:
        """The type of the dataset source, e.g. 'databricks-uc-table', 'trace', 'human', 'document'."""
        return self._source_type

    @property
    def schema(self) -> Optional[str]:
        """The schema of the dataset in JSON format."""
        return self._schema

    @property
    def profile(self) -> Optional[str]:
        """The profile of the dataset containing summary statistics and metadata."""
        return self._profile

    @property
    def digest(self) -> Optional[str]:
        """String digest (hash) that uniquely identifies the dataset content."""
        return self._digest

    @property
    def created_time(self) -> int:
        """Unix timestamp of when the dataset was created in milliseconds."""
        return self._created_time

    @property
    def last_update_time(self) -> int:
        """Unix timestamp of when the dataset was last updated in milliseconds."""
        return self._last_update_time

    @property
    def created_by(self) -> Optional[str]:
        """User who created the dataset."""
        return self._created_by

    @property
    def last_updated_by(self) -> Optional[str]:
        """User who last updated the dataset."""
        return self._last_updated_by

    @property
    def experiment_ids(self) -> List[str]:
        """List of experiment IDs associated with this dataset."""
        return self._experiment_ids

    @property
    def records(self) -> List[DatasetRecord]:
        """List of dataset records contained in this dataset."""
        return self._records

    # Core methods matching Databricks interface
    def to_df(self) -> "pd.DataFrame":
        """
        Convert the dataset records to a pandas DataFrame for analysis.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to convert to DataFrame")
        
        if not self._records:
            return pd.DataFrame()
        
        # Convert records to flat dictionary format
        rows = []
        for record in self._records:
            row = {"dataset_record_id": record.dataset_record_id}
            
            # Add inputs as columns
            for inp in record.inputs:
                row[f"input_{inp.key}"] = inp.value
            
            # Add expectations as columns
            for key, exp_val in record.expectations.items():
                row[f"expected_{key}"] = exp_val.value if hasattr(exp_val, 'value') else exp_val
            
            # Add tags as columns
            for key, value in record.tags.items():
                row[f"tag_{key}"] = value
            
            # Add metadata
            row.update({
                "created_time": record.created_time,
                "created_by": record.created_by,
                "source_type": record.source.source_type if record.source else None,
            })
            
            rows.append(row)
        
        return pd.DataFrame(rows)

    def set_profile(self, profile: str) -> "ManagedDataset":
        """
        Set the profile data containing summary statistics and metadata.
        
        Returns a new ManagedDataset instance (immutable pattern matching Databricks).
        """
        return ManagedDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            source=self.source,
            source_type=self.source_type,
            schema=self.schema,
            profile=profile,
            digest=self.digest,
            created_time=self.created_time,
            last_update_time=int(time.time() * 1000),
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
            experiment_ids=self.experiment_ids,
            records=self.records,
        )

    def merge_records(self, records: Union[List[Dict[str, Any]], "pd.DataFrame"]) -> "ManagedDataset":
        """
        Merge records from another source, handling deduplication based on record content.
        
        Returns a new ManagedDataset instance (immutable pattern matching Databricks).
        
        Args:
            records: Records to merge - can be list of dicts or pandas DataFrame
        """
        import pandas as pd
        
        # Convert input to standardized format
        if isinstance(records, pd.DataFrame):
            records_list = records.to_dict('records')
        else:
            records_list = records
        
        # Convert to DatasetRecord objects
        new_records = []
        for record_data in records_list:
            # Extract inputs (fields starting with 'input_' or provided directly)
            inputs = {}
            expectations = {}
            tags = {}
            
            for key, value in record_data.items():
                if key.startswith('input_'):
                    inputs[key[6:]] = value  # Remove 'input_' prefix
                elif key.startswith('expected_'):
                    expectations[key[9:]] = value  # Remove 'expected_' prefix
                elif key.startswith('tag_'):
                    tags[key[4:]] = value  # Remove 'tag_' prefix
                elif key in ['inputs', 'expectations', 'tags']:
                    # Direct field access
                    if key == 'inputs':
                        inputs.update(value)
                    elif key == 'expectations':
                        expectations.update(value)
                    elif key == 'tags':
                        tags.update(value)
            
            # Create new record
            new_record = DatasetRecord.create_new(
                dataset_id=self.dataset_id,
                inputs=inputs,
                expectations=expectations,
                tags=tags,
                created_by=self.last_updated_by,
            )
            new_records.append(new_record)
        
        # Merge with existing records (deduplication)
        merged_records = list(self._records)  # Copy existing records
        existing_hashes = {self._compute_record_hash(record) for record in merged_records}
        
        for record in new_records:
            record_hash = self._compute_record_hash(record)
            if record_hash not in existing_hashes:
                merged_records.append(record)
                existing_hashes.add(record_hash)
        
        # Return new instance
        return ManagedDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            source=self.source,
            source_type=self.source_type,
            schema=self.schema,
            profile=self.profile,
            digest=self.digest,
            created_time=self.created_time,
            last_update_time=int(time.time() * 1000),
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
            experiment_ids=self.experiment_ids,
            records=merged_records,
        )

    def _compute_record_hash(self, record: DatasetRecord) -> str:
        """Compute a hash for a dataset record based on its content."""
        # Create a deterministic representation of the record
        content = {
            "inputs": {inp.key: str(inp.value) for inp in record.inputs},
            "expectations": {k: str(v.value) for k, v in record.expectations.items()},
            "tags": record.tags,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    # MLflow entity methods
    def to_proto(self) -> ProtoManagedDataset:
        """Convert this entity to a protobuf message."""
        dataset = ProtoManagedDataset()
        dataset.dataset_id = self.dataset_id
        dataset.name = self.name
        
        if self.source:
            dataset.source = self.source
        if self.source_type:
            dataset.source_type = self.source_type
        if self.schema:
            dataset.schema = self.schema
        if self.profile:
            dataset.profile = self.profile
        if self.digest:
            dataset.digest = self.digest
        if self.created_by:
            dataset.created_by = self.created_by
        if self.last_updated_by:
            dataset.last_updated_by = self.last_updated_by
        
        dataset.experiment_ids.extend(self.experiment_ids)
        
        # Add all records for bulk operations
        for record in self.records:
            dataset.records.append(record.to_proto())
        
        # Set timestamps (convert from milliseconds)
        if self.created_time:
            dataset.created_time.FromMilliseconds(self.created_time)
        if self.last_update_time:
            dataset.last_update_time.FromMilliseconds(self.last_update_time)
        
        return dataset

    @classmethod
    def from_proto(cls, proto: ProtoManagedDataset) -> "ManagedDataset":
        """Create an entity from a protobuf message."""
        created_time = None
        if proto.HasField("created_time"):
            created_time = proto.created_time.ToMilliseconds()
        
        last_update_time = None
        if proto.HasField("last_update_time"):
            last_update_time = proto.last_update_time.ToMilliseconds()
        
        # Convert records from protobuf
        records = []
        for record_proto in proto.records:
            records.append(DatasetRecord.from_proto(record_proto))
        
        return cls(
            dataset_id=proto.dataset_id,
            name=proto.name,
            source=proto.source if proto.HasField("source") else None,
            source_type=proto.source_type if proto.HasField("source_type") else None,
            schema=proto.schema if proto.HasField("schema") else None,
            profile=proto.profile if proto.HasField("profile") else None,
            digest=proto.digest if proto.HasField("digest") else None,
            created_time=created_time,
            last_update_time=last_update_time,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
            experiment_ids=list(proto.experiment_ids),
            records=records,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "source_type": self.source_type,
            "schema": self.schema,
            "profile": self.profile,
            "digest": self.digest,
            "created_time": self.created_time,
            "last_update_time": self.last_update_time,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "experiment_ids": self.experiment_ids,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManagedDataset":
        """Create an entity from a dictionary."""
        created_time = data.get("created_time")
        last_update_time = data.get("last_update_time")
        
        records = []
        if data.get("records"):
            records = [DatasetRecord.from_dict(record_data) for record_data in data["records"]]
        
        return cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            source=data.get("source"),
            source_type=data.get("source_type"),
            schema=data.get("schema"),
            profile=data.get("profile"),
            digest=data.get("digest"),
            created_time=created_time,
            last_update_time=last_update_time,
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            experiment_ids=data.get("experiment_ids", []),
            records=records,
        )

    @classmethod
    def create_new(
        cls,
        name: str,
        experiment_ids: List[str],
        source_type: Optional[str] = None,
        source: Optional[str] = None,
        digest: Optional[str] = None,
        schema: Optional[str] = None,
        profile: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> "ManagedDataset":
        """Create a new managed dataset with a generated ID."""
        dataset_id = str(uuid.uuid4())
        
        return cls(
            dataset_id=dataset_id,
            name=name,
            source=source,
            source_type=source_type,
            schema=schema,
            profile=profile,
            digest=digest,
            created_by=created_by,
            last_updated_by=created_by,
            experiment_ids=experiment_ids,
        )


# Factory functions for creating sources
def create_human_source(user_id: str) -> HumanSource:
    """Create a human source for manually created records."""
    return HumanSource(user_id)


def create_document_source(doc_uri: str, content: Optional[str] = None) -> DocumentSource:
    """Create a document source for records derived from documents."""
    return DocumentSource(doc_uri, content)


def create_trace_source(trace_id: str, span_id: Optional[str] = None) -> TraceSource:
    """Create a trace source for records derived from MLflow traces."""
    return TraceSource(trace_id, span_id)


# Utility functions
def get_source_summary(source: DatasetRecordSource) -> str:
    """Get a human-readable summary of the source."""
    if isinstance(source, HumanSource):
        return f"Human annotator: {source.user_id}"
    elif isinstance(source, DocumentSource):
        if source.content:
            content_preview = source.content[:50] + "..." if len(source.content) > 50 else source.content
            return f"Document: {source.doc_uri} (content: {content_preview})"
        else:
            return f"Document: {source.doc_uri}"
    elif isinstance(source, TraceSource):
        if source.span_id:
            return f"Trace: {source.trace_id}, Span: {source.span_id}"
        else:
            return f"Trace: {source.trace_id}"
    else:
        return f"Source: {source.source_type}"
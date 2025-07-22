"""
Managed Datasets entities for MLflow GenAI evaluation.

This module provides the core entity classes for managed datasets functionality,
including ManagedDataset, DatasetRecord, and various source types.
"""

import hashlib
import json
import time
import uuid
from typing import Any, Optional, Union

try:
    import pandas as pd
except ImportError:
    pd = None


from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.managed_datasets_pb2 import (
    DatasetRecord as ProtoDatasetRecord,
)
from mlflow.protos.managed_datasets_pb2 import (
    DatasetRecordSource as ProtoDatasetRecordSource,
)
from mlflow.protos.managed_datasets_pb2 import (
    DocumentSource as ProtoDocumentSource,
)
from mlflow.protos.managed_datasets_pb2 import (
    ExpectationValue as ProtoExpectationValue,
)
from mlflow.protos.managed_datasets_pb2 import (
    HumanSource as ProtoHumanSource,
)
from mlflow.protos.managed_datasets_pb2 import (
    InputValue as ProtoInputValue,
)
from mlflow.protos.managed_datasets_pb2 import (
    ManagedDataset as ProtoManagedDataset,
)
from mlflow.protos.managed_datasets_pb2 import (
    TraceSource as ProtoTraceSource,
)


class InputValue(_MlflowObject):
    """Represents a single input field within a dataset record."""

    def __init__(self, key: str, value: Any) -> None:
        if key is None:
            raise ValueError("InputValue key cannot be None")
        self._key = key
        self._value = str(value) if value is not None else ""

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> str:
        return self._value

    def to_proto(self) -> ProtoInputValue:
        input_value = ProtoInputValue()
        input_value.key = self.key
        input_value.value = self.value
        return input_value

    @classmethod
    def from_proto(cls, proto: ProtoInputValue) -> "InputValue":
        instance = cls.__new__(cls)
        instance._key = proto.key
        instance._value = proto.value
        return instance

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputValue":
        return cls(key=data["key"], value=data["value"])

    def __repr__(self) -> str:
        return f"<InputValue: key='{self.key}', value='{self.value}'>"


class ExpectationValue(_MlflowObject):
    """Represents an expected output for evaluation scoring."""

    def __init__(self, value: Any) -> None:
        self._value = str(value) if value is not None else ""

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def value(self) -> str:
        return self._value

    def to_proto(self) -> ProtoExpectationValue:
        expectation_value = ProtoExpectationValue()
        expectation_value.value = self._value
        return expectation_value

    @classmethod
    def from_proto(cls, proto: ProtoExpectationValue) -> "ExpectationValue":
        return cls(value=proto.value)

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectationValue":
        return cls(value=data["value"])


class DatasetRecordSource(_MlflowObject):
    """
    Base class for dataset record sources indicating the origin of a dataset record.

    This is a union type that can represent human, document, or trace sources.
    """

    def __init__(self, source_type: str, source_data: dict[str, Any]) -> None:
        self._source_type = source_type
        self._source_data = source_data

    def __eq__(self, other: "_MlflowObject") -> bool:
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_data(self) -> dict[str, Any]:
        return self._source_data

    def to_proto(self) -> ProtoDatasetRecordSource:
        source = ProtoDatasetRecordSource()

        if self.source_type == "human":
            human_source = ProtoHumanSource()
            if user_id := self.source_data.get("user_id"):
                human_source.user_id = user_id
            source.human.CopyFrom(human_source)
        elif self.source_type == "document":
            doc_source = ProtoDocumentSource()
            if doc_uri := self.source_data.get("doc_uri"):
                doc_source.doc_uri = doc_uri
            if content := self.source_data.get("content"):
                doc_source.content = content
            source.document.CopyFrom(doc_source)
        elif self.source_type == "trace":
            trace_source = ProtoTraceSource()
            if trace_id := self.source_data.get("trace_id"):
                trace_source.trace_id = trace_id
            if span_id := self.source_data.get("span_id"):
                trace_source.span_id = span_id
            source.trace.CopyFrom(trace_source)

        return source

    @classmethod
    def from_proto(cls, proto: ProtoDatasetRecordSource) -> "DatasetRecordSource":
        which = proto.WhichOneof("source_type")

        if which == "human":
            source_data = {}
            if proto.human.HasField("user_id"):
                source_data["user_id"] = proto.human.user_id
            return HumanSource(source_data["user_id"])
        elif which == "document":
            source_data = {}
            if proto.document.HasField("doc_uri"):
                source_data["doc_uri"] = proto.document.doc_uri
            if proto.document.HasField("content"):
                source_data["content"] = proto.document.content
            return DocumentSource(source_data["doc_uri"], source_data.get("content"))
        elif which == "trace":
            source_data = {}
            if proto.trace.HasField("trace_id"):
                source_data["trace_id"] = proto.trace.trace_id
            if proto.trace.HasField("span_id"):
                source_data["span_id"] = proto.trace.span_id
            return TraceSource(source_data["trace_id"], source_data.get("span_id"))
        else:
            raise ValueError(f"Unknown source type in proto: {which}")

    def to_dict(self) -> dict[str, Any]:
        """Convert this entity to a dictionary."""
        return {
            "source_type": self.source_type,
            "source_data": self.source_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecordSource":
        source_type = data["source_type"]
        source_data = data["source_data"]

        if source_type == "human":
            return HumanSource(source_data["user_id"])
        elif source_type == "document":
            return DocumentSource(source_data["doc_uri"], source_data.get("content"))
        elif source_type == "trace":
            return TraceSource(source_data["trace_id"], source_data.get("span_id"))
        else:
            return cls(source_type, source_data)


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
        return self.source_data["doc_uri"]

    @property
    def content(self) -> Optional[str]:
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
        return self.source_data["trace_id"]

    @property
    def span_id(self) -> Optional[str]:
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
        inputs: list[InputValue],
        expectations: dict[str, ExpectationValue],
        tags: Optional[dict[str, str]] = None,
        source: Optional[DatasetRecordSource] = None,
        created_time: Optional[int] = None,
        last_update_time: Optional[int] = None,
        created_by: Optional[str] = None,
        last_updated_by: Optional[str] = None,
    ) -> None:
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list of InputValue objects")
        if not isinstance(expectations, dict):
            raise TypeError("expectations must be a dictionary")

        self._dataset_record_id = dataset_record_id
        self._dataset_id = dataset_id
        self._inputs = inputs or []
        self._expectations = expectations or {}
        self._tags = tags or {}
        self._source = source
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
    def inputs(self) -> list[InputValue]:
        """Input values for the evaluation record (e.g., questions, prompts, context)."""
        return self._inputs

    @property
    def expectations(self) -> dict[str, ExpectationValue]:
        """Expected outputs/answers for the given inputs, used for evaluation scoring."""
        return self._expectations

    @property
    def tags(self) -> dict[str, str]:
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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        """Create an entity from a dictionary."""
        inputs = [InputValue.from_dict(input_data) for input_data in data.get("inputs", [])]

        expectations_data = data.get("expectations", {})
        expectations = {
            key: ExpectationValue.from_dict(exp_data) for key, exp_data in expectations_data.items()
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
        inputs: dict[str, Any],
        expectations: dict[str, Any],
        tags: Optional[dict[str, str]] = None,
        source: Optional[DatasetRecordSource] = None,
        created_by: Optional[str] = None,
    ) -> "DatasetRecord":
        """Create a new dataset record with a generated ID."""
        record_id = uuid.uuid4().hex

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
        experiment_ids: Optional[list[str]] = None,
        records: Optional[list[DatasetRecord]] = None,
    ) -> None:
        if experiment_ids is not None and not isinstance(experiment_ids, list):
            raise TypeError("experiment_ids must be a list of strings")
        if records is not None and not isinstance(records, list):
            raise TypeError("records must be a list of DatasetRecord objects")

        self._dataset_id = dataset_id
        self._name = name
        self._source = source
        self._source_type = source_type
        self._schema = schema
        self._profile = profile
        self._digest = digest
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
        """The type of the dataset source, e.g. 'trace', 'human', 'document'."""
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
    def last_updated_time(self) -> int:
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
    def experiment_ids(self) -> list[str]:
        """List of experiment IDs associated with this dataset."""
        return self._experiment_ids

    @property
    def records(self) -> list[DatasetRecord]:
        """List of dataset records contained in this dataset."""
        return self._records

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

        rows = []
        for record in self._records:
            record_dict = record.to_dict()

            inputs_dict = {
                input_data["key"]: input_data["value"] for input_data in record_dict["inputs"]
            }

            expectations_dict = {
                key: exp_data["value"] for key, exp_data in record_dict["expectations"].items()
            }

            row = {
                "dataset_record_id": record_dict["dataset_record_id"],
                "inputs": inputs_dict,
                "expectations": expectations_dict,
                "tags": record_dict["tags"],
                "source": record_dict["source"],
                "created_time": record_dict["created_time"],
                "last_update_time": record_dict["last_update_time"],
                "created_by": record_dict["created_by"],
                "last_updated_by": record_dict["last_updated_by"],
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def set_profile(self, profile: str) -> "ManagedDataset":
        """
        Set the profile data containing summary statistics and metadata.

        Returns a new ManagedDataset instance.
        """
        time.sleep(0.001)
        new_timestamp = int(time.time() * 1000)

        return ManagedDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            source=self.source,
            source_type=self.source_type,
            schema=self.schema,
            profile=profile,
            digest=self.digest,
            created_time=self.created_time,
            last_update_time=new_timestamp,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
            experiment_ids=self.experiment_ids,
            records=self.records,
        )

    def merge_records(
        self, records: Union[list[dict[str, Any]], list["DatasetRecord"], "pd.DataFrame"]
    ) -> "ManagedDataset":
        """
        Merge records from another source, handling deduplication based on record content.

        Returns a new ManagedDataset instance.

        Args:
            records: Records to merge - can be list of dicts, list of
                DatasetRecord objects, or pandas DataFrame
        """
        import pandas as pd

        if not isinstance(records, (list, pd.DataFrame)):
            raise TypeError(
                "records must be a list, list of DatasetRecord objects, or pandas DataFrame"
            )

        records_list = records.to_dict("records") if isinstance(records, pd.DataFrame) else records

        new_records = []
        for record_data in records_list:
            # If it's already a DatasetRecord, use it directly
            if isinstance(record_data, DatasetRecord):
                new_records.append(record_data)
                continue

            inputs = record_data.get("inputs", {})
            expectations = record_data.get("expectations", {})
            tags = record_data.get("tags", {})

            new_record = DatasetRecord.create_new(
                dataset_id=self.dataset_id,
                inputs=inputs,
                expectations=expectations,
                tags=tags,
                created_by=self.last_updated_by,
            )
            new_records.append(new_record)

        # Merge with existing records (deduplication)
        merged_records = list(self._records)
        existing_hashes = {self._compute_record_hash(record) for record in merged_records}

        for record in new_records:
            record_hash = self._compute_record_hash(record)
            if record_hash not in existing_hashes:
                merged_records.append(record)
                existing_hashes.add(record_hash)

        time.sleep(0.001)
        new_timestamp = int(time.time() * 1000)

        return ManagedDataset(
            dataset_id=self.dataset_id,
            name=self.name,
            source=self.source,
            source_type=self.source_type,
            schema=self.schema,
            profile=self.profile,
            digest=self.digest,
            created_time=self.created_time,
            last_update_time=new_timestamp,
            created_by=self.created_by,
            last_updated_by=self.last_updated_by,
            experiment_ids=self.experiment_ids,
            records=merged_records,
        )

    def _compute_record_hash(self, record: DatasetRecord) -> str:
        """Compute a hash for a dataset record based on its content."""
        content = {
            "inputs": {inp.key: str(inp.value) for inp in record.inputs},
            "expectations": {k: str(v.value) for k, v in record.expectations.items()},
            "tags": record.tags,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

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

    def to_dict(self) -> dict[str, Any]:
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
            "last_updated_time": self.last_updated_time,
            "created_by": self.created_by,
            "last_updated_by": self.last_updated_by,
            "experiment_ids": self.experiment_ids,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagedDataset":
        """Create an entity from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        if "dataset_id" not in data:
            raise KeyError("dataset_id is required")
        if "name" not in data:
            raise KeyError("name is required")

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
        experiment_ids: list[str],
        source_type: Optional[str] = None,
        source: Optional[str] = None,
        digest: Optional[str] = None,
        schema: Optional[str] = None,
        profile: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> "ManagedDataset":
        """Create a new managed dataset with a generated ID."""
        dataset_id = uuid.uuid4().hex

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

    @classmethod
    def from_df(
        cls,
        df: "pd.DataFrame",
        name: str,
        experiment_ids: list[str],
        input_columns: list[str],
        expectation_columns: list[str],
        tag_columns: Optional[list[str]] = None,
        source_type: Optional[str] = None,
        source: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> "ManagedDataset":
        """Create a managed dataset from a pandas DataFrame."""
        if pd is None:
            raise ImportError("pandas is required to create dataset from DataFrame")

        dataset = cls.create_new(
            name=name,
            experiment_ids=experiment_ids,
            source_type=source_type,
            source=source,
            created_by=created_by,
        )

        records = []
        for _, row in df.iterrows():
            inputs = {col: row[col] for col in input_columns if col in row.index}
            expectations = {col: row[col] for col in expectation_columns if col in row.index}
            tags = {}
            if tag_columns:
                tags = {col: str(row[col]) for col in tag_columns if col in row.index}

            record = DatasetRecord.create_new(
                dataset_id=dataset.dataset_id,
                inputs=inputs,
                expectations=expectations,
                tags=tags,
                created_by=created_by,
            )
            records.append(record)

        return cls(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            source=dataset.source,
            source_type=dataset.source_type,
            schema=dataset.schema,
            profile=dataset.profile,
            digest=dataset.digest,
            created_time=dataset.created_time,
            last_update_time=dataset.last_update_time,
            created_by=dataset.created_by,
            last_updated_by=dataset.last_updated_by,
            experiment_ids=dataset.experiment_ids,
            records=records,
        )


def create_human_source(user_id: str) -> HumanSource:
    """Create a human source for manually created records."""
    return HumanSource(user_id)


def create_document_source(doc_uri: str, content: Optional[str] = None) -> DocumentSource:
    """Create a document source for records derived from documents."""
    return DocumentSource(doc_uri, content)


def create_trace_source(trace_id: str, span_id: Optional[str] = None) -> TraceSource:
    """Create a trace source for records derived from MLflow traces."""
    return TraceSource(trace_id, span_id)


def get_source_summary(source: DatasetRecordSource) -> str:
    """Get a human-readable summary of the source."""
    if isinstance(source, HumanSource):
        return f"Human annotator: {source.user_id}"
    elif isinstance(source, DocumentSource):
        if source.content:
            content_preview = (
                source.content[:50] + "..." if len(source.content) > 50 else source.content
            )
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

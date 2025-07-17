from typing import List, Optional, Dict, Any
import uuid
import time

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.managed_datasets_pb2 import (
    DatasetRecord as ProtoDatasetRecord,
    InputValue as ProtoInputValue,
    ExpectationValue as ProtoExpectationValue,
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
        source: Optional["DatasetRecordSource"] = None,
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
    def source(self) -> Optional["DatasetRecordSource"]:
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
        # Import here to avoid circular imports
        from mlflow.entities.dataset_source import DatasetRecordSource
        
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
        # Import here to avoid circular imports
        from mlflow.entities.dataset_source import DatasetRecordSource
        
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
        source: Optional["DatasetRecordSource"] = None,
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

    def update_metadata(
        self,
        tags: Optional[Dict[str, str]] = None,
        last_updated_by: Optional[str] = None,
    ) -> None:
        """Update record metadata fields."""
        if tags is not None:
            self._tags.update(tags)
        if last_updated_by is not None:
            self._last_updated_by = last_updated_by
        
        self._last_update_time = int(time.time() * 1000)
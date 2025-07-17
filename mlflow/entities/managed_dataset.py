from typing import List, Optional, Dict, Any
import uuid
import json
import time

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.managed_datasets_pb2 import (
    ManagedDataset as ProtoManagedDataset,
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
        experiment_ids: Optional[List[str]] = None,
        records: Optional[List["DatasetRecord"]] = None,
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
    def records(self) -> List["DatasetRecord"]:
        """List of dataset records contained in this dataset."""
        return self._records

    def add_record(self, record: "DatasetRecord") -> None:
        """Add a dataset record to this managed dataset."""
        if record.dataset_id != self.dataset_id:
            record._dataset_id = self.dataset_id
        self._records.append(record)
        self._last_update_time = int(time.time() * 1000)

    def merge_records(self, other_records: List["DatasetRecord"]) -> None:
        """
        Merge records from another source, handling deduplication based on record content.
        
        Args:
            other_records: List of DatasetRecord instances to merge into this dataset
        """
        # Simple deduplication based on record content hash
        existing_hashes = {self._compute_record_hash(record) for record in self._records}
        
        for record in other_records:
            record_hash = self._compute_record_hash(record)
            if record_hash not in existing_hashes:
                self.add_record(record)
                existing_hashes.add(record_hash)

    def _compute_record_hash(self, record: "DatasetRecord") -> str:
        """Compute a hash for a dataset record based on its content."""
        import hashlib
        
        # Create a deterministic representation of the record
        content = {
            "inputs": {inp.key: str(inp.value) for inp in record.inputs},
            "expectations": {k: str(v) for k, v in record.expectations.items()},
            "tags": record.tags,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def to_df(self) -> "pd.DataFrame":
        """Convert the dataset records to a pandas DataFrame for analysis."""
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

    def set_profile(self, profile_data: Dict[str, Any]) -> None:
        """Set the profile data containing summary statistics and metadata."""
        self._profile = json.dumps(profile_data)
        self._last_update_time = int(time.time() * 1000)

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
        
        # Import DatasetRecord here to avoid circular imports
        from mlflow.entities.dataset_record import DatasetRecord
        
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

    def update_metadata(
        self,
        name: Optional[str] = None,
        source: Optional[str] = None,
        source_type: Optional[str] = None,
        schema: Optional[str] = None,
        profile: Optional[str] = None,
        digest: Optional[str] = None,
        last_updated_by: Optional[str] = None,
        experiment_ids: Optional[List[str]] = None,
    ) -> None:
        """Update dataset metadata fields."""
        if name is not None:
            self._name = name
        if source is not None:
            self._source = source
        if source_type is not None:
            self._source_type = source_type
        if schema is not None:
            self._schema = schema
        if profile is not None:
            self._profile = profile
        if digest is not None:
            self._digest = digest
        if last_updated_by is not None:
            self._last_updated_by = last_updated_by
        if experiment_ids is not None:
            self._experiment_ids = experiment_ids
        
        self._last_update_time = int(time.time() * 1000)
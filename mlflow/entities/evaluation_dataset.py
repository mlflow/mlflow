from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.exceptions import MlflowException
from mlflow.protos.evaluation_datasets_pb2 import DatasetRecordSource as ProtoDatasetRecordSource
from mlflow.protos.evaluation_datasets_pb2 import EvaluationDataset as ProtoEvaluationDataset
from mlflow.utils.time import get_current_time_millis

if TYPE_CHECKING:
    import pandas as pd

    from mlflow.entities.trace import Trace


@dataclass
class EvaluationDataset(_MlflowObject):
    """
    Evaluation dataset for storing inputs and expectations for GenAI evaluation.

    This class supports lazy loading of records - when retrieved via get_evaluation_dataset(),
    only metadata is loaded. Records are fetched when to_df() or merge_records() is called.
    """

    dataset_id: Optional[str] = None
    name: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    schema: Optional[str] = None
    profile: Optional[str] = None
    digest: Optional[str] = None
    created_time: Optional[int] = None
    last_update_time: Optional[int] = None
    created_by: Optional[str] = None
    last_updated_by: Optional[str] = None
    experiment_ids: list[str] = field(default_factory=list)
    _records: Optional[list[DatasetRecord]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = get_current_time_millis()
        if self.last_update_time is None:
            self.last_update_time = get_current_time_millis()

    @property
    def records(self) -> list[DatasetRecord]:
        """
        Get dataset records, loading them if necessary.

        This property implements lazy loading - records are only fetched from the backend
        when accessed for the first time.
        """
        if self._records is None:
            from mlflow.tracking._tracking_service.utils import _get_store

            tracking_store = _get_store()
            # TODO: Remove this hasattr check once all tracking stores
            #  implement the evaluation dataset APIs
            if hasattr(tracking_store, "_load_dataset_records"):
                self._records = tracking_store._load_dataset_records(self.dataset_id)
        return self._records or []

    def has_records(self) -> bool:
        """Check if dataset records are loaded without triggering a load."""
        return self._records is not None

    def merge_records(
        self, records: Union[list[dict[str, Any]], "pd.DataFrame", list["Trace"]]
    ) -> "EvaluationDataset":
        """
        Merge new records with existing ones.

        Args:
            records: Records to merge. Can be:
                - List of dictionaries with 'inputs' and optionally 'expectations' and 'tags'
                - DataFrame with 'inputs' column and optionally 'expectations' and 'tags' columns
                - List of Trace objects

        Returns:
            Self for method chaining
        """
        import pandas as pd

        from mlflow.entities.trace import Trace

        if isinstance(records, pd.DataFrame):
            record_dicts = records.to_dict("records")
        elif isinstance(records, list) and records and isinstance(records[0], Trace):
            record_dicts = []
            for i, trace in enumerate(records):
                if not isinstance(trace, Trace):
                    raise MlflowException.invalid_parameter_value(
                        f"Mixed types in trace list. Expected all elements to be Trace objects, "
                        f"but element at index {i} is {type(trace).__name__}"
                    )
                root_span = trace.data._get_root_span()
                inputs = root_span.inputs if root_span and root_span.inputs is not None else {}

                expectations = {}
                expectation_assessments = trace.search_assessments(type="expectation")
                for expectation in expectation_assessments:
                    expectations[expectation.name] = expectation.value

                trace_id = trace.info.trace_id

                record_dict = {
                    "inputs": inputs,
                    "expectations": expectations,
                    "source": {"source_type": "TRACE", "source_data": {"trace_id": trace_id}},
                }
                record_dicts.append(record_dict)
        else:
            record_dicts = records

        for record in record_dicts:
            if not isinstance(record, dict):
                raise MlflowException.invalid_parameter_value("Each record must be a dictionary")
            if "inputs" not in record:
                raise MlflowException.invalid_parameter_value(
                    "Each record must have an 'inputs' field"
                )

        from mlflow.tracking._tracking_service.utils import _get_store

        tracking_store = _get_store()

        # TODO: Remove this hasattr check once all tracking stores implement
        #  the evaluation dataset APIs
        if hasattr(tracking_store, "upsert_evaluation_dataset_records"):
            try:
                tracking_store.get_evaluation_dataset(self.dataset_id)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Cannot add records to dataset {self.dataset_id}: Dataset not found in "
                    f"current tracking store. The dataset must exist in the backend before "
                    f"adding records."
                ) from e

            tracking_store.upsert_evaluation_dataset_records(
                dataset_id=self.dataset_id, records=record_dicts, updated_by=self.last_updated_by
            )
            self._records = None
        else:
            if self._records is None:
                self._records = []

            existing_records_map = {}
            for existing_record in self._records:
                inputs_key = json.dumps(existing_record.inputs, sort_keys=True)
                existing_records_map[inputs_key] = existing_record

            for record_dict in record_dicts:
                inputs = record_dict.get("inputs", {})
                inputs_key = json.dumps(inputs, sort_keys=True)

                if inputs_key in existing_records_map:
                    existing_record = existing_records_map[inputs_key]

                    if new_expectations := record_dict.get("expectations"):
                        if existing_record.expectations is None:
                            existing_record.expectations = {}
                        existing_record.expectations.update(new_expectations)

                    if new_tags := record_dict.get("tags"):
                        if existing_record.tags is None:
                            existing_record.tags = {}
                        existing_record.tags.update(new_tags)

                    existing_record.last_update_time = get_current_time_millis()
                else:
                    source = record_dict.get("source")
                    if source and isinstance(source, dict):
                        source = DatasetRecordSource.from_dict(source)

                    record = DatasetRecord(
                        dataset_id=self.dataset_id,
                        inputs=inputs,
                        expectations=record_dict.get("expectations"),
                        tags=record_dict.get("tags"),
                        source=source,
                        created_by=self.created_by,
                        last_updated_by=self.last_updated_by,
                    )
                    self._records.append(record)
                    existing_records_map[inputs_key] = record

        return self

    def to_df(self) -> "pd.DataFrame":
        """
        Convert dataset records to a pandas DataFrame.

        This method triggers lazy loading of records if they haven't been loaded yet.

        Returns:
            DataFrame with columns for inputs, expectations, tags, and metadata
        """
        import pandas as pd

        records = self.records

        if not records:
            return pd.DataFrame(
                columns=["inputs", "expectations", "tags", "source_type", "source_id"]
            )

        data = []
        for record in records:
            row = {
                "inputs": record.inputs,
                "expectations": record.expectations,
                "tags": record.tags,
                "source_type": record.source.get("source_type") if record.source else None,
                "source_id": record.source_id,
                "created_time": record.created_time,
                "dataset_record_id": record.dataset_record_id,
            }
            data.append(row)

        return pd.DataFrame(data)

    def to_proto(self) -> ProtoEvaluationDataset:
        """Convert to protobuf representation."""
        proto = ProtoEvaluationDataset()

        if self.dataset_id is not None:
            proto.dataset_id = self.dataset_id
        if self.name is not None:
            proto.name = self.name
        if self.source is not None:
            proto.source = self.source
        if self.source_type is not None:
            proto.source_type = ProtoDatasetRecordSource.SourceType.Value(self.source_type)
        if self.schema is not None:
            proto.schema = self.schema
        if self.profile is not None:
            proto.profile = self.profile
        if self.digest is not None:
            proto.digest = self.digest
        if self.created_time is not None:
            proto.created_time = self.created_time
        if self.last_update_time is not None:
            proto.last_update_time = self.last_update_time
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by

        proto.experiment_ids.extend(self.experiment_ids)

        return proto

    @classmethod
    def from_proto(cls, proto: ProtoEvaluationDataset) -> "EvaluationDataset":
        """Create instance from protobuf representation."""
        return cls(
            dataset_id=proto.dataset_id if proto.HasField("dataset_id") else None,
            name=proto.name if proto.HasField("name") else None,
            source=proto.source if proto.HasField("source") else None,
            source_type=DatasetRecordSourceType.from_proto(proto.source_type)
            if proto.HasField("source_type")
            else None,
            schema=proto.schema if proto.HasField("schema") else None,
            profile=proto.profile if proto.HasField("profile") else None,
            digest=proto.digest if proto.HasField("digest") else None,
            created_time=proto.created_time if proto.HasField("created_time") else None,
            last_update_time=proto.last_update_time if proto.HasField("last_update_time") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
            experiment_ids=list(proto.experiment_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
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
        }

        if self._records is not None:
            result["records"] = [record.to_dict() for record in self._records]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        """Create instance from dictionary representation."""
        dataset = cls(
            dataset_id=data.get("dataset_id"),
            name=data.get("name"),
            source=data.get("source"),
            source_type=data.get("source_type"),
            schema=data.get("schema"),
            profile=data.get("profile"),
            digest=data.get("digest"),
            created_time=data.get("created_time"),
            last_update_time=data.get("last_update_time"),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            experiment_ids=data.get("experiment_ids", []),
        )

        if records_data := data.get("records"):
            dataset._records = [
                DatasetRecord.from_dict(record_data) for record_data in records_data
            ]

        return dataset

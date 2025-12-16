from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mlflow.data import Dataset
from mlflow.data.evaluation_dataset_source import EvaluationDatasetSource
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin
from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.entities.dataset_record_source import DatasetRecordSourceType
from mlflow.exceptions import MlflowException
from mlflow.protos.datasets_pb2 import Dataset as ProtoDataset
from mlflow.telemetry.events import MergeRecordsEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_USER

if TYPE_CHECKING:
    import pandas as pd

    from mlflow.entities.trace import Trace


class EvaluationDataset(_MlflowObject, Dataset, PyFuncConvertibleDatasetMixin):
    """
    Evaluation dataset for storing inputs and expectations for GenAI evaluation.

    This class supports lazy loading of records - when retrieved via get_evaluation_dataset(),
    only metadata is loaded. Records are fetched when to_df() or merge_records() is called.
    """

    def __init__(
        self,
        dataset_id: str,
        name: str,
        digest: str,
        created_time: int,
        last_update_time: int,
        tags: dict[str, Any] | None = None,
        schema: str | None = None,
        profile: str | None = None,
        created_by: str | None = None,
        last_updated_by: str | None = None,
    ):
        """Initialize the EvaluationDataset."""
        self.dataset_id = dataset_id
        self.created_time = created_time
        self.last_update_time = last_update_time
        self.tags = tags
        self._schema = schema
        self._profile = profile
        self.created_by = created_by
        self.last_updated_by = last_updated_by
        self._experiment_ids = None
        self._records = None

        source = EvaluationDatasetSource(dataset_id=self.dataset_id)
        Dataset.__init__(self, source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Compute digest for the dataset. This is called by Dataset.__init__ if no digest is provided.
        Since we always have a digest from the dataclass initialization, this should not be called.
        """
        return self.digest

    @property
    def source(self) -> EvaluationDatasetSource:
        """Override source property to return the correct type."""
        return self._source

    @property
    def schema(self) -> str | None:
        """
        Dataset schema information.
        """
        return self._schema

    @property
    def profile(self) -> str | None:
        """
        Dataset profile information.
        """
        return self._profile

    @property
    def experiment_ids(self) -> list[str]:
        """
        Get associated experiment IDs, loading them if necessary.

        This property implements lazy loading - experiment IDs are only fetched from the backend
        when accessed for the first time.
        """
        if self._experiment_ids is None:
            self._load_experiment_ids()
        return self._experiment_ids or []

    @experiment_ids.setter
    def experiment_ids(self, value: list[str]):
        """Set experiment IDs directly."""
        self._experiment_ids = value or []

    def _load_experiment_ids(self):
        """Load experiment IDs from the backend."""
        from mlflow.tracking._tracking_service.utils import _get_store

        tracking_store = _get_store()
        self._experiment_ids = tracking_store.get_dataset_experiment_ids(self.dataset_id)

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
            # For lazy loading, we want all records (no pagination)
            self._records, _ = tracking_store._load_dataset_records(
                self.dataset_id, max_results=None
            )
        return self._records or []

    def has_records(self) -> bool:
        """Check if dataset records are loaded without triggering a load."""
        return self._records is not None

    def _process_trace_records(self, traces: list["Trace"]) -> list[dict[str, Any]]:
        """Convert a list of Trace objects to dataset record dictionaries.

        Args:
            traces: List of Trace objects to convert

        Returns:
            List of dictionaries with 'inputs', 'expectations', and 'source' fields
        """
        from mlflow.entities.trace import Trace

        record_dicts = []
        for i, trace in enumerate(traces):
            if not isinstance(trace, Trace):
                raise MlflowException.invalid_parameter_value(
                    f"Mixed types in trace list. Expected all elements to be Trace objects, "
                    f"but element at index {i} is {type(trace).__name__}"
                )

            root_span = trace.data._get_root_span()
            inputs = root_span.inputs if root_span and root_span.inputs is not None else {}
            outputs = root_span.outputs if root_span and root_span.outputs is not None else None

            expectations = {}
            expectation_assessments = trace.search_assessments(type="expectation")
            for expectation in expectation_assessments:
                expectations[expectation.name] = expectation.value

            # Preserve session metadata from the original trace
            source_data = {"trace_id": trace.info.trace_id}
            if session_id := trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
                source_data["session_id"] = session_id

            record_dict = {
                "inputs": inputs,
                "outputs": outputs,
                "expectations": expectations,
                "source": {
                    "source_type": DatasetRecordSourceType.TRACE.value,
                    "source_data": source_data,
                },
            }
            record_dicts.append(record_dict)

        return record_dicts

    def _process_dataframe_records(self, df: "pd.DataFrame") -> list[dict[str, Any]]:
        """Process a DataFrame into dataset record dictionaries.

        Args:
            df: DataFrame to process. Can be either:
                - DataFrame from search_traces with 'trace' column containing Trace objects/JSON
                - Standard DataFrame with 'inputs', 'expectations' columns

        Returns:
            List of dictionaries with 'inputs', 'expectations', and optionally 'source' fields
        """
        if "trace" in df.columns:
            from mlflow.entities.trace import Trace

            traces = [
                Trace.from_json(trace_item) if isinstance(trace_item, str) else trace_item
                for trace_item in df["trace"]
            ]

            return self._process_trace_records(traces)
        else:
            return df.to_dict("records")

    @record_usage_event(MergeRecordsEvent)
    def merge_records(
        self, records: list[dict[str, Any]] | "pd.DataFrame" | list["Trace"]
    ) -> "EvaluationDataset":
        """
        Merge new records with existing ones.

        Args:
            records: Records to merge. Can be:
                - List of dictionaries with 'inputs' and optionally 'expectations' and 'tags'
                - DataFrame from mlflow.search_traces() - automatically parsed and converted
                - DataFrame with 'inputs' column and optionally 'expectations' and 'tags' columns
                - List of Trace objects

        Returns:
            Self for method chaining

        Example:
            .. code-block:: python

                # Direct usage with search_traces DataFrame output
                traces_df = mlflow.search_traces()  # Returns DataFrame by default
                dataset.merge_records(traces_df)  # No extraction needed

                # Or with standard DataFrame
                df = pd.DataFrame([{"inputs": {"q": "What?"}, "expectations": {"a": "Answer"}}])
                dataset.merge_records(df)
        """
        import pandas as pd

        from mlflow.entities.trace import Trace
        from mlflow.tracking._tracking_service.utils import _get_store, get_tracking_uri

        if isinstance(records, pd.DataFrame):
            record_dicts = self._process_dataframe_records(records)
        elif isinstance(records, list) and records and isinstance(records[0], Trace):
            record_dicts = self._process_trace_records(records)
        else:
            record_dicts = records

        self._validate_record_dicts(record_dicts)

        self._infer_source_types(record_dicts)

        tracking_store = _get_store()

        try:
            tracking_store.get_dataset(self.dataset_id)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Cannot add records to dataset {self.dataset_id}: Dataset not found. "
                f"Please verify the dataset exists and check your tracking URI is set correctly "
                f"(currently set to: {get_tracking_uri()})."
            ) from e

        context_tags = context_registry.resolve_tags()
        if user_tag := context_tags.get(MLFLOW_USER):
            for record in record_dicts:
                if "tags" not in record:
                    record["tags"] = {}
                if MLFLOW_USER not in record["tags"]:
                    record["tags"][MLFLOW_USER] = user_tag

        tracking_store.upsert_dataset_records(dataset_id=self.dataset_id, records=record_dicts)
        self._records = None

        return self

    def _validate_record_dicts(self, record_dicts: list[dict[str, Any]]) -> None:
        """Validate that record dictionaries have the required structure.

        Args:
            record_dicts: List of record dictionaries to validate

        Raises:
            MlflowException: If records don't have the required structure
        """
        for record in record_dicts:
            if not isinstance(record, dict):
                raise MlflowException.invalid_parameter_value("Each record must be a dictionary")
            if "inputs" not in record:
                raise MlflowException.invalid_parameter_value(
                    "Each record must have an 'inputs' field"
                )

    def _infer_source_types(self, record_dicts: list[dict[str, Any]]) -> None:
        """Infer source types for records without explicit source information.

        Simple inference rules:
        - Records with expectations -> HUMAN (manual test cases/ground truth)
        - Records with inputs but no expectations -> CODE (programmatically generated)

        Inference can be overridden by providing explicit source information.

        Note that trace inputs (from List[Trace] or pd.DataFrame of Trace data) will
        always be inferred as a trace source type when processing trace records.

        Args:
            record_dicts: List of record dictionaries to process (modified in place)
        """
        for record in record_dicts:
            if "source" in record:
                continue

            if "expectations" in record and record["expectations"]:
                record["source"] = {
                    "source_type": DatasetRecordSourceType.HUMAN.value,
                    "source_data": {},
                }
            elif "inputs" in record and "expectations" not in record:
                record["source"] = {
                    "source_type": DatasetRecordSourceType.CODE.value,
                    "source_data": {},
                }

    def to_df(self) -> "pd.DataFrame":
        """
        Convert dataset records to a pandas DataFrame.

        This method triggers lazy loading of records if they haven't been loaded yet.

        Returns:
            DataFrame with columns for inputs, outputs, expectations, tags, and metadata
        """
        import pandas as pd

        records = self.records

        if not records:
            return pd.DataFrame(
                columns=[
                    "inputs",
                    "outputs",
                    "expectations",
                    "tags",
                    "source_type",
                    "source_id",
                    "source",
                    "created_time",
                    "dataset_record_id",
                ]
            )

        data = [
            {
                "inputs": record.inputs,
                "outputs": record.outputs,
                "expectations": record.expectations,
                "tags": record.tags,
                "source_type": record.source_type,
                "source_id": record.source_id,
                "source": record.source,
                "created_time": record.created_time,
                "dataset_record_id": record.dataset_record_id,
            }
            for record in records
        ]

        return pd.DataFrame(data)

    def to_proto(self) -> ProtoDataset:
        """Convert to protobuf representation."""
        proto = ProtoDataset()

        proto.dataset_id = self.dataset_id
        proto.name = self.name
        if self.tags is not None:
            proto.tags = json.dumps(self.tags)
        if self.schema is not None:
            proto.schema = self.schema
        if self.profile is not None:
            proto.profile = self.profile
        proto.digest = self.digest
        proto.created_time = self.created_time
        proto.last_update_time = self.last_update_time
        if self.created_by is not None:
            proto.created_by = self.created_by
        if self.last_updated_by is not None:
            proto.last_updated_by = self.last_updated_by
        if self._experiment_ids is not None:
            proto.experiment_ids.extend(self._experiment_ids)

        return proto

    @classmethod
    def from_proto(cls, proto: ProtoDataset) -> "EvaluationDataset":
        """Create instance from protobuf representation."""
        tags = None
        if proto.HasField("tags"):
            tags = json.loads(proto.tags)

        dataset = cls(
            dataset_id=proto.dataset_id,
            name=proto.name,
            digest=proto.digest,
            created_time=proto.created_time,
            last_update_time=proto.last_update_time,
            tags=tags,
            schema=proto.schema if proto.HasField("schema") else None,
            profile=proto.profile if proto.HasField("profile") else None,
            created_by=proto.created_by if proto.HasField("created_by") else None,
            last_updated_by=proto.last_updated_by if proto.HasField("last_updated_by") else None,
        )
        if proto.experiment_ids:
            dataset._experiment_ids = list(proto.experiment_ids)
        return dataset

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()

        result.update(
            {
                "dataset_id": self.dataset_id,
                "tags": self.tags,
                "schema": self.schema,
                "profile": self.profile,
                "created_time": self.created_time,
                "last_update_time": self.last_update_time,
                "created_by": self.created_by,
                "last_updated_by": self.last_updated_by,
                "experiment_ids": self.experiment_ids,
            }
        )

        result["records"] = [record.to_dict() for record in self.records]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        """Create instance from dictionary representation."""
        if "dataset_id" not in data:
            raise ValueError("dataset_id is required")
        if "name" not in data:
            raise ValueError("name is required")
        if "digest" not in data:
            raise ValueError("digest is required")
        if "created_time" not in data:
            raise ValueError("created_time is required")
        if "last_update_time" not in data:
            raise ValueError("last_update_time is required")

        dataset = cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            digest=data["digest"],
            created_time=data["created_time"],
            last_update_time=data["last_update_time"],
            tags=data.get("tags"),
            schema=data.get("schema"),
            profile=data.get("profile"),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
        )
        if "experiment_ids" in data:
            dataset._experiment_ids = data["experiment_ids"]

        if "records" in data:
            dataset._records = [
                DatasetRecord.from_dict(record_data) for record_data in data["records"]
            ]

        return dataset

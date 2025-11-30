from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2 as pb

_UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME = "mlflow_experiment_trace_otel_spans"
_UC_SCHEMA_DEFAULT_LOGS_TABLE_NAME = "mlflow_experiment_trace_otel_logs"


@dataclass
class TraceLocationBase(_MlflowObject, ABC):
    """
    Base class for trace location classes.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceLocationBase": ...


@dataclass
class MlflowExperimentLocation(TraceLocationBase):
    """
    Represents the location of an MLflow experiment.

    Args:
        experiment_id: The ID of the MLflow experiment where the trace is stored.
    """

    experiment_id: str

    def to_proto(self):
        return pb.TraceLocation.MlflowExperimentLocation(experiment_id=self.experiment_id)

    @classmethod
    def from_proto(cls, proto) -> "MlflowExperimentLocation":
        return cls(experiment_id=proto.experiment_id)

    def to_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MlflowExperimentLocation":
        return cls(experiment_id=d["experiment_id"])


@dataclass
class InferenceTableLocation(TraceLocationBase):
    """
    Represents the location of a Databricks inference table.

    Args:
        full_table_name: The fully qualified name of the inference table where
            the trace is stored, in the format of `<catalog>.<schema>.<table>`.
    """

    full_table_name: str

    def to_proto(self):
        return pb.TraceLocation.InferenceTableLocation(full_table_name=self.full_table_name)

    @classmethod
    def from_proto(cls, proto) -> "InferenceTableLocation":
        return cls(full_table_name=proto.full_table_name)

    def to_dict(self) -> dict[str, Any]:
        return {"full_table_name": self.full_table_name}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InferenceTableLocation":
        return cls(full_table_name=d["full_table_name"])


@dataclass
class UCSchemaLocation(TraceLocationBase):
    """
    Represents the location of a Databricks Unity Catalog (UC) schema.

    Args:
        catalog_name: The name of the Unity Catalog catalog name.
        schema_name: The name of the Unity Catalog schema.
    """

    catalog_name: str
    schema_name: str

    # These table names are set by the backend
    _otel_spans_table_name: str | None = _UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME
    _otel_logs_table_name: str | None = _UC_SCHEMA_DEFAULT_LOGS_TABLE_NAME

    @property
    def schema_location(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    @property
    def full_otel_spans_table_name(self) -> str | None:
        if self._otel_spans_table_name:
            return f"{self.catalog_name}.{self.schema_name}.{self._otel_spans_table_name}"

    @property
    def full_otel_logs_table_name(self) -> str | None:
        if self._otel_logs_table_name:
            return f"{self.catalog_name}.{self.schema_name}.{self._otel_logs_table_name}"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "catalog_name": self.catalog_name,
            "schema_name": self.schema_name,
        }
        if self._otel_spans_table_name:
            d["otel_spans_table_name"] = self._otel_spans_table_name
        if self._otel_logs_table_name:
            d["otel_logs_table_name"] = self._otel_logs_table_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UCSchemaLocation":
        location = cls(
            catalog_name=d["catalog_name"],
            schema_name=d["schema_name"],
        )
        if otel_spans_table_name := d.get("otel_spans_table_name"):
            location._otel_spans_table_name = otel_spans_table_name
        if otel_logs_table_name := d.get("otel_logs_table_name"):
            location._otel_logs_table_name = otel_logs_table_name
        return location


class TraceLocationType(str, Enum):
    TRACE_LOCATION_TYPE_UNSPECIFIED = "TRACE_LOCATION_TYPE_UNSPECIFIED"
    MLFLOW_EXPERIMENT = "MLFLOW_EXPERIMENT"
    INFERENCE_TABLE = "INFERENCE_TABLE"
    UC_SCHEMA = "UC_SCHEMA"

    def to_proto(self):
        return pb.TraceLocation.TraceLocationType.Value(self)

    @classmethod
    def from_proto(cls, proto: int) -> "TraceLocationType":
        return TraceLocationType(pb.TraceLocation.TraceLocationType.Name(proto))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceLocationType":
        return cls(d["type"])


@dataclass
class TraceLocation(_MlflowObject):
    """
    Represents the location where the trace is stored.

    Currently, MLflow supports two types of trace locations:

        - MLflow experiment: The trace is stored in an MLflow experiment.
        - Inference table: The trace is stored in a Databricks inference table.

    Args:
        type: The type of the trace location, should be one of the
            :py:class:`TraceLocationType` enum values.
        mlflow_experiment: The MLflow experiment location. Set this when the
            location type is MLflow experiment.
        inference_table: The inference table location. Set this when the
            location type is Databricks Inference table.
    """

    type: TraceLocationType
    mlflow_experiment: MlflowExperimentLocation | None = None
    inference_table: InferenceTableLocation | None = None
    uc_schema: UCSchemaLocation | None = None

    def __post_init__(self) -> None:
        if (
            sum(
                [
                    self.mlflow_experiment is not None,
                    self.inference_table is not None,
                    self.uc_schema is not None,
                ]
            )
            > 1
        ):
            raise MlflowException.invalid_parameter_value(
                "Only one of mlflow_experiment, inference_table, or uc_schema can be provided."
            )

        if (
            (self.mlflow_experiment and self.type != TraceLocationType.MLFLOW_EXPERIMENT)
            or (self.inference_table and self.type != TraceLocationType.INFERENCE_TABLE)
            or (self.uc_schema and self.type != TraceLocationType.UC_SCHEMA)
        ):
            raise MlflowException.invalid_parameter_value(
                f"Trace location type {self.type} does not match the provided location "
                f"{self.mlflow_experiment or self.inference_table or self.uc_schema}."
            )

    def to_dict(self) -> dict[str, Any]:
        d = {"type": self.type.value}
        if self.mlflow_experiment:
            d["mlflow_experiment"] = self.mlflow_experiment.to_dict()
        elif self.inference_table:
            d["inference_table"] = self.inference_table.to_dict()
        elif self.uc_schema:
            d["uc_schema"] = self.uc_schema.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceLocation":
        return cls(
            type=TraceLocationType(d["type"]),
            mlflow_experiment=(
                MlflowExperimentLocation.from_dict(v) if (v := d.get("mlflow_experiment")) else None
            ),
            inference_table=(
                InferenceTableLocation.from_dict(v) if (v := d.get("inference_table")) else None
            ),
            uc_schema=(UCSchemaLocation.from_dict(v) if (v := d.get("uc_schema")) else None),
        )

    def to_proto(self) -> pb.TraceLocation:
        if self.mlflow_experiment:
            return pb.TraceLocation(
                type=self.type.to_proto(),
                mlflow_experiment=self.mlflow_experiment.to_proto(),
            )
        elif self.inference_table:
            return pb.TraceLocation(
                type=self.type.to_proto(),
                inference_table=self.inference_table.to_proto(),
            )
        # uc schema is not supported in to_proto since it's databricks specific, should use
        # databricks_service_utils to convert to proto
        else:
            return pb.TraceLocation(type=self.type.to_proto())

    @classmethod
    def from_proto(cls, proto) -> "TraceLocation":
        from mlflow.utils.databricks_tracing_utils import trace_location_from_proto

        return trace_location_from_proto(proto)

    @classmethod
    def from_experiment_id(cls, experiment_id: str) -> "TraceLocation":
        return cls(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        )

    @classmethod
    def from_databricks_uc_schema(cls, catalog_name: str, schema_name: str) -> "TraceLocation":
        return cls(
            type=TraceLocationType.UC_SCHEMA,
            uc_schema=UCSchemaLocation(catalog_name=catalog_name, schema_name=schema_name),
        )

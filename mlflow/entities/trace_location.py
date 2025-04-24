from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_trace_server_pb2 as pb


@dataclass
class MlflowExperimentLocation(_MlflowObject):
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
class InferenceTableLocation(_MlflowObject):
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


class TraceLocationType(str, Enum):
    TRACE_LOCATION_TYPE_UNSPECIFIED = "TRACE_LOCATION_TYPE_UNSPECIFIED"
    MLFLOW_EXPERIMENT = "MLFLOW_EXPERIMENT"
    INFERENCE_TABLE = "INFERENCE_TABLE"

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
    type: TraceLocationType
    mlflow_experiment: Optional[MlflowExperimentLocation] = None
    inference_table: Optional[InferenceTableLocation] = None

    def __post_init__(self) -> None:
        if self.mlflow_experiment is not None and self.inference_table is not None:
            raise MlflowException.invalid_parameter_value(
                "Only one of mlflow_experiment or inference_table can be provided."
            )

        if (self.mlflow_experiment and self.type != TraceLocationType.MLFLOW_EXPERIMENT) or (
            self.inference_table and self.type != TraceLocationType.INFERENCE_TABLE
        ):
            raise MlflowException.invalid_parameter_value(
                f"Trace location type {type} does not match the provided location "
                f"{self.mlflow_experiment or self.inference_table}."
            )

    def to_dict(self) -> dict[str, Any]:
        d = {"type": self.type.value}
        if self.mlflow_experiment:
            d["mlflow_experiment"] = self.mlflow_experiment.to_dict()
        elif self.inference_table:
            d["inference_table"] = self.inference_table.to_dict()
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
        )

    def to_proto(self):
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

        else:
            return pb.TraceLocation(type=self.type.to_proto())

    @classmethod
    def from_proto(cls, proto) -> "TraceLocation":
        type_ = TraceLocationType.from_proto(proto.type)
        if proto.WhichOneof("identifier") == "mlflow_experiment":
            return cls(
                type=type_,
                mlflow_experiment=MlflowExperimentLocation.from_proto(proto.mlflow_experiment),
            )
        elif proto.WhichOneof("identifier") == "inference_table":
            return cls(
                type=type_,
                inference_table=InferenceTableLocation.from_proto(proto.inference_table),
            )
        else:
            return cls(type=type_)

    @classmethod
    def from_experiment_id(cls, experiment_id: str) -> "TraceLocation":
        return cls(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        )

from __future__ import annotations

from typing import cast

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo
from mlflow.protos.service_pb2 import RunStatus as ProtoRunStatus


def check_run_is_active(run_info: RunInfo) -> None:
    if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
        raise MlflowException(
            f"The run {run_info.run_id} must be in 'active' lifecycle_stage.",
            error_code=INVALID_PARAMETER_VALUE,
        )


class searchable_attribute(property):
    # Wrapper class over property to designate some of the properties as searchable
    # run attributes
    pass


class orderable_attribute(property):
    # Wrapper class over property to designate some of the properties as orderable
    # run attributes
    pass


class RunInfo(_MlflowObject):
    """
    Metadata about a run.
    """

    def __init__(
        self,
        run_id: str,
        experiment_id: str,
        user_id: str,
        status: str,
        start_time: int,
        end_time: int | None,
        lifecycle_stage: str,
        artifact_uri: str | None = None,
        run_name: str | None = None,
    ) -> None:
        if experiment_id is None:
            raise Exception("experiment_id cannot be None")
        if user_id is None:
            raise Exception("user_id cannot be None")
        if status is None:
            raise Exception("status cannot be None")
        if start_time is None:
            raise Exception("start_time cannot be None")
        self._run_id = run_id
        self._experiment_id = experiment_id
        self._user_id = user_id
        self._status = status
        self._start_time = start_time
        self._end_time = end_time
        self._lifecycle_stage = lifecycle_stage
        self._artifact_uri = artifact_uri
        self._run_name = run_name

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            # TODO deep equality here?
            return self.__dict__ == other.__dict__
        return False

    def _copy_with_overrides(self, status=None, end_time=None, lifecycle_stage=None, run_name=None):
        """A copy of the RunInfo with certain attributes modified."""
        proto = self.to_proto()
        if status:
            proto.status = status
        if end_time:
            proto.end_time = end_time
        if lifecycle_stage:
            proto.lifecycle_stage = lifecycle_stage
        if run_name:
            proto.run_name = run_name
        return RunInfo.from_proto(proto)

    @searchable_attribute
    def run_id(self) -> str:
        """String containing run id."""
        return self._run_id

    @property
    def experiment_id(self) -> str:
        """String ID of the experiment for the current run."""
        return self._experiment_id

    @searchable_attribute
    def run_name(self) -> str | None:
        """String containing run name."""
        return self._run_name

    def _set_run_name(self, new_name):
        self._run_name = new_name

    @searchable_attribute
    def user_id(self) -> str:
        """String ID of the user who initiated this run."""
        return self._user_id

    @searchable_attribute
    def status(self) -> str:
        """
        One of the values in :py:class:`mlflow.entities.RunStatus`
        describing the status of the run.
        """
        return self._status

    @searchable_attribute
    def start_time(self) -> int:
        """Start time of the run, in number of milliseconds since the UNIX epoch."""
        return self._start_time

    @searchable_attribute
    def end_time(self) -> int | None:
        """End time of the run, in number of milliseconds since the UNIX epoch."""
        return self._end_time

    @searchable_attribute
    def artifact_uri(self) -> str | None:
        """String root artifact URI of the run."""
        return self._artifact_uri

    @property
    def lifecycle_stage(self) -> str:
        """
        One of the values in :py:class:`mlflow.entities.lifecycle_stage.LifecycleStage`
        describing the lifecycle stage of the run.
        """
        return self._lifecycle_stage

    def to_proto(self) -> ProtoRunInfo:
        proto = ProtoRunInfo()
        proto.run_uuid = self.run_id
        proto.run_id = self.run_id
        if self.run_name is not None:
            proto.run_name = self.run_name
        proto.experiment_id = self.experiment_id
        proto.user_id = self.user_id
        # The proto field is typed as the protobuf enum wrapper (an int subclass), while
        # `RunStatus.from_string` returns a plain `int`, hence the cast.
        proto.status = cast(ProtoRunStatus, RunStatus.from_string(self.status))
        proto.start_time = self.start_time
        if self.end_time:
            proto.end_time = self.end_time
        if self.artifact_uri:
            proto.artifact_uri = self.artifact_uri
        proto.lifecycle_stage = self.lifecycle_stage
        return proto

    @classmethod
    def from_proto(cls, proto: ProtoRunInfo) -> RunInfo:
        end_time: int | None = proto.end_time
        # The proto2 default scalar value of zero indicates that the run's end time is absent.
        # An absent end time is represented with a NoneType in the `RunInfo` class
        if end_time == 0:
            end_time = None
        return cls(
            run_id=proto.run_id,
            run_name=proto.run_name,
            experiment_id=proto.experiment_id,
            user_id=proto.user_id,
            status=RunStatus.to_string(proto.status),
            start_time=proto.start_time,
            end_time=end_time,
            lifecycle_stage=proto.lifecycle_stage,
            artifact_uri=proto.artifact_uri,
        )

    @classmethod
    def get_searchable_attributes(cls) -> list[str]:
        return sorted([
            p for p in cls.__dict__ if isinstance(getattr(cls, p), searchable_attribute)
        ])

    @classmethod
    def get_orderable_attributes(cls) -> list[str]:
        # Note that all searchable attributes are also orderable.
        return sorted([
            p
            for p in cls.__dict__
            if isinstance(getattr(cls, p), (searchable_attribute, orderable_attribute))
        ])

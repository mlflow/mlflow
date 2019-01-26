from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException

from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo


def check_run_is_active(run_info):
    if run_info.lifecycle_stage != LifecycleStage.ACTIVE:
        raise MlflowException("The run {} must be in 'active' lifecycle_stage."
                              .format(run_info.run_uuid))


def check_run_is_deleted(run_info):
    if run_info.lifecycle_stage != LifecycleStage.DELETED:
        raise MlflowException("The run {} must be in 'deleted' lifecycle_stage."
                              .format(run_info.run_uuid))


class RunInfo(_MLflowObject):
    """
    Metadata about a run.
    """

    def __init__(self, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                 user_id, status, start_time, end_time, source_version, lifecycle_stage,
                 artifact_uri=None):
        if run_uuid is None:
            raise Exception("run_uuid cannot be None")
        if experiment_id is None:
            raise Exception("experiment_id cannot be None")
        if name is None:
            raise Exception("name cannot be None")
        if source_type is None:
            raise Exception("source_type cannot be None")
        if source_name is None:
            raise Exception("source_name cannot be None")
        if user_id is None:
            raise Exception("user_id cannot be None")
        if status is None:
            raise Exception("status cannot be None")
        if start_time is None:
            raise Exception("start_time cannot be None")
        self._run_uuid = run_uuid
        self._experiment_id = experiment_id
        self._name = name
        self._source_type = source_type
        self._source_name = source_name
        self._entry_point_name = entry_point_name
        self._user_id = user_id
        self._status = status
        self._start_time = start_time
        self._end_time = end_time
        self._source_version = source_version
        self._lifecycle_stage = lifecycle_stage
        self._artifact_uri = artifact_uri

    def __eq__(self, other):
        if type(other) is type(self):
            # TODO deep equality here?
            return self.__dict__ == other.__dict__
        return False

    def _copy_with_overrides(self, status=None, end_time=None, lifecycle_stage=None):
        """A copy of the RunInfo with certain attributes modified."""
        proto = self.to_proto()
        if status:
            proto.status = status
        if end_time:
            proto.end_time = end_time
        if lifecycle_stage:
            proto.lifecycle_stage = lifecycle_stage
        return RunInfo.from_proto(proto)

    @property
    def run_uuid(self):
        """String containing run UUID."""
        return self._run_uuid

    @property
    def experiment_id(self):
        """Integer ID of the experiment for the current run."""
        return self._experiment_id

    @property
    def name(self):
        """String name of the run."""
        return self._name

    @property
    def source_type(self):
        """
        :py:class:`mlflow.entities.SourceType` describing the source of the run.
        """
        return self._source_type

    @property
    def source_name(self):
        """
        String name of the source of the run (GitHub URI of the project corresponding to the run,
        etc).
        """
        return self._source_name

    @property
    def entry_point_name(self):
        """String name of the entry point for the run."""
        return self._entry_point_name

    @property
    def user_id(self):
        """String ID of the user who initiated this run."""
        return self._user_id

    @property
    def status(self):
        """
        One of the values in :py:class:`mlflow.entities.RunStatus`
        describing the status of the run.
        """
        return self._status

    @property
    def start_time(self):
        """Start time of the run, in number of milliseconds since the UNIX epoch."""
        return self._start_time

    @property
    def end_time(self):
        """End time of the run, in number of milliseconds since the UNIX epoch."""
        return self._end_time

    @property
    def source_version(self):
        """String Git commit hash of the code used for the run, if available."""
        return self._source_version

    @property
    def artifact_uri(self):
        """String root artifact URI of the run."""
        return self._artifact_uri

    @property
    def lifecycle_stage(self):
        return self._lifecycle_stage

    def to_proto(self):
        proto = ProtoRunInfo()
        proto.run_uuid = self.run_uuid
        proto.experiment_id = self.experiment_id
        proto.name = self.name
        proto.source_type = self.source_type
        proto.source_name = self.source_name
        if self.entry_point_name:
            proto.entry_point_name = self.entry_point_name
        proto.user_id = self.user_id
        proto.status = self.status
        proto.start_time = self.start_time
        if self.end_time:
            proto.end_time = self.end_time
        if self.source_version:
            proto.source_version = self.source_version
        if self.artifact_uri:
            proto.artifact_uri = self.artifact_uri
        proto.lifecycle_stage = self.lifecycle_stage
        return proto

    @classmethod
    def from_proto(cls, proto):
        end_time = proto.end_time
        # The proto2 default scalar value of zero indicates that the run's end time is absent.
        # An absent end time is represented with a NoneType in the `RunInfo` class
        if end_time == 0:
            end_time = None
        return cls(run_uuid=proto.run_uuid, experiment_id=proto.experiment_id, name=proto.name,
                   source_type=proto.source_type, source_name=proto.source_name,
                   entry_point_name=proto.entry_point_name, user_id=proto.user_id,
                   status=proto.status, start_time=proto.start_time, end_time=end_time,
                   source_version=proto.source_version, lifecycle_stage=proto.lifecycle_stage,
                   artifact_uri=proto.artifact_uri)

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.entities.run_tag import RunTag

from mlflow.protos.service_pb2 import RunInfo as ProtoRunInfo


class RunInfo(_MLflowObject):

    def __init__(self, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
                 user_id, status, start_time, end_time, source_version, tags, artifact_uri=None):
        """ Class containing metadata for a run. """
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
        self._tags = tags
        self._artifact_uri = artifact_uri

    def __eq__(self, other):
        if type(other) is type(self):
            # TODO deep equality here?
            return self.__dict__ == other.__dict__
        return False

    def copy_with_overrides(self, status, end_time):
        """ Returns a copy the current RunInfo with certain attributes modified """
        proto = self.to_proto()
        proto.status = status
        if end_time:
            proto.end_time = end_time
        return RunInfo.from_proto(proto)

    @property
    def run_uuid(self):
        return self._run_uuid

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def name(self):
        return self._name

    @property
    def source_type(self):
        return self._source_type

    @property
    def source_name(self):
        return self._source_name

    @property
    def entry_point_name(self):
        return self._entry_point_name

    @property
    def user_id(self):
        return self._user_id

    @property
    def status(self):
        return self._status

    @property
    def start_time(self):
        """ Start time of the run, in number of milliseconds since the UNIX epoch. """
        return self._start_time

    @property
    def end_time(self):
        """ End time of the run, in number of milliseconds since the UNIX epoch. """
        return self._end_time

    @property
    def source_version(self):
        return self._source_version

    @property
    def tags(self):
        return self._tags

    @property
    def artifact_uri(self):
        return self._artifact_uri

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
        proto.tags.extend([tag.to_proto() for tag in self.tags])
        if self.artifact_uri:
            proto.artifact_uri = self.artifact_uri
        return proto

    @classmethod
    def from_proto(cls, proto):
        tags = [RunTag.from_proto(proto_tag) for proto_tag in proto.tags]
        return cls(proto.run_uuid, proto.experiment_id, proto.name, proto.source_type,
                   proto.source_name, proto.entry_point_name, proto.user_id, proto.status,
                   proto.start_time, proto.end_time, proto.source_version, tags,
                   proto.artifact_uri)

    @classmethod
    def from_dictionary(cls, the_dict):
        info = cls(**the_dict)
        # We must manually deserialize tags into proto tags
        info._tags = the_dict.get("tags", [])
        return info

    @classmethod
    def _properties(cls):
        # TODO: Hard coding this list of props for now. There has to be a clearer way...
        return ["run_uuid", "experiment_id", "name", "source_type", "source_name",
                "entry_point_name", "user_id", "status", "start_time", "end_time",
                "source_version", "tags", "artifact_uri"]

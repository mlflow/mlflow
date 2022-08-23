from mlflow.protos.service_pb2 import RunStatus as ProtoRunStatus


class RunStatus:
    """Enum for status of an :py:class:`mlflow.entities.Run`."""

    RUNNING = ProtoRunStatus.Value("RUNNING")
    SCHEDULED = ProtoRunStatus.Value("SCHEDULED")
    FINISHED = ProtoRunStatus.Value("FINISHED")
    FAILED = ProtoRunStatus.Value("FAILED")
    KILLED = ProtoRunStatus.Value("KILLED")

    _STRING_TO_STATUS = {k: ProtoRunStatus.Value(k) for k in ProtoRunStatus.keys()}
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}
    _TERMINATED_STATUSES = {FINISHED, FAILED, KILLED}

    @staticmethod
    def from_string(status_str):
        if status_str not in RunStatus._STRING_TO_STATUS:
            raise Exception(
                "Could not get run status corresponding to string %s. Valid run "
                "status strings: %s" % (status_str, list(RunStatus._STRING_TO_STATUS.keys()))
            )
        return RunStatus._STRING_TO_STATUS[status_str]

    @staticmethod
    def to_string(status):
        if status not in RunStatus._STATUS_TO_STRING:
            raise Exception(
                "Could not get string corresponding to run status %s. Valid run "
                "statuses: %s" % (status, list(RunStatus._STATUS_TO_STRING.keys()))
            )
        return RunStatus._STATUS_TO_STRING[status]

    @staticmethod
    def is_terminated(status):
        return status in RunStatus._TERMINATED_STATUSES

    @staticmethod
    def all_status():
        return list(RunStatus._STATUS_TO_STRING.keys())

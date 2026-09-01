from mlflow.protos.service_pb2 import RunStatus as ProtoRunStatus


class RunStatus:
    """Enum for status of an :py:class:`mlflow.entities.Run`."""

    RUNNING: int = ProtoRunStatus.Value("RUNNING")
    SCHEDULED: int = ProtoRunStatus.Value("SCHEDULED")
    FINISHED: int = ProtoRunStatus.Value("FINISHED")
    FAILED: int = ProtoRunStatus.Value("FAILED")
    KILLED: int = ProtoRunStatus.Value("KILLED")

    _STRING_TO_STATUS: dict[str, int] = {k: ProtoRunStatus.Value(k) for k in ProtoRunStatus.keys()}
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}
    _TERMINATED_STATUSES = {FINISHED, FAILED, KILLED}

    @staticmethod
    def from_string(status_str: str) -> int:
        if status_str not in RunStatus._STRING_TO_STATUS:
            raise Exception(
                f"Could not get run status corresponding to string {status_str}. Valid run "
                f"status strings: {list(RunStatus._STRING_TO_STATUS.keys())}"
            )
        return RunStatus._STRING_TO_STATUS[status_str]

    @staticmethod
    def to_string(status: int) -> str:
        if status not in RunStatus._STATUS_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to run status {status}. Valid run "
                f"statuses: {list(RunStatus._STATUS_TO_STRING.keys())}"
            )
        return RunStatus._STATUS_TO_STRING[status]

    @staticmethod
    def is_terminated(status: int) -> bool:
        return status in RunStatus._TERMINATED_STATUSES

    @staticmethod
    def all_status() -> list[int]:
        return list(RunStatus._STATUS_TO_STRING.keys())

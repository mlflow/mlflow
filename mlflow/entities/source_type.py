class SourceType:
    """Enum for originating source of a :py:class:`mlflow.entities.Run`."""

    NOTEBOOK, JOB, PROJECT, LOCAL, UNKNOWN = range(1, 6)

    _STRING_TO_SOURCETYPE = {
        "NOTEBOOK": NOTEBOOK,
        "JOB": JOB,
        "PROJECT": PROJECT,
        "LOCAL": LOCAL,
        "UNKNOWN": UNKNOWN,
    }
    SOURCETYPE_TO_STRING = {value: key for key, value in _STRING_TO_SOURCETYPE.items()}

    @staticmethod
    def from_string(status_str):
        if status_str not in SourceType._STRING_TO_SOURCETYPE:
            raise Exception(
                f"Could not get run status corresponding to string {status_str}. Valid run "
                f"status strings: {list(SourceType._STRING_TO_SOURCETYPE.keys())}"
            )
        return SourceType._STRING_TO_SOURCETYPE[status_str]

    @staticmethod
    def to_string(status):
        if status not in SourceType.SOURCETYPE_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to run status {status}. Valid run "
                f"statuses: {list(SourceType.SOURCETYPE_TO_STRING.keys())}"
            )
        return SourceType.SOURCETYPE_TO_STRING[status]

from mlflow.protos.service_pb2 import LabelingSessionItemStatus as ProtoLabelingSessionItemStatus


class LabelingSessionItemStatus:
    """Enum for status of a labeling session item."""

    PENDING = ProtoLabelingSessionItemStatus.Value("PENDING")
    IN_PROGRESS = ProtoLabelingSessionItemStatus.Value("IN_PROGRESS")
    COMPLETED = ProtoLabelingSessionItemStatus.Value("COMPLETED")
    SKIPPED = ProtoLabelingSessionItemStatus.Value("SKIPPED")

    _STRING_TO_STATUS = {
        k: ProtoLabelingSessionItemStatus.Value(k)
        for k in ProtoLabelingSessionItemStatus.keys()
    }
    _STATUS_TO_STRING = {value: key for key, value in _STRING_TO_STATUS.items()}

    @staticmethod
    def from_string(status_str):
        if status_str not in LabelingSessionItemStatus._STRING_TO_STATUS:
            raise Exception(
                f"Could not get labeling item status corresponding to string {status_str}. "
                f"Valid statuses: {list(LabelingSessionItemStatus._STRING_TO_STATUS.keys())}"
            )
        return LabelingSessionItemStatus._STRING_TO_STATUS[status_str]

    @staticmethod
    def to_string(status):
        if status not in LabelingSessionItemStatus._STATUS_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to labeling item status {status}. "
                f"Valid statuses: {list(LabelingSessionItemStatus._STATUS_TO_STRING.keys())}"
            )
        return LabelingSessionItemStatus._STATUS_TO_STRING[status]

    @staticmethod
    def all_status():
        return list(LabelingSessionItemStatus._STATUS_TO_STRING.keys())

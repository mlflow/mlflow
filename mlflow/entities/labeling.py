from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.labeling_item_status import LabelingSessionItemStatus
from mlflow.protos.service_pb2 import (
    LabelingSchemaProto,
    LabelingSessionItemProto,
    LabelingSessionProto,
)


class LabelingSessionEntity(_MlflowObject):
    """
    A labeling session entity that represents a session for labeling traces or dataset records.

    Args:
        labeling_session_id: The unique identifier for the labeling session.
        experiment_id: The ID of the experiment this session belongs to.
        name: The name of the labeling session.
        creation_time: Unix timestamp (in milliseconds) when this session was created.
        last_update_time: Unix timestamp (in milliseconds) when this session was last updated.
    """

    def __init__(
        self,
        labeling_session_id: str,
        experiment_id: int,
        name: str,
        creation_time: int,
        last_update_time: int,
    ):
        self._labeling_session_id = labeling_session_id
        self._experiment_id = experiment_id
        self._name = name
        self._creation_time = creation_time
        self._last_update_time = last_update_time

    @property
    def labeling_session_id(self):
        return self._labeling_session_id

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def name(self):
        return self._name

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def last_update_time(self):
        return self._last_update_time

    @classmethod
    def from_proto(cls, proto):
        return cls(
            labeling_session_id=proto.labeling_session_id,
            experiment_id=proto.experiment_id,
            name=proto.name,
            creation_time=proto.creation_time,
            last_update_time=proto.last_update_time,
        )

    def to_proto(self):
        proto = LabelingSessionProto()
        proto.labeling_session_id = self.labeling_session_id
        proto.experiment_id = int(self.experiment_id)
        proto.name = self.name
        proto.creation_time = self.creation_time
        proto.last_update_time = self.last_update_time
        return proto

    def __repr__(self):
        return (
            f"<LabelingSessionEntity(labeling_session_id='{self.labeling_session_id}', "
            f"experiment_id={self.experiment_id}, "
            f"name='{self.name}')>"
        )


class LabelingSchema(_MlflowObject):
    """
    A labeling schema entity that defines the structure for labeling assessments.

    Args:
        labeling_schema_id: The unique identifier for the labeling schema.
        labeling_session_id: The ID of the labeling session this schema belongs to.
        name: The name of the labeling schema.
        assessment_type: The type of assessment (feedback/expectation).
        assessment_value_type: JSON string containing value type and configuration.
        title: The display title for the schema.
        instructions: Optional instructions for using this schema.
        creation_time: Unix timestamp (in milliseconds) when this schema was created.
        last_update_time: Unix timestamp (in milliseconds) when this schema was last updated.
    """

    def __init__(
        self,
        labeling_schema_id: str,
        labeling_session_id: str,
        name: str,
        assessment_type: str,
        assessment_value_type: str,
        title: str,
        instructions: str | None,
        creation_time: int,
        last_update_time: int,
    ):
        self._labeling_schema_id = labeling_schema_id
        self._labeling_session_id = labeling_session_id
        self._name = name
        self._assessment_type = assessment_type
        self._assessment_value_type = assessment_value_type
        self._title = title
        self._instructions = instructions
        self._creation_time = creation_time
        self._last_update_time = last_update_time

    @property
    def labeling_schema_id(self):
        return self._labeling_schema_id

    @property
    def labeling_session_id(self):
        return self._labeling_session_id

    @property
    def name(self):
        return self._name

    @property
    def assessment_type(self):
        return self._assessment_type

    @property
    def assessment_value_type(self):
        return self._assessment_value_type

    @property
    def title(self):
        return self._title

    @property
    def instructions(self):
        return self._instructions

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def last_update_time(self):
        return self._last_update_time

    @classmethod
    def from_proto(cls, proto):
        return cls(
            labeling_schema_id=proto.labeling_schema_id,
            labeling_session_id=proto.labeling_session_id,
            name=proto.name,
            assessment_type=proto.assessment_type,
            assessment_value_type=proto.assessment_value_type,
            title=proto.title,
            instructions=proto.instructions if proto.HasField("instructions") else None,
            creation_time=proto.creation_time,
            last_update_time=proto.last_update_time,
        )

    def to_proto(self):
        proto = LabelingSchemaProto()
        proto.labeling_schema_id = self.labeling_schema_id
        proto.labeling_session_id = self.labeling_session_id
        proto.name = self.name
        proto.assessment_type = self.assessment_type
        proto.assessment_value_type = self.assessment_value_type
        proto.title = self.title
        if self.instructions is not None:
            proto.instructions = self.instructions
        proto.creation_time = self.creation_time
        proto.last_update_time = self.last_update_time
        return proto

    def __repr__(self):
        return (
            f"<LabelingSchema(labeling_schema_id='{self.labeling_schema_id}', "
            f"labeling_session_id='{self.labeling_session_id}', "
            f"name='{self.name}')>"
        )


class LabelingSessionItem(_MlflowObject):
    """
    A labeling session item entity that represents an item to be labeled in a session.

    Args:
        labeling_item_id: The unique identifier for the labeling item.
        labeling_session_id: The ID of the labeling session this item belongs to.
        trace_id: Optional trace ID associated with this item.
        dataset_record_id: Optional dataset record ID associated with this item.
        dataset_id: Optional dataset ID associated with this item.
        status: The status of the labeling item (PENDING, IN_PROGRESS, COMPLETED, SKIPPED).
        creation_time: Unix timestamp (in milliseconds) when this item was created.
        last_update_time: Unix timestamp (in milliseconds) when this item was last updated.
    """

    def __init__(
        self,
        labeling_item_id: str,
        labeling_session_id: str,
        trace_id: str | None,
        dataset_record_id: str | None,
        dataset_id: str | None,
        status: int,
        creation_time: int,
        last_update_time: int,
    ):
        self._labeling_item_id = labeling_item_id
        self._labeling_session_id = labeling_session_id
        self._trace_id = trace_id
        self._dataset_record_id = dataset_record_id
        self._dataset_id = dataset_id
        self._status = status
        self._creation_time = creation_time
        self._last_update_time = last_update_time

    @property
    def labeling_item_id(self):
        return self._labeling_item_id

    @property
    def labeling_session_id(self):
        return self._labeling_session_id

    @property
    def trace_id(self):
        return self._trace_id

    @property
    def dataset_record_id(self):
        return self._dataset_record_id

    @property
    def dataset_id(self):
        return self._dataset_id

    @property
    def status(self):
        return self._status

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def last_update_time(self):
        return self._last_update_time

    @classmethod
    def from_proto(cls, proto):
        return cls(
            labeling_item_id=proto.labeling_item_id,
            labeling_session_id=proto.labeling_session_id,
            trace_id=proto.trace_id if proto.HasField("trace_id") else None,
            dataset_record_id=(
                proto.dataset_record_id if proto.HasField("dataset_record_id") else None
            ),
            dataset_id=proto.dataset_id if proto.HasField("dataset_id") else None,
            status=proto.status,
            creation_time=proto.creation_time,
            last_update_time=proto.last_update_time,
        )

    def to_proto(self):
        proto = LabelingSessionItemProto()
        proto.labeling_item_id = self.labeling_item_id
        proto.labeling_session_id = self.labeling_session_id
        if self.trace_id is not None:
            proto.trace_id = self.trace_id
        if self.dataset_record_id is not None:
            proto.dataset_record_id = self.dataset_record_id
        if self.dataset_id is not None:
            proto.dataset_id = self.dataset_id
        proto.status = self.status
        proto.creation_time = self.creation_time
        proto.last_update_time = self.last_update_time
        return proto

    def __repr__(self):
        return (
            f"<LabelingSessionItem(labeling_item_id='{self.labeling_item_id}', "
            f"labeling_session_id='{self.labeling_session_id}', "
            f"status={LabelingSessionItemStatus.to_string(self.status)})>"
        )

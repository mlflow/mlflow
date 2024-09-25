from typing import Any, Dict, List, Optional, Union

from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.metric import Metric
from mlflow.entities.model_param import ModelParam
from mlflow.entities.model_status import ModelStatus
from mlflow.entities.model_tag import ModelTag


class LoggedModel(_MlflowObject):
    """
    MLflow entity representing a Model logged to an MLflow Experiment.
    """

    def __init__(
        self,
        experiment_id: str,
        model_id: str,
        name: str,
        artifact_location: str,
        creation_timestamp: int,
        last_updated_timestamp: int,
        model_type: Optional[str] = None,
        run_id: Optional[str] = None,
        status: ModelStatus = ModelStatus.READY,
        status_message: Optional[str] = None,
        tags: Optional[Union[List[ModelTag], Dict[str, str]]] = None,
        params: Optional[Union[List[ModelParam], Dict[str, str]]] = None,
        metrics: Optional[List[Metric]] = None,
    ):
        super().__init__()
        self._experiment_id: str = experiment_id
        self._model_id: str = model_id
        self._name: str = name
        self._artifact_location: str = artifact_location
        self._creation_time: int = creation_timestamp
        self._last_updated_timestamp: int = last_updated_timestamp
        self._model_type: Optional[str] = model_type
        self._run_id: Optional[str] = run_id
        self._status: ModelStatus = status
        self._status_message: Optional[str] = status_message
        self._tags: Dict[str, str] = (
            {tag.key: tag.value for tag in (tags or [])} if isinstance(tags, list) else (tags or {})
        )
        self._params: Dict[str, str] = (
            {param.key: param.value for param in (params or [])}
            if isinstance(params, list)
            else (params or {})
        )
        self._metrics: Optional[List[Metric]] = metrics

    @property
    def experiment_id(self) -> str:
        """String. Experiment ID associated with this Model."""
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, new_experiment_id: str):
        self._experiment_id = new_experiment_id

    @property
    def model_id(self) -> str:
        """String. Unique ID for this Model."""
        return self._model_id

    @model_id.setter
    def model_id(self, new_model_id: str):
        self._model_id = new_model_id

    @property
    def name(self) -> str:
        """String. Name for this Model."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def artifact_location(self) -> str:
        """String. Location of the model artifacts."""
        return self._artifact_location

    @artifact_location.setter
    def artifact_location(self, new_artifact_location: str):
        self._artifact_location = new_artifact_location

    @property
    def creation_timestamp(self) -> int:
        """Integer. Model creation timestamp (milliseconds since the Unix epoch)."""
        return self._creation_time

    @property
    def last_updated_timestamp(self) -> int:
        """Integer. Timestamp of last update for this Model (milliseconds since the Unix
        epoch).
        """
        return self._last_updated_timestamp

    @last_updated_timestamp.setter
    def last_updated_timestamp(self, updated_timestamp: int):
        self._last_updated_timestamp = updated_timestamp

    @property
    def model_type(self) -> Optional[str]:
        """String. Type of the model."""
        return self._model_type

    @model_type.setter
    def model_type(self, new_model_type: Optional[str]):
        self._model_type = new_model_type

    @property
    def run_id(self) -> Optional[str]:
        """String. MLflow run ID that generated this model."""
        return self._run_id

    @property
    def status(self) -> ModelStatus:
        """String. Current status of this Model."""
        return self._status

    @status.setter
    def status(self, updated_status: str):
        self._status = updated_status

    @property
    def status_message(self) -> Optional[str]:
        """String. Descriptive message for error status conditions."""
        return self._status_message

    @property
    def tags(self) -> Dict[str, str]:
        """Dictionary of tag key (string) -> tag value for this Model."""
        return self._tags

    @property
    def params(self) -> Dict[str, str]:
        """Model parameters."""
        return self._params

    @property
    def metrics(self) -> Optional[List[Metric]]:
        """List of metrics associated with this Model."""
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics: Optional[List[Metric]]):
        self._metrics = new_metrics

    @classmethod
    def _properties(cls) -> List[str]:
        # aggregate with base class properties since cls.__dict__ does not do it automatically
        return sorted(cls._get_properties_helper())

    def _add_tag(self, tag):
        self._tags[tag.key] = tag.value

    def to_dictionary(self) -> Dict[str, Any]:
        model_dict = dict(self)
        model_dict["status"] = str(self.status)
        return model_dict

    @classmethod
    def from_proto(cls, proto):
        return cls(
            experiment_id=proto.info.experiment_id,
            model_id=proto.info.model_id,
            name=proto.info.name,
            artifact_location=proto.info.artifact_uri,
            creation_timestamp=proto.info.creation_timestamp_ms,
            last_updated_timestamp=proto.info.last_updated_timestamp_ms,
            model_type=proto.info.model_type,
            run_id=proto.info.source_run_id,
            status=ModelStatus.from_proto(proto.info.status),
            status_message=proto.info.status_message,
            tags=[ModelTag.from_proto(tag) for tag in proto.info.tags],
            params=[ModelParam.from_proto(param) for param in proto.data.params],
        )

        # message LoggedModel {
        #   // LoggedModel attributes such as model ID, status, tags, etc.
        #   optional LoggedModelInfo info = 1;
        #   // LoggedModel params and metrics.
        #   optional LoggedModelData data = 2;
        # }

        # // A LoggedModelStatus enum value represents the status of a logged
        # // model.
        # enum LoggedModelStatus {
        #   LOGGED_MODEL_STATUS_UNSPECIFIED = 0;
        #   // The LoggedModel has been created, but the LoggedModel files are not
        #   // completely uploaded.
        #   LOGGED_MODEL_PENDING = 1;
        #   // The LoggedModel is created, and the LoggedModel files are completely uploaded.
        #   LOGGED_MODEL_READY = 2;
        #   // The LoggedModel is created, but an error occurred when uploading the
        #   // LoggedModel files such as model weights / agent code.
        #   LOGGED_MODEL_UPLOAD_FAILED = 3;
        # }

        # // A LoggedModelInfo includes logged model attributes,
        # // tags, and registration info.
        # message LoggedModelInfo {
        #   // A unique identifier for the model.
        #   optional string model_id = 1;
        #   // The ID of the experiment that owns the model.
        #   optional string experiment_id = 2;
        #   // Name of the model.
        #   optional string name = 3;

        #   // Timestamp when the model was created, in milliseconds since the UNIX epoch.
        #   optional int64 creation_timestamp_ms = 4;
        #   // Timestamp when the model was last updated, in milliseconds since the UNIX epoch
        #   optional int64 last_updated_timestamp_ms = 5;

        #   // URI of the directory where model artifacts are stored.
        #   optional string artifact_uri = 6;

        #   // Whether or not the model is ready for use.
        #   optional LoggedModelStatus status = 7;

        #   // The ID of the user or principal that created the model.
        #   optional int64 creator_id = 8;

        #   // The type of model, such as "Agent", "Classifier", "LLM".
        #   optional string model_type = 9;
        #   // Run ID of the run that created the model.
        #   optional string source_run_id = 10;
        #   // Details on the current status.
        #   optional string status_message = 11;

        #   // Mutable String key-value pairs set on the model.
        #   repeated LoggedModelTag tags = 12;

        #   // If the model has been promoted to the Model Registry, this field includes
        #   // information like the Registered Model name, Model Version number, etc.
        #   repeated LoggedModelRegistrationInfo registrations = 13;
        # }

        # // Tag for a LoggedModel.
        # message LoggedModelTag {
        #   // The tag key.
        #   optional string key = 1;
        #   // The tag value.
        #   optional string value = 2;
        # }

        # // RegistrationInfo for a LoggedModel.
        # message LoggedModelRegistrationInfo {
        #   // The name of the Registered Model to which the model has been promoted.
        #   optional string name = 1;
        #   // The version number of the promoted model.
        #   optional string version = 2;
        # }

        # // A LoggedModelData message includes logged model params and linked metrics.
        # message LoggedModelData {
        #   // Immutable String key-value pairs of the model.
        #   repeated LoggedModelParameter params = 1;

        #   // Performance metrics linked to the model.
        #   repeated Metric metrics = 2;
        # }

        # // Parameter associated with a LoggedModel.
        # message LoggedModelParameter {
        #   // Key identifying this param.
        #   optional string key = 1;

        #   // Value associated with this param.
        #   optional string value = 2;
        # }

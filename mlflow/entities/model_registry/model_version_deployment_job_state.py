from mlflow.entities.model_registry._model_registry_entity import _ModelRegistryEntity
from mlflow.entities.model_registry.model_version_deployment_job_run_state import (
    ModelVersionDeploymentJobRunState,
)
from mlflow.entities.model_registry.registered_model_deployment_job_state import (
    RegisteredModelDeploymentJobState,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionDeploymentJobState as ProtoModelVersionDeploymentJobState,
)


class ModelVersionDeploymentJobState(_ModelRegistryEntity):
    """Deployment Job state object associated with a model version."""

    def __init__(self, job_id, run_id, job_state, run_state, current_task_name):
        self._job_id = job_id
        self._run_id = run_id
        self._job_state = job_state
        self._run_state = run_state
        self._current_task_name = current_task_name

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def job_id(self):
        return self._job_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def job_state(self):
        return self._job_state

    @property
    def run_state(self):
        return self._run_state

    @property
    def current_task_name(self):
        return self._current_task_name

    @classmethod
    def from_proto(cls, proto):
        return cls(
            job_id=proto.job_id,
            run_id=proto.run_id,
            job_state=RegisteredModelDeploymentJobState.to_string(proto.job_state),
            run_state=ModelVersionDeploymentJobRunState.to_string(proto.run_state),
            current_task_name=proto.current_task_name,
        )

    def to_proto(self):
        state = ProtoModelVersionDeploymentJobState()
        if self.job_id is not None:
            state.job_id = self.job_id
        if self.run_id is not None:
            state.run_id = self.run_id
        if self.job_state is not None:
            state.job_state = RegisteredModelDeploymentJobState.from_string(self.job_state)
        if self.run_state is not None:
            state.run_state = ModelVersionDeploymentJobRunState.from_string(self.run_state)
        if self.current_task_name is not None:
            state.current_task_name = self.current_task_name
        return state

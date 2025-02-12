from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    ModelVersionDeploymentJobState as ProtoModelVersionDeploymentJobState,
)


class ModelVersionDeploymentJobRunState:
    """Enum for model version deployment state of an
    :py:class:`mlflow.entities.model_registry.ModelVersion`.
    """

    NO_VALID_DEPLOYMENT_JOB_FOUND = ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value(
        "NO_VALID_DEPLOYMENT_JOB_FOUND"
    )
    RUNNING = ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value("RUNNING")
    SUCCEEDED = ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value("SUCCEEDED")
    FAILED = ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value("FAILED")
    PENDING = ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value("PENDING")
    _STRING_TO_STATE = {
        k: ProtoModelVersionDeploymentJobState.DeploymentJobRunState.Value(k)
        for k in ProtoModelVersionDeploymentJobState.DeploymentJobRunState.keys()
    }
    _STATE_TO_STRING = {value: key for key, value in _STRING_TO_STATE.items()}

    @staticmethod
    def from_string(state_str):
        if state_str not in ModelVersionDeploymentJobRunState._STRING_TO_STATE:
            raise Exception(
                f"Could not get deployment job run state corresponding to string {state_str}. "
                f"Valid state strings: {ModelVersionDeploymentJobRunState.all_states()}"
            )
        return ModelVersionDeploymentJobRunState._STRING_TO_STATE[state_str]

    @staticmethod
    def to_string(state):
        if state not in ModelVersionDeploymentJobRunState._STATE_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to deployment job run {state}. "
                f"Valid states: {ModelVersionDeploymentJobRunState.all_states()}"
            )
        return ModelVersionDeploymentJobRunState._STATE_TO_STRING[state]

    @staticmethod
    def all_states():
        return list(ModelVersionDeploymentJobRunState._STATE_TO_STRING.keys())

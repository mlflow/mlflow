from mlflow.protos.databricks_uc_registry_messages_pb2 import DeploymentJobConnection


class RegisteredModelDeploymentJobState:
    """Enum for registered model deployment state of an
    :py:class:`mlflow.entities.model_registry.RegisteredModel`.
    """

    NOT_SET_UP = DeploymentJobConnection.State.Value("NOT_SET_UP")
    CONNECTED = DeploymentJobConnection.State.Value("CONNECTED")
    NOT_FOUND = DeploymentJobConnection.State.Value("NOT_FOUND")
    REQUIRED_PARAMETERS_CHANGED = DeploymentJobConnection.State.Value("REQUIRED_PARAMETERS_CHANGED")
    _STRING_TO_STATE = {
        k: DeploymentJobConnection.State.Value(k) for k in DeploymentJobConnection.State.keys()
    }
    _STATE_TO_STRING = {value: key for key, value in _STRING_TO_STATE.items()}

    @staticmethod
    def from_string(state_str):
        if state_str not in RegisteredModelDeploymentJobState._STRING_TO_STATE:
            raise Exception(
                f"Could not get deployment job connection state corresponding to string "
                f"{state_str}. "
                f"Valid state strings: {RegisteredModelDeploymentJobState.all_states()}"
            )
        return RegisteredModelDeploymentJobState._STRING_TO_STATE[state_str]

    @staticmethod
    def to_string(state):
        if state not in RegisteredModelDeploymentJobState._STATE_TO_STRING:
            raise Exception(
                f"Could not get string corresponding to deployment job connection {state}. "
                f"Valid states: {RegisteredModelDeploymentJobState.all_states()}"
            )
        return RegisteredModelDeploymentJobState._STATE_TO_STRING[state]

    @staticmethod
    def all_states():
        return list(RegisteredModelDeploymentJobState._STATE_TO_STRING.keys())

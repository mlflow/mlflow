from mlflow.tracking._default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.tracking.client import MlflowClient
from mlflow.utils import env

_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_EXPERIMENT_NAME_ENV_VAR = "MLFLOW_EXPERIMENT_NAME"

class DatabricksEnvironmentVariableExperimentProvider(DefaultExperimentProvider):
    def in_context(self):
        return env.get_env(_EXPERIMENT_NAME_ENV_VAR)

    def get_experiment_id(self):
      experiment_name = env.get_env(_EXPERIMENT_NAME_ENV_VAR)
      if experiment_name is not None:
          exp = MlflowClient().get_experiment_by_name(experiment_name)
          return exp.experiment_id if exp else None
      return env.get_env(_EXPERIMENT_ID_ENV_VAR)

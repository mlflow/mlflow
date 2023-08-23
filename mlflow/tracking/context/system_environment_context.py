import json

from mlflow.environment_variables import MLFLOW_RUN_CONTEXT
from mlflow.tracking.context.abstract_context import RunContextProvider

# The constant MLFLOW_RUN_CONTEXT_ENV_VAR is marked as @developer_stable
MLFLOW_RUN_CONTEXT_ENV_VAR = MLFLOW_RUN_CONTEXT.name


class SystemEnvironmentContext(RunContextProvider):
    def in_context(self):
        return MLFLOW_RUN_CONTEXT.defined

    def tags(self):
        return json.loads(MLFLOW_RUN_CONTEXT.get())

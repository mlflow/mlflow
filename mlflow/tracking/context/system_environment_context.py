import os
import json

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.annotations import developer_stable


# The constant MLFLOW_RUN_CONTEXT_ENV_VAR is marked as @developer_stable
MLFLOW_RUN_CONTEXT_ENV_VAR = "MLFLOW_RUN_CONTEXT"


@developer_stable
class SystemEnvironmentContext(RunContextProvider):
    def in_context(self):
        return MLFLOW_RUN_CONTEXT_ENV_VAR in os.environ

    def tags(self):
        return json.loads(os.environ[MLFLOW_RUN_CONTEXT_ENV_VAR])

import os
import json

from mlflow.tracking.context.abstract_context import RunContextProvider


MLFLOW_RUN_CONTEXT_ENV_VAR = "MLFLOW_RUN_CONTEXT"


class SystemEnvironmentContext(RunContextProvider):
    def in_context(self):
        return MLFLOW_RUN_CONTEXT_ENV_VAR in os.environ

    def tags(self):
        return json.loads(os.environ[MLFLOW_RUN_CONTEXT_ENV_VAR])

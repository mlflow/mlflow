from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
    MLFLOW_EXPERIMENT_SOURCE_ID,
)


class DatabricksNotebookExperimentProvider(DefaultExperimentProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_notebook()

    def get_experiment_id(self):
        return databricks_utils.get_notebook_id()


class DatabricksRepoNotebookExperimentProvider(DefaultExperimentProvider):
    _resolved_repo_notebook_experiment_id = None

    def in_context(self):
        return databricks_utils.is_in_databricks_repo_notebook()

    def get_experiment_id(self):
        if DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id:
            return DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id

        source_notebook_id = databricks_utils.get_notebook_id()
        source_notebook_name = databricks_utils.get_notebook_path()
        tags = {
            MLFLOW_EXPERIMENT_SOURCE_TYPE: "REPO_NOTEBOOK",
            MLFLOW_EXPERIMENT_SOURCE_ID: source_notebook_id,
        }

        # With the presence of the above tags, the following is a get or create in which it will
        # return the corresponding experiment if one exists for the repo notebook.
        # If no corresponding experiment exist, it will create a new one and return
        # the newly created experiment ID.
        try:
            experiment_id = MlflowClient().create_experiment(source_notebook_name, None, tags)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.INVALID_PARAMETER_VALUE
            ):
                # If repo notebook experiment creation isn't enabled, fall back to
                # using the notebook ID
                experiment_id = source_notebook_id
            else:
                raise e

        DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id = (
            experiment_id
        )

        return experiment_id

from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.utils import databricks_utils


class DatabricksNotebookExperimentProvider(DefaultExperimentProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_notebook()

    def get_experiment_id(self):
        return databricks_utils.get_notebook_id()

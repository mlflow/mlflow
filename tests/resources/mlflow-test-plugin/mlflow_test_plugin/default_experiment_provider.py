from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider


class PluginDefaultExperimentProvider(DefaultExperimentProvider):
    """DefaultExperimentProvider provided through plugin system"""

    def in_context(self):
        return False

    def get_experiment_id(self):
        return "experiment_id_1"

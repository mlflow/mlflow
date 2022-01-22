import entrypoints
import warnings
import logging

from mlflow.tracking._default_experiment.databricks_notebook_context import (
    DatabricksNotebookExperimentContext,
)
from mlflow.tracking._default_experiment.databricks_job_context import (
    DatabricksJobExperimentContext,
)


_logger = logging.getLogger(__name__)


class DefaultExperimentProviderRegistry(object):
    """Registry for default experiment provider implementations

    This class allows the registration of a default experiment provider
    which can be used to infer meta information about the context of an
    MLflow experiment. Implementations declared though the
    entrypoints `mlflow.default_experiment_provider` group can be automatically
    registered through the `register_entrypoints` method.
    """

    def __init__(self):
        self._registry = []

    def register(self, default_experiment_provider_cls):
        self._registry.append(default_experiment_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in entrypoints.get_group_all("mlflow.default_experiment_provider"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    "Failure attempting to register default experiment"
                    + 'context provider "{}": {}'.format(entrypoint.name, str(exc)),
                    stacklevel=2,
                )

    def __iter__(self):
        return iter(self._registry)


_default_experiment_provider_registry = DefaultExperimentProviderRegistry()
_default_experiment_provider_registry.register(DatabricksNotebookExperimentContext)
_default_experiment_provider_registry.register(DatabricksJobExperimentContext)

_default_experiment_provider_registry.register_entrypoints()


def get_experiment_id():
    """get experiment_id for the current run context. experiment_id is fetched in the order,
    contexts are registered.

    This function iterates through all default experiment context providers in the registry.

    :return: An experiment_id for the run.
    """

    # TODO: Replace with None for 1.0, leaving for 0.9.1 release backcompat with existing servers
    deprecated_default_exp_id = "0"
    experiment_id = deprecated_default_exp_id
    for provider in _default_experiment_provider_registry:
        try:
            if provider.in_context():
                experiment_id = provider.get_experiment_id()
        except Exception as e:
            _logger.warning("Encountered unexpected error while getting experiment_id: %s", e)

    return experiment_id

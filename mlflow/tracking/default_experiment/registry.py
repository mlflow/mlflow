import logging
import warnings

from mlflow.entities import Experiment
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.tracking import get_tracking_uri
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.default_experiment.databricks_notebook_experiment_provider import (
    DatabricksNotebookExperimentProvider,
)
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)
# Listed below are the list of providers, which are used to provide MLflow Experiment IDs based on
# the current context where the MLflow client is running when the user has not explicitly set
# an experiment. The order below is the order in which the these providers are registered.
_EXPERIMENT_PROVIDERS = (DatabricksNotebookExperimentProvider,)


class DefaultExperimentProviderRegistry:
    """Registry for default experiment provider implementations

    This class allows the registration of default experiment providers, which are used to provide
    MLflow Experiment IDs based on the current context where the MLflow client is running when
    the user has not explicitly set an experiment. Implementations declared though the entrypoints
    `mlflow.default_experiment_provider` group can be automatically registered through the
    `register_entrypoints` method.
    """

    def __init__(self):
        self._registry = []

    def register(self, default_experiment_provider_cls):
        self._registry.append(default_experiment_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in get_entry_points("mlflow.default_experiment_provider"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    "Failure attempting to register default experiment"
                    + f'context provider "{entrypoint.name}": {exc}',
                    stacklevel=2,
                )

    def __iter__(self):
        return iter(self._registry)


_default_experiment_provider_registry = DefaultExperimentProviderRegistry()
for exp_provider in _EXPERIMENT_PROVIDERS:
    _default_experiment_provider_registry.register(exp_provider)

_default_experiment_provider_registry.register_entrypoints()


def get_experiment_id() -> str | None:
    """Get an experiment ID for the current context.

    The experiment ID is fetched by querying providers, in the order that they were registered.
    This function iterates through all default experiment context providers in the registry.

    Returns:
        An experiment_id.
    """
    for provider in _default_experiment_provider_registry:
        try:
            if provider.in_context():
                return provider.get_experiment_id()
        except Exception as e:
            _logger.warning("Encountered unexpected error while getting experiment_id: %s", e)

    tracking_uri = get_tracking_uri()
    is_db_uri = is_databricks_uri(tracking_uri)

    if MLFLOW_ENABLE_WORKSPACES.get() and not is_db_uri:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(Experiment.DEFAULT_EXPERIMENT_NAME)

        if experiment is None:
            try:
                experiment_id = client.create_experiment(Experiment.DEFAULT_EXPERIMENT_NAME)
                experiment = client.get_experiment(experiment_id)
            except MlflowException as exc:
                if exc.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                    raise
                experiment = client.get_experiment_by_name(Experiment.DEFAULT_EXPERIMENT_NAME)

        if experiment is not None:
            return experiment.experiment_id

        raise MlflowException(
            "Default experiment could not be resolved for the active workspace. "
            "Call mlflow.set_workspace() and ensure the workspace contains or permits creation of "
            f"an experiment named '{Experiment.DEFAULT_EXPERIMENT_NAME}'.",
            error_code=INVALID_STATE,
        )

    return DEFAULT_EXPERIMENT_ID if not is_db_uri else None

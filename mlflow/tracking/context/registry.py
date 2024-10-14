import logging
import warnings
from typing import List, Optional

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext
from mlflow.tracking.context.databricks_command_context import DatabricksCommandRunContext
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_repo_context import DatabricksRepoRunContext
from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.system_environment_context import SystemEnvironmentContext
from mlflow.utils.plugins import get_entry_points

_logger = logging.getLogger(__name__)


class RunContextProviderRegistry:
    """Registry for run context provider implementations

    This class allows the registration of a run context provider which can be used to infer meta
    information about the context of an MLflow experiment run. Implementations declared though the
    entrypoints `mlflow.run_context_provider` group can be automatically registered through the
    `register_entrypoints` method.

    Registered run context providers can return tags that override those implemented in the core
    library, however the order in which plugins are resolved is undefined.
    """

    def __init__(self):
        self._registry = []

    def register(self, run_context_provider_cls):
        self._registry.append(run_context_provider_cls())

    def register_entrypoints(self):
        """Register tracking stores provided by other packages"""
        for entrypoint in get_entry_points("mlflow.run_context_provider"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register context provider "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def __iter__(self):
        return iter(self._registry)


_run_context_provider_registry = RunContextProviderRegistry()
_run_context_provider_registry.register(DefaultRunContext)
_run_context_provider_registry.register(GitRunContext)
_run_context_provider_registry.register(DatabricksNotebookRunContext)
_run_context_provider_registry.register(DatabricksJobRunContext)
_run_context_provider_registry.register(DatabricksClusterRunContext)
_run_context_provider_registry.register(DatabricksCommandRunContext)
_run_context_provider_registry.register(DatabricksRepoRunContext)
_run_context_provider_registry.register(SystemEnvironmentContext)

_run_context_provider_registry.register_entrypoints()


def resolve_tags(tags=None, ignore: Optional[List[RunContextProvider]] = None):
    """Generate a set of tags for the current run context. Tags are resolved in the order,
    contexts are registered. Argument tags are applied last.

    This function iterates through all run context providers in the registry. Additional context
    providers can be registered as described in
    :py:class:`mlflow.tracking.context.RunContextProvider`.

    Args:
        tags: A dictionary of tags to override. If specified, tags passed in this argument will
            override those inferred from the context.
        ignore: A list of RunContextProvider classes to exclude from the resolution.

    Returns:
        A dictionary of resolved tags.
    """
    ignore = ignore or []
    all_tags = {}
    for provider in _run_context_provider_registry:
        if any(isinstance(provider, ig) for ig in ignore):
            continue

        try:
            if provider.in_context():
                all_tags.update(provider.tags())
        except Exception as e:
            _logger.warning("Encountered unexpected error during resolving tags: %s", e)

    if tags is not None:
        all_tags.update(tags)

    return all_tags

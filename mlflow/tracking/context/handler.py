from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext


class RunContextHandler


def resolve_tags(tags=None):
    """Generate a set of tags for the current run context. Tags are resolved in the order,
    contexts are registered. Argument tags are applied last.

    This function iterates through all run context providers in the registry. Additional context
    providers can be registered as described in
    :py:class:`mlflow.tracking.context.RunContextProvider`.

    :param tags: A dictionary of tags to override. If specified, tags passed in this argument will
                 override those inferred from the context.
    :return: A dicitonary of resolved tags.
    """

    all_tags = {}
    for provider in _run_context_provider_registry:
        if provider.in_context():
            # TODO: Error out gracefully if provider's tags are not valid or have wrong types.
            all_tags.update(provider.tags())

    if tags is not None:
        all_tags.update(tags)

    return all_tags


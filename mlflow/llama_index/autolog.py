from mlflow.llama_index.constant import FLAVOR_NAME
from mlflow.utils.autologging_utils import autologging_integration


def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from LlamaIndex to MLflow. Currently, MLflow
    only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for LlamaIndex models by using. If ``False``,
            no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the LlamaIndex autologging integration. If ``False``,
            enables the LlamaIndex autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during LlamaIndex
            autologging. If ``False``, show all events and warnings.
    """
    from mlflow.llama_index.tracer import remove_llama_index_tracer, set_llama_index_tracer

    # NB: The @autologging_integration annotation is used for adding shared logic. However, one
    # caveat is that the wrapped function is NOT executed when disable=True is passed. This prevents
    # us from running cleaning up logging when autologging is turned off. To workaround this, we
    # annotate _autolog() instead of this entrypoint, and define the cleanup logic outside it.
    # TODO: since this implementation is inconsistent, explore a universal way to solve the issue.
    if log_traces and not disable:
        set_llama_index_tracer()
    else:
        remove_llama_index_tracer()

    _autolog(
        log_traces=log_traces,
        disable=disable,
        silent=silent,
    )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    """
    TODO: Implement patching logic for autologging models and artifacts.
    """

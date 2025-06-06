from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "ag2"


@experimental
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from ag2 to MLflow. Currently, MLflow
    only supports tracing for ag2 agents.

    Args:
        log_traces: If ``True``, traces are logged for AG2 agents by using runtime logging.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the AG2 autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during AG2
            autologging. If ``False``, show all events and warnings.
    """
    from autogen import runtime_logging

    from mlflow.ag2.ag2_logger import MlflowAg2Logger

    # NB: The @autologging_integration annotation is used for adding shared logic. However, one
    # caveat is that the wrapped function is NOT executed when disable=True is passed. This prevents
    # us from running cleaning up logging when autologging is turned off. To workaround this, we
    # annotate _autolog() instead of this entrypoint, and define the cleanup logic outside it.
    # TODO: since this implementation is inconsistent, explore a universal way to solve the issue.
    if log_traces and not disable:
        runtime_logging.start(logger=MlflowAg2Logger())
    else:
        runtime_logging.stop()

    _autolog(log_traces=log_traces, disable=disable, silent=silent)


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    This is a dummy function only for the purpose of adding the autologging_integration annotation.
    We cannot add the annotation directly to the autolog() function above due to the reason
    mentioned in the comment above. Note that this function MUST declare the same signature as the
    autolog(), otherwise the annotation will not work properly.
    """

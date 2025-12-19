from mlflow.haystack.autolog import setup_haystack_tracing, teardown_haystack_tracing
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "haystack"


@experimental(version="3.4.0")
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Haystack to MLflow.

    Args:
        log_traces: If ``True``, traces are logged for Haystack. If ``False``, no traces
            are collected.
        disable: If ``True``, disables the Haystack autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during
            Haystack autologging. If ``False``, show all events and warnings.
    """
    if disable or not log_traces:
        teardown_haystack_tracing()
        return

    setup_haystack_tracing()


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    This function exists solely to attach the autologging_integration decorator without
    preventing cleanup logic from running when disable=True. Do not add implementation here.
    """

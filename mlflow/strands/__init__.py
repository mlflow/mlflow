import logging

from mlflow.strands.autolog import setup_strands_tracing, teardown_strands_tracing
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "strands"
_logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enables (or disables) and configures autologging from Strands Agents SDK to MLflow.

    Args:
        log_traces: If ``True``, traces are logged for Strands Agents.
        disable: If ``True``, disables Strands autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    _autolog(log_traces=log_traces, disable=disable, silent=silent)
    if disable or not log_traces:
        teardown_strands_tracing()
    else:
        setup_strands_tracing()


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    This function exists solely to attach the autologging_integration decorator without
    preventing cleanup logic from running when disable=True. Do not add implementation here.
    """

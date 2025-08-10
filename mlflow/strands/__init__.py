import logging

from mlflow.strands.autolog import setup_strands_tracing
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration

FLAVOR_NAME = "strands"
_logger = logging.getLogger(__name__)


@experimental(version="3.0.0")
@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enables (or disables) and configures autologging from Strands Agents SDK to MLflow.

    Args:
        log_traces: If ``True``, traces are logged for Strands Agents.
        disable: If ``True``, disables Strands autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    if disable or not log_traces:
        return
    setup_strands_tracing()
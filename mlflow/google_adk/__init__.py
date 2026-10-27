import logging

from mlflow.google_adk.autolog import (
    patched_runner_init,
    setup_google_adk_tracing,
    teardown_google_adk_tracing,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "google_adk"
_logger = logging.getLogger(__name__)


@experimental(version="3.18.0")
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enables (or disables) and configures autologging from Google Agent Development Kit (ADK)
    to MLflow.

    ADK's own OpenTelemetry spans (agent runs, LLM calls, tool calls) are logged to MLflow, and
    each turn's user message and visible reply are recorded on the trace's root span, so
    request/response and session-level judges read the conversation. Runners created after this
    call are traced.

    Args:
        log_traces: If ``True``, traces are logged for ADK agents.
        disable: If ``True``, disables Google ADK autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    _autolog(log_traces=log_traces, disable=disable, silent=silent)
    if disable or not log_traces:
        teardown_google_adk_tracing()
    else:
        setup_google_adk_tracing()
        from google.adk.runners import Runner

        safe_patch(FLAVOR_NAME, Runner, "__init__", patched_runner_init)

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    This function exists solely to attach the autologging_integration decorator without
    preventing cleanup logic from running when disable=True. Do not add implementation here.
    """

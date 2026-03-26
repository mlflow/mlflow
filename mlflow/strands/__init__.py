import logging

from mlflow.strands.autolog import (
    patched_agent_call,
    setup_strands_tracing,
    teardown_strands_tracing,
)
from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import safe_patch

FLAVOR_NAME = "strands"
_logger = logging.getLogger(__name__)


def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enables (or disables) and configures autologging from Strands Agents SDK to MLflow.

    Args:
        log_traces: If ``True``, traces are logged for Strands Agents.
        disable: If ``True``, disables Strands autologging.
        silent: If ``True``, suppresses all MLflow event logs and warnings.
    """
    # _autolog must be called before safe_patch (otherwise safe_patch is a no-op).
    _autolog(log_traces=log_traces, disable=disable, silent=silent)
    if disable or not log_traces:
        teardown_strands_tracing()
    else:
        setup_strands_tracing()
        try:
            from strands import Agent

            safe_patch(FLAVOR_NAME, Agent, "__call__", patched_agent_call)
        except ImportError:
            pass

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )


def trace_agent_with_session(agent_session_manager):
    """
    Context manager to automatically trace Strands Agent calls with session tracking.

    This is a convenience wrapper around mlflow.set_session() that extracts the
    session_id from a Strands SessionManager and uses it for MLflow session tracking.

    Args:
        agent_session_manager: A Strands SessionManager instance (e.g., FileSessionManager)
            with a session_id attribute.

    Returns:
        Context manager that yields the session_id.

    Example:
        .. code-block:: python

            import mlflow
            from strands import Agent
            from strands.session.file_session_manager import FileSessionManager

            mlflow.strands.autolog()

            session_manager = FileSessionManager(session_id="user-123")
            agent = Agent(model=model, session_manager=session_manager)

            # Automatically track session in MLflow
            with mlflow.strands.trace_agent_with_session(session_manager):
                agent("First message")
                agent("Second message")
                # Both traces will have session_id="user-123"
    """
    import mlflow

    if agent_session_manager is None:
        _logger.warning(
            "trace_agent_with_session called with None session_manager. "
            "No session tracking will be applied."
        )
        return mlflow.set_session(auto_generate=False)

    if not hasattr(agent_session_manager, "session_id"):
        _logger.warning(
            f"SessionManager {agent_session_manager} does not have a 'session_id' attribute. "
            "No session tracking will be applied."
        )
        return mlflow.set_session(auto_generate=False)

    session_id = agent_session_manager.session_id
    if not session_id:
        _logger.warning("SessionManager has empty session_id. No session tracking will be applied.")
        return mlflow.set_session(auto_generate=False)

    return mlflow.set_session(session_id)


# This is required by mlflow.autolog()
autolog.integration_name = FLAVOR_NAME


@autologging_integration(FLAVOR_NAME)
def _autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    This function exists solely to attach the autologging_integration decorator without
    preventing cleanup logic from running when disable=True. Do not add implementation here.
    """

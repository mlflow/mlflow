from mlflow.telemetry.events import AutologgingEvent
from mlflow.telemetry.track import _record_event
from mlflow.typesafe.autolog import async_patched_class_call, patched_class_call
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "typesafe"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from TypeSafe AI to MLflow.
    Synchronous and asynchronous calls to the System One API are supported.

    Args:
        log_traces: If ``True``, traces are logged for TypeSafe AI System One calls.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables TypeSafe AI autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during TypeSafe AI
            autologging. If ``False``, show all events and warnings.
    """
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

    safe_patch(
        FLAVOR_NAME,
        TypeSafeClient,
        "system_one",
        patched_class_call,
    )
    safe_patch(
        FLAVOR_NAME,
        AsyncTypeSafeClient,
        "system_one",
        async_patched_class_call,
    )

    _record_event(
        AutologgingEvent, {"flavor": FLAVOR_NAME, "log_traces": log_traces, "disable": disable}
    )

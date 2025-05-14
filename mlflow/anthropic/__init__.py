from mlflow.anthropic.autolog import async_patched_class_call, patched_class_call
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "anthropic"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Anthropic to MLflow.
    Only synchronous calls and asynchnorous APIs are supported. Streaming is not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Anthropic models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Anthropic autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """
    from anthropic.resources import AsyncMessages, Messages

    safe_patch(
        FLAVOR_NAME,
        Messages,
        "create",
        patched_class_call,
    )

    safe_patch(
        FLAVOR_NAME,
        AsyncMessages,
        "create",
        async_patched_class_call,
    )

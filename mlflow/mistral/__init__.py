from mlflow.mistral.autolog import patched_class_call
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "mistral"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Mistral AI to MLflow.
    Only synchronous calls to the Text generation API are supported.
    Asynchronous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Mistral AI models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Mistral AI autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Mistral AI
            autologging. If ``False``, show all events and warnings.
    """
    from mistralai.chat import Chat

    safe_patch(
        FLAVOR_NAME,
        Chat,
        "complete",
        patched_class_call,
    )

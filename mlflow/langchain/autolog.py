import logging

from mlflow.langchain.constant import FLAVOR_NAME
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import safe_patch

logger = logging.getLogger(__name__)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Enables (or disables) and configures autologging from Langchain to MLflow.

    Args:
        disable: If ``True``, disables the Langchain autologging integration. If ``False``,
            enables the Langchain autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            langchain that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Langchain
            autologging. If ``False``, show all events and warnings during Langchain
            autologging.
        log_traces: If ``True``, traces are logged for Langchain models by using
            MlflowLangchainTracer as a callback during inference. If ``False``, no traces are
            collected during inference. Default to ``True``.
    """
    from mlflow.langchain.langchain_tracer import (
        patched_callback_manager_init,
        patched_callback_manager_merge,
        patched_runnable_sequence_batch,
    )

    try:
        from langchain_core.callbacks import BaseCallbackManager

        safe_patch(
            FLAVOR_NAME,
            BaseCallbackManager,
            "__init__",
            patched_callback_manager_init,
        )
    except Exception as e:
        logger.warning(f"Failed to enable tracing for LangChain. Error: {e}")

    # Special handlings for edge cases.
    try:
        from langchain_core.callbacks import BaseCallbackManager
        from langchain_core.runnables import RunnableSequence

        safe_patch(
            FLAVOR_NAME,
            RunnableSequence,
            "batch",
            patched_runnable_sequence_batch,
        )

        safe_patch(
            FLAVOR_NAME,
            BaseCallbackManager,
            "merge",
            patched_callback_manager_merge,
        )
    except Exception:
        logger.debug("Failed to patch RunnableSequence or BaseCallbackManager.", exc_info=True)

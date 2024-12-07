"""
The ``mlflow.groq`` module provides an API for logging and loading Groq models.
"""

import importlib.metadata

from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.groq._groq_autolog import (
    patched_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "groq"


def _get_groq_package_version():
    return importlib.metadata.version("groq")


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Groq to MLflow.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for OpenAI models. If ``False``, no traces are
            collected during inference. Default to ``True``.
        disable: If ``True``, disables the Groq autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Groq
            autologging. If ``False``, show all events and warnings.
    """

    if Version(_get_groq_package_version()) < Version("0.13.0"):
        raise MlflowException("Groq autologging is only supported for groq >= 0.13.0")

    from groq.resources.audio.transcriptions import Transcriptions
    from groq.resources.audio.translations import Translations
    from groq.resources.chat.completions import Completions as ChatCompletions
    from groq.resources.embeddings import Embeddings

    for task in (ChatCompletions, Translations, Transcriptions, Embeddings):
        safe_patch(
            FLAVOR_NAME,
            task,
            "create",
            patched_call,
        )

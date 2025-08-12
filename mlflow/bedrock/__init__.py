import logging

from mlflow.utils.autologging_utils import autologging_integration, safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "bedrock"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Amazon Bedrock to MLflow.
    Only synchronous calls are supported. Asynchronous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Bedrock models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Bedrock autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Bedrock
            autologging. If ``False``, show all events and warnings.
    """
    from botocore.client import ClientCreator

    from mlflow.bedrock._autolog import patched_create_client

    # NB: In boto3, the client class for each service is dynamically created at
    # runtime via the ClientCreator factory class. Therefore, we cannot patch
    # the service client directly, and instead patch the factory to return
    # a patched client class.
    safe_patch(FLAVOR_NAME, ClientCreator, "create_client", patched_create_client)

    # Since we patch the ClientCreator factory, it only takes effect for new client instances.
    if log_traces:
        _logger.info(
            "Enabled auto-tracing for Bedrock. Note that MLflow can only trace boto3 "
            "service clients that are created after this call. If you have already "
            "created one, please recreate the client by calling `boto3.client`."
        )

import gc
import inspect

from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "bedrock"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Amazon Bedrock to MLflow.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Bedrock models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Bedrock autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Bedrock
            autologging. If ``False``, show all events and warnings.
    """
    from botocore.client import ClientCreator

    from mlflow.bedrock._autolog import patch_bedrock_runtime_client, patched_create_client

    # NB: In boto3, the client class for each service is dynamically created at
    # runtime via the ClientCreator factory class. Therefore, we cannot patch
    # the service client directly, and instead patch the factory to return
    # a patched client class.
    safe_patch(FLAVOR_NAME, ClientCreator, "create_client", patched_create_client)

    # If the boto3 client has already been created, we need to patch them as well.
    # This is a bit hacky, but since the class is generated dynamically, we need to
    # check all existing objects to find the client class.
    for obj in gc.get_objects():
        if isinstance(obj, type) and obj.__name__ == "BedrockRuntime":
            patch_bedrock_runtime_client(obj)
            break

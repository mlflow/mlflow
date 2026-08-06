"""Store Assistant-managed vendor keys as Gateway LLM Connections."""

from mlflow.entities.gateway_endpoint import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking._tracking_service.utils import _get_store

_GATEWAY_VENDOR_MODELS = {
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3-pro",
}

_NOT_FOUND = ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


class GatewayUnsupportedError(Exception):
    """Raised when the tracking store has no AI Gateway support."""


def ensure_gateway_connection(vendor: str, api_key: str) -> str:
    """Create or rotate the Gateway resources for an Assistant vendor key."""
    if (model_name := _GATEWAY_VENDOR_MODELS.get(vendor)) is None:
        raise ValueError(f"Unknown Gateway vendor: {vendor!r}")

    name = f"mlflow-assistant-{vendor}"
    store = _get_store()

    try:
        try:
            secret = store.get_secret_info(secret_name=name)
        except MlflowException as e:
            if e.error_code != _NOT_FOUND:
                raise
            secret = store.create_gateway_secret(
                secret_name=name,
                secret_value={"api_key": api_key},
                provider=vendor,
            )
        else:
            store.update_gateway_secret(
                secret_id=secret.secret_id,
                secret_value={"api_key": api_key},
            )

        try:
            model_definition = store.get_gateway_model_definition(name=name)
        except MlflowException as e:
            if e.error_code != _NOT_FOUND:
                raise
            model_definition = store.create_gateway_model_definition(
                name=name,
                secret_id=secret.secret_id,
                provider=vendor,
                model_name=model_name,
            )

        try:
            endpoint = store.get_gateway_endpoint(name=name)
        except MlflowException as e:
            if e.error_code != _NOT_FOUND:
                raise
            endpoint = store.create_gateway_endpoint(
                name=name,
                model_configs=[
                    GatewayEndpointModelConfig(
                        model_definition_id=model_definition.model_definition_id,
                        linkage_type=GatewayModelLinkageType.PRIMARY,
                    )
                ],
            )
    except NotImplementedError as e:
        raise GatewayUnsupportedError(
            "This MLflow server's tracking backend does not support the AI Gateway. "
            "Assistant-managed LLM Connections require a database-backed tracking store."
        ) from e

    return endpoint.name

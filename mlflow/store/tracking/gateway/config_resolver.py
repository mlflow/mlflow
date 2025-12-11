"""
Server-side only configuration resolver for Gateway endpoints.

This module provides functions to retrieve decrypted endpoint configurations
for resources. These functions are privileged operations that should only be
called server-side and never exposed to clients via MlflowClient.
"""

import json

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlGatewayEndpoint,
    SqlGatewayEndpointBinding,
    SqlGatewayModelDefinition,
    SqlGatewaySecret,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig, GatewayModelConfig
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.crypto import KEKManager, _decrypt_secret


def get_resource_endpoint_configs(
    resource_type: str,
    resource_id: str,
    store: SqlAlchemyStore | None = None,
) -> list[GatewayEndpointConfig]:
    """
    Get complete endpoint configurations for a resource (server-side only).

    A resource can be bound to multiple endpoints. This returns everything
    needed to make LLM API calls: endpoint details, models, and resolved
    LiteLLM parameters. This is a privileged operation that should only be
    called server-side and never exposed to clients.

    If no store is provided, this function automatically retrieves the tracking
    store from the current MLflow configuration. It only works with SqlAlchemyStore
    backends.

    Args:
        resource_type: Type of resource (e.g., "scorer_job").
        resource_id: Unique identifier for the resource instance.
        store: Optional SqlAlchemyStore instance. If not provided, the current
            tracking store is used.

    Returns:
        List of GatewayEndpointConfig entities, each containing endpoint_id,
        endpoint_name, and list of GatewayModelConfig with resolved litellm_params
        ready to pass to litellm.completion().

    Raises:
        MlflowException: If the tracking store is not a SqlAlchemyStore,
            or if an endpoint, model definition, or secret is not found.
    """
    if store is None:
        store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise MlflowException(
            "Gateway endpoint configuration is only supported with SqlAlchemyStore backends. "
            f"Current store type: {type(store).__name__}"
        )

    with store.ManagedSessionMaker() as session:
        sql_bindings = (
            session.query(SqlGatewayEndpointBinding)
            .filter(
                SqlGatewayEndpointBinding.resource_type == resource_type,
                SqlGatewayEndpointBinding.resource_id == resource_id,
            )
            .all()
        )

        kek_manager = KEKManager()
        endpoint_configs = []

        for sql_binding in sql_bindings:
            sql_endpoint = store._get_entity_or_raise(
                session,
                SqlGatewayEndpoint,
                {"endpoint_id": sql_binding.endpoint_id},
                "GatewayEndpoint",
            )

            model_configs = []

            for sql_mapping in sql_endpoint.model_mappings:
                sql_model_def = store._get_entity_or_raise(
                    session,
                    SqlGatewayModelDefinition,
                    {"model_definition_id": sql_mapping.model_definition_id},
                    "GatewayModelDefinition",
                )

                if sql_model_def.secret_id is None:
                    continue

                sql_secret = store._get_entity_or_raise(
                    session,
                    SqlGatewaySecret,
                    {"secret_id": sql_model_def.secret_id},
                    "GatewaySecret",
                )

                # Decrypt secret (returns dict since we always store as JSON)
                secret_value = _decrypt_secret(
                    encrypted_value=sql_secret.encrypted_value,
                    wrapped_dek=sql_secret.wrapped_dek,
                    kek_manager=kek_manager,
                    secret_id=sql_secret.secret_id,
                    secret_name=sql_secret.secret_name,
                )

                # Parse auth_config
                auth_config = json.loads(sql_secret.auth_config) if sql_secret.auth_config else None

                model_configs.append(
                    GatewayModelConfig(
                        model_definition_id=sql_model_def.model_definition_id,
                        provider=sql_model_def.provider,
                        model_name=sql_model_def.model_name,
                        secret_value=secret_value,
                        auth_config=auth_config,
                    )
                )

            endpoint_configs.append(
                GatewayEndpointConfig(
                    endpoint_id=sql_endpoint.endpoint_id,
                    endpoint_name=sql_endpoint.name,
                    models=model_configs,
                )
            )

        return endpoint_configs


def get_endpoint_config(
    endpoint_id: str,
    store: SqlAlchemyStore | None = None,
) -> GatewayEndpointConfig:
    """
    Get complete endpoint configuration for a specific endpoint (server-side only).

    This returns everything needed to make LLM API calls for a specific endpoint:
    endpoint details, models, and decrypted secrets. This is a privileged operation
    that should only be called server-side and never exposed to clients.

    If no store is provided, this function automatically retrieves the tracking
    store from the current MLflow configuration. It only works with SqlAlchemyStore
    backends.

    Args:
        endpoint_id: Unique identifier for the endpoint.
        store: Optional SqlAlchemyStore instance. If not provided, the current
            tracking store is used.

    Returns:
        GatewayEndpointConfig entity containing endpoint_id, endpoint_name, and
        list of GatewayModelConfig with decrypted secret_value and auth_config.

    Raises:
        MlflowException: If the tracking store is not a SqlAlchemyStore,
            or if the endpoint, model definition, or secret is not found.
    """
    if store is None:
        store = _get_store()
    if not isinstance(store, SqlAlchemyStore):
        raise MlflowException(
            "Gateway endpoint configuration is only supported with SqlAlchemyStore backends. "
            f"Current store type: {type(store).__name__}"
        )

    with store.ManagedSessionMaker() as session:
        sql_endpoint = store._get_entity_or_raise(
            session,
            SqlGatewayEndpoint,
            {"endpoint_id": endpoint_id},
            "GatewayEndpoint",
        )

        kek_manager = KEKManager()
        model_configs = []

        for sql_mapping in sql_endpoint.model_mappings:
            sql_model_def = store._get_entity_or_raise(
                session,
                SqlGatewayModelDefinition,
                {"model_definition_id": sql_mapping.model_definition_id},
                "GatewayModelDefinition",
            )

            if sql_model_def.secret_id is None:
                continue

            sql_secret = store._get_entity_or_raise(
                session,
                SqlGatewaySecret,
                {"secret_id": sql_model_def.secret_id},
                "GatewaySecret",
            )

            decrypted_value = _decrypt_secret(
                encrypted_value=sql_secret.encrypted_value,
                wrapped_dek=sql_secret.wrapped_dek,
                kek_manager=kek_manager,
                secret_id=sql_secret.secret_id,
                secret_name=sql_secret.secret_name,
            )

            model_configs.append(
                GatewayModelConfig(
                    model_definition_id=sql_model_def.model_definition_id,
                    provider=sql_model_def.provider,
                    model_name=sql_model_def.model_name,
                    secret_value=decrypted_value,
                    auth_config=json.loads(sql_secret.auth_config)
                    if sql_secret.auth_config
                    else None,
                )
            )

        return GatewayEndpointConfig(
            endpoint_id=sql_endpoint.endpoint_id,
            endpoint_name=sql_endpoint.name,
            models=model_configs,
        )

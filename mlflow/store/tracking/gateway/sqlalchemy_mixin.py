from __future__ import annotations

import json
import os
import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewaySecretInfo,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking._secret_cache import (
    _DEFAULT_CACHE_MAX_SIZE,
    _DEFAULT_CACHE_TTL,
    SECRETS_CACHE_MAX_SIZE_ENV_VAR,
    SECRETS_CACHE_TTL_ENV_VAR,
    SecretCache,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlGatewayEndpoint,
    SqlGatewayEndpointBinding,
    SqlGatewayEndpointModelMapping,
    SqlGatewayEndpointTag,
    SqlGatewayModelDefinition,
    SqlGatewaySecret,
)
from mlflow.utils.crypto import (
    KEKManager,
    _encrypt_secret,
    _mask_secret_value,
)
from mlflow.utils.time import get_current_time_millis


def _validate_one_of(
    param1_name: str, param1_value: Any, param2_name: str, param2_value: Any
) -> None:
    """Validate that exactly one of two parameters is provided."""
    if (param1_value is None) == (param2_value is None):
        raise MlflowException(
            f"Exactly one of {param1_name} or {param2_name} must be provided",
            error_code=INVALID_PARAMETER_VALUE,
        )


class SqlAlchemyGatewayStoreMixin:
    """Mixin class providing SQLAlchemy Gateway implementations for tracking stores.

    This mixin adds Gateway functionality to SQLAlchemy-based tracking stores,
    enabling management of secrets, model definitions, endpoints, and bindings
    for the MLflow AI Gateway.

    Requires the base class to provide:
    - ManagedSessionMaker: Context manager for database sessions
    - _get_entity_or_raise: Helper method for fetching entities or raising if not found
    """

    _secret_cache: SecretCache | None = None

    @property
    def secret_cache(self) -> SecretCache:
        """Lazy-initialized secret cache for endpoint configurations."""
        if self._secret_cache is None:
            ttl = int(os.environ.get(SECRETS_CACHE_TTL_ENV_VAR, _DEFAULT_CACHE_TTL))
            max_size = int(os.environ.get(SECRETS_CACHE_MAX_SIZE_ENV_VAR, _DEFAULT_CACHE_MAX_SIZE))
            self._secret_cache = SecretCache(ttl_seconds=ttl, max_size=max_size)
        return self._secret_cache

    def _get_cache_key(self, resource_type: str, resource_id: str) -> str:
        """Generate cache key for resource endpoint configs."""
        return f"{resource_type}:{resource_id}"

    def _invalidate_secret_cache(self) -> None:
        """Clear the secret cache on mutations."""
        if self._secret_cache is not None:
            self._secret_cache.clear()

    def create_gateway_secret(
        self,
        secret_name: str,
        secret_value: dict[str, str],
        provider: str | None = None,
        auth_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> GatewaySecretInfo:
        """
        Create a new encrypted secret using envelope encryption.

        Args:
            secret_name: Unique user-friendly name for the secret.
            secret_value: The secret value(s) to encrypt as key-value pairs.
                For simple API keys: {"api_key": "sk-xxx"}
                For compound credentials: {"aws_access_key_id": "...",
                  "aws_secret_access_key": "..."}
            provider: Optional LLM provider (e.g., "openai", "anthropic").
            auth_config: Optional provider-specific auth configuration dict.
                Should include "auth_mode" for providers with multiple auth options.
            created_by: Username of the creator.

        Returns:
            Secret entity with metadata (encrypted value not included).
        """
        with self.ManagedSessionMaker() as session:
            secret_id = f"s-{uuid.uuid4().hex}"
            current_time = get_current_time_millis()

            value_to_encrypt = json.dumps(secret_value)
            masked_value = _mask_secret_value(secret_value)

            kek_manager = KEKManager()

            encrypted = _encrypt_secret(
                secret_value=value_to_encrypt,
                kek_manager=kek_manager,
                secret_id=secret_id,
                secret_name=secret_name,
            )

            sql_secret = SqlGatewaySecret(
                secret_id=secret_id,
                secret_name=secret_name,
                encrypted_value=encrypted.encrypted_value,
                wrapped_dek=encrypted.wrapped_dek,
                masked_value=json.dumps(masked_value),
                kek_version=encrypted.kek_version,
                provider=provider,
                auth_config=json.dumps(auth_config) if auth_config else None,
                created_at=current_time,
                last_updated_at=current_time,
                created_by=created_by,
                last_updated_by=created_by,
            )

            try:
                session.add(sql_secret)
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"Secret with name '{secret_name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e

            return sql_secret.to_mlflow_entity()

    def get_secret_info(
        self, secret_id: str | None = None, secret_name: str | None = None
    ) -> GatewaySecretInfo:
        """
        Retrieve secret metadata by ID or name (does not decrypt the value and only
        returns the masked secret for the purposes of key identification for users).

        Args:
            secret_id: ID of the secret to retrieve.
            secret_name: Name of the secret to retrieve.

        Returns:
            Secret entity with metadata (encrypted value not included).
        """
        _validate_one_of("secret_id", secret_id, "secret_name", secret_name)

        with self.ManagedSessionMaker() as session:
            if secret_id:
                sql_secret = self._get_entity_or_raise(
                    session, SqlGatewaySecret, {"secret_id": secret_id}, "GatewaySecret"
                )
            else:
                sql_secret = self._get_entity_or_raise(
                    session, SqlGatewaySecret, {"secret_name": secret_name}, "GatewaySecret"
                )

            return sql_secret.to_mlflow_entity()

    def update_gateway_secret(
        self,
        secret_id: str,
        secret_value: dict[str, str] | None = None,
        auth_config: dict[str, Any] | None = None,
        updated_by: str | None = None,
    ) -> GatewaySecretInfo:
        """
        Update an existing secret's configuration.

        Args:
            secret_id: ID of the secret to update.
            secret_value: Optional new secret value(s) for key rotation as key-value pairs,
                or None to leave unchanged.
                For simple API keys: {"api_key": "sk-xxx"}
                For compound credentials: {"aws_access_key_id": "...",
                  "aws_secret_access_key": "..."}
            auth_config: Optional updated auth configuration. If provided, replaces existing
                auth_config. If None, auth_config is unchanged. If empty dict, clears auth_config.
            updated_by: Username of the updater.

        Returns:
            Updated Secret entity.
        """
        with self.ManagedSessionMaker() as session:
            sql_secret = self._get_entity_or_raise(
                session, SqlGatewaySecret, {"secret_id": secret_id}, "GatewaySecret"
            )

            if secret_value is not None:
                value_to_encrypt = json.dumps(secret_value)
                masked_value = _mask_secret_value(secret_value)

                kek_manager = KEKManager()

                encrypted = _encrypt_secret(
                    secret_value=value_to_encrypt,
                    kek_manager=kek_manager,
                    secret_id=sql_secret.secret_id,
                    secret_name=sql_secret.secret_name,
                )

                sql_secret.encrypted_value = encrypted.encrypted_value
                sql_secret.wrapped_dek = encrypted.wrapped_dek
                sql_secret.kek_version = encrypted.kek_version
                sql_secret.masked_value = json.dumps(masked_value)

            if auth_config is not None:
                # Empty dict {} explicitly clears auth_config, non-empty dict replaces it
                sql_secret.auth_config = json.dumps(auth_config) if auth_config else None

            sql_secret.last_updated_by = updated_by
            sql_secret.last_updated_at = get_current_time_millis()

            session.flush()
            session.refresh(sql_secret)

            self._invalidate_secret_cache()
            return sql_secret.to_mlflow_entity()

    def delete_gateway_secret(self, secret_id: str) -> None:
        """
        Permanently delete a secret.

        Model definitions that reference this secret will become orphaned (their
        secret_id will be set to NULL). They can still be used but will need to
        be updated with a new secret before they can be used for LLM calls.

        Args:
            secret_id: ID of the secret to delete.
        """
        with self.ManagedSessionMaker() as session:
            sql_secret = self._get_entity_or_raise(
                session, SqlGatewaySecret, {"secret_id": secret_id}, "GatewaySecret"
            )

            session.delete(sql_secret)
            self._invalidate_secret_cache()

    def list_secret_infos(self, provider: str | None = None) -> list[GatewaySecretInfo]:
        """
        List all secret metadata with optional filtering.

        Args:
            provider: Optional filter by LLM provider (e.g., "openai", "anthropic").

        Returns:
            List of Secret entities with metadata (encrypted values not included).
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewaySecret)

            if provider is not None:
                query = query.filter(SqlGatewaySecret.provider == provider)

            sql_secrets = query.all()
            return [secret.to_mlflow_entity() for secret in sql_secrets]

    def create_gateway_model_definition(
        self,
        name: str,
        secret_id: str,
        provider: str,
        model_name: str,
        created_by: str | None = None,
    ) -> GatewayModelDefinition:
        """
        Create a reusable model definition.

        Args:
            name: User-friendly name for identification and reuse. Must be unique.
            secret_id: ID of the secret containing authentication credentials.
            provider: LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock").
            model_name: Provider-specific model identifier (e.g., "gpt-4o").
            created_by: Username of the creator.

        Returns:
            GatewayModelDefinition entity with metadata.
        """
        with self.ManagedSessionMaker() as session:
            sql_secret = self._get_entity_or_raise(
                session, SqlGatewaySecret, {"secret_id": secret_id}, "GatewaySecret"
            )

            model_definition_id = f"d-{uuid.uuid4().hex}"
            current_time = get_current_time_millis()

            sql_model_def = SqlGatewayModelDefinition(
                model_definition_id=model_definition_id,
                name=name,
                secret_id=secret_id,
                provider=provider,
                model_name=model_name,
                created_at=current_time,
                last_updated_at=current_time,
                created_by=created_by,
                last_updated_by=created_by,
            )

            try:
                session.add(sql_model_def)
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"Model definition with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e

            session.refresh(sql_model_def)

            return GatewayModelDefinition(
                model_definition_id=sql_model_def.model_definition_id,
                name=sql_model_def.name,
                secret_id=sql_model_def.secret_id,
                secret_name=sql_secret.secret_name,
                provider=sql_model_def.provider,
                model_name=sql_model_def.model_name,
                created_at=sql_model_def.created_at,
                last_updated_at=sql_model_def.last_updated_at,
                created_by=sql_model_def.created_by,
                last_updated_by=sql_model_def.last_updated_by,
            )

    def get_gateway_model_definition(
        self, model_definition_id: str | None = None, name: str | None = None
    ) -> GatewayModelDefinition:
        """
        Retrieve a model definition by ID or name.

        Args:
            model_definition_id: ID of the model definition to retrieve.
            name: Name of the model definition to retrieve.

        Returns:
            GatewayModelDefinition entity with metadata.
        """
        _validate_one_of("model_definition_id", model_definition_id, "name", name)

        with self.ManagedSessionMaker() as session:
            if model_definition_id:
                sql_model_def = self._get_entity_or_raise(
                    session,
                    SqlGatewayModelDefinition,
                    {"model_definition_id": model_definition_id},
                    "GatewayModelDefinition",
                )
            else:
                sql_model_def = self._get_entity_or_raise(
                    session, SqlGatewayModelDefinition, {"name": name}, "GatewayModelDefinition"
                )

            return sql_model_def.to_mlflow_entity()

    def list_gateway_model_definitions(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayModelDefinition]:
        """
        List all model definitions with optional filtering.

        Args:
            provider: Optional filter by LLM provider.
            secret_id: Optional filter by secret ID.

        Returns:
            List of GatewayModelDefinition entities with metadata.
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewayModelDefinition)

            if provider is not None:
                query = query.filter(SqlGatewayModelDefinition.provider == provider)
            if secret_id is not None:
                query = query.filter(SqlGatewayModelDefinition.secret_id == secret_id)

            sql_model_defs = query.all()
            return [model_def.to_mlflow_entity() for model_def in sql_model_defs]

    def update_gateway_model_definition(
        self,
        model_definition_id: str,
        name: str | None = None,
        secret_id: str | None = None,
        model_name: str | None = None,
        updated_by: str | None = None,
        provider: str | None = None,
    ) -> GatewayModelDefinition:
        """
        Update a model definition.

        Args:
            model_definition_id: ID of the model definition to update.
            name: Optional new name.
            secret_id: Optional new secret ID.
            model_name: Optional new model name.
            updated_by: Username of the updater.
            provider: Optional new provider.

        Returns:
            Updated GatewayModelDefinition entity.

        Raises:
            MlflowException: If the model definition or secret is not found
                (RESOURCE_DOES_NOT_EXIST), or if the new name conflicts with an existing
                model definition (RESOURCE_ALREADY_EXISTS).
        """
        with self.ManagedSessionMaker() as session:
            sql_model_def = self._get_entity_or_raise(
                session,
                SqlGatewayModelDefinition,
                {"model_definition_id": model_definition_id},
                "GatewayModelDefinition",
            )

            if name is not None:
                sql_model_def.name = name
            if secret_id is not None:
                self._get_entity_or_raise(
                    session, SqlGatewaySecret, {"secret_id": secret_id}, "GatewaySecret"
                )
                sql_model_def.secret_id = secret_id
            if model_name is not None:
                sql_model_def.model_name = model_name
            if provider is not None:
                sql_model_def.provider = provider

            sql_model_def.last_updated_at = get_current_time_millis()
            if updated_by:
                sql_model_def.last_updated_by = updated_by

            try:
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"Model definition with name '{name}' already exists",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e

            session.refresh(sql_model_def)

            self._invalidate_secret_cache()
            return sql_model_def.to_mlflow_entity()

    def delete_gateway_model_definition(self, model_definition_id: str) -> None:
        """
        Delete a model definition.

        Fails with an error if the model definition is currently attached to any
        endpoints (RESTRICT behavior enforced by database constraint).

        Args:
            model_definition_id: ID of the model definition to delete.

        Raises:
            MlflowException: If the model definition is not found (RESOURCE_DOES_NOT_EXIST),
                or if it is currently in use by endpoints (INVALID_STATE).
        """
        with self.ManagedSessionMaker() as session:
            sql_model_def = self._get_entity_or_raise(
                session,
                SqlGatewayModelDefinition,
                {"model_definition_id": model_definition_id},
                "GatewayModelDefinition",
            )

            try:
                session.delete(sql_model_def)
                session.flush()
                self._invalidate_secret_cache()
            except IntegrityError as e:
                raise MlflowException(
                    "Cannot delete model definition that is currently in use by endpoints. "
                    "Detach it from all endpoints first.",
                    error_code=INVALID_STATE,
                ) from e

    def create_gateway_endpoint(
        self,
        name: str,
        model_definition_ids: list[str],
        created_by: str | None = None,
    ) -> GatewayEndpoint:
        """
        Create a new endpoint with references to existing model definitions.

        Args:
            name: User-friendly name for the endpoint.
            model_definition_ids: List of model definition IDs to attach to the endpoint.
                                  At least one model definition is required.
            created_by: Username of the creator.

        Returns:
            Endpoint entity with model_mappings populated.

        Raises:
            MlflowException: If model_definition_ids list is empty (INVALID_PARAMETER_VALUE),
                or if any referenced model definition does not exist (RESOURCE_DOES_NOT_EXIST).
        """
        if not model_definition_ids:
            raise MlflowException(
                "Endpoint must have at least one model definition",
                error_code=INVALID_PARAMETER_VALUE,
            )

        with self.ManagedSessionMaker() as session:
            existing_model_defs = (
                session.query(SqlGatewayModelDefinition.model_definition_id)
                .filter(SqlGatewayModelDefinition.model_definition_id.in_(model_definition_ids))
                .all()
            )
            existing_ids = {m.model_definition_id for m in existing_model_defs}
            if missing := set(model_definition_ids) - existing_ids:
                raise MlflowException(
                    f"Model definitions not found: {', '.join(missing)}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            endpoint_id = f"e-{uuid.uuid4().hex}"
            current_time = get_current_time_millis()

            sql_endpoint = SqlGatewayEndpoint(
                endpoint_id=endpoint_id,
                name=name,
                created_at=current_time,
                last_updated_at=current_time,
                created_by=created_by,
                last_updated_by=created_by,
            )
            session.add(sql_endpoint)

            for model_def_id in model_definition_ids:
                mapping_id = f"m-{uuid.uuid4().hex}"
                sql_mapping = SqlGatewayEndpointModelMapping(
                    mapping_id=mapping_id,
                    endpoint_id=endpoint_id,
                    model_definition_id=model_def_id,
                    weight=1,
                    created_at=current_time,
                    created_by=created_by,
                )
                session.add(sql_mapping)

            session.flush()
            session.refresh(sql_endpoint)

            return sql_endpoint.to_mlflow_entity()

    def get_gateway_endpoint(
        self, endpoint_id: str | None = None, name: str | None = None
    ) -> GatewayEndpoint:
        """
        Retrieve an endpoint by ID or name with its model mappings populated.

        Args:
            endpoint_id: ID of the endpoint to retrieve.
            name: Name of the endpoint to retrieve.

        Returns:
            Endpoint entity with model_mappings list populated.

        Raises:
            MlflowException: If exactly one of endpoint_id or name is not provided
                (INVALID_PARAMETER_VALUE), or if the endpoint is not found
                (RESOURCE_DOES_NOT_EXIST).
        """
        _validate_one_of("endpoint_id", endpoint_id, "name", name)

        with self.ManagedSessionMaker() as session:
            if endpoint_id:
                sql_endpoint = self._get_entity_or_raise(
                    session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
                )
            else:
                sql_endpoint = self._get_entity_or_raise(
                    session, SqlGatewayEndpoint, {"name": name}, "GatewayEndpoint"
                )

            return sql_endpoint.to_mlflow_entity()

    def update_gateway_endpoint(
        self,
        endpoint_id: str,
        name: str,
        updated_by: str | None = None,
    ) -> GatewayEndpoint:
        """
        Update an endpoint's name.

        Args:
            endpoint_id: ID of the endpoint to update.
            name: New name for the endpoint.
            updated_by: Username of the updater.

        Returns:
            Updated Endpoint entity.
        """
        with self.ManagedSessionMaker() as session:
            sql_endpoint = self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )

            sql_endpoint.name = name
            sql_endpoint.last_updated_at = get_current_time_millis()
            if updated_by:
                sql_endpoint.last_updated_by = updated_by

            session.flush()
            session.refresh(sql_endpoint)

            self._invalidate_secret_cache()
            return sql_endpoint.to_mlflow_entity()

    def delete_gateway_endpoint(self, endpoint_id: str) -> None:
        """
        Delete an endpoint (CASCADE deletes bindings and model mappings).

        Args:
            endpoint_id: ID of the endpoint to delete.
        """
        with self.ManagedSessionMaker() as session:
            sql_endpoint = self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )

            session.delete(sql_endpoint)
            self._invalidate_secret_cache()

    def list_gateway_endpoints(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayEndpoint]:
        """
        List all endpoints with their model mappings populated.

        Args:
            provider: Optional filter by LLM provider (e.g., "openai", "anthropic").
                Returns only endpoints that have at least one model from this provider.
            secret_id: Optional filter by secret ID. Returns only endpoints using this secret.
                Useful for showing which endpoints would be affected by secret deletion.

        Returns:
            List of Endpoint entities with model_mappings.
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewayEndpoint).join(SqlGatewayEndpointModelMapping)

            if provider or secret_id:
                query = query.join(
                    SqlGatewayModelDefinition,
                    SqlGatewayEndpointModelMapping.model_definition_id
                    == SqlGatewayModelDefinition.model_definition_id,
                )

                if provider:
                    query = query.filter(SqlGatewayModelDefinition.provider == provider)
                if secret_id:
                    query = query.filter(SqlGatewayModelDefinition.secret_id == secret_id)

            endpoints = query.distinct().all()
            return [endpoint.to_mlflow_entity() for endpoint in endpoints]

    def attach_model_to_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
        weight: float = 1.0,
        created_by: str | None = None,
    ) -> GatewayEndpointModelMapping:
        """
        Attach an existing model definition to an endpoint.

        Args:
            endpoint_id: ID of the endpoint to attach the model to.
            model_definition_id: ID of the model definition to attach.
            weight: Routing weight for traffic distribution (default 1.0).
            created_by: Username of the creator.

        Returns:
            EndpointModelMapping entity.

        Raises:
            MlflowException: If the endpoint or model definition is not found
                (RESOURCE_DOES_NOT_EXIST), or if the model definition is already
                attached to this endpoint (RESOURCE_ALREADY_EXISTS).
        """
        with self.ManagedSessionMaker() as session:
            sql_endpoint = self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )
            self._get_entity_or_raise(
                session,
                SqlGatewayModelDefinition,
                {"model_definition_id": model_definition_id},
                "GatewayModelDefinition",
            )

            mapping_id = f"m-{uuid.uuid4().hex}"
            current_time = get_current_time_millis()

            sql_mapping = SqlGatewayEndpointModelMapping(
                mapping_id=mapping_id,
                endpoint_id=endpoint_id,
                model_definition_id=model_definition_id,
                weight=weight,
                created_at=current_time,
                created_by=created_by,
            )

            sql_endpoint.last_updated_at = current_time
            if created_by:
                sql_endpoint.last_updated_by = created_by

            try:
                session.add(sql_mapping)
                session.flush()
            except IntegrityError as e:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is already attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from e

            session.refresh(sql_mapping)

            self._invalidate_secret_cache()
            return sql_mapping.to_mlflow_entity()

    def detach_model_from_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
    ) -> None:
        """
        Detach a model definition from an endpoint.

        This removes the mapping but does not delete the model definition itself.

        Args:
            endpoint_id: ID of the endpoint.
            model_definition_id: ID of the model definition to detach.

        Raises:
            MlflowException: If the mapping is not found (RESOURCE_DOES_NOT_EXIST).
        """
        with self.ManagedSessionMaker() as session:
            sql_mapping = (
                session.query(SqlGatewayEndpointModelMapping)
                .filter(
                    SqlGatewayEndpointModelMapping.endpoint_id == endpoint_id,
                    SqlGatewayEndpointModelMapping.model_definition_id == model_definition_id,
                )
                .first()
            )
            if not sql_mapping:
                raise MlflowException(
                    f"Model definition '{model_definition_id}' is not attached to "
                    f"endpoint '{endpoint_id}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )

            session.delete(sql_mapping)
            self._invalidate_secret_cache()

    def create_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: str,
        resource_id: str,
        created_by: str | None = None,
    ) -> GatewayEndpointBinding:
        """
        Bind an endpoint to an MLflow resource.

        Args:
            endpoint_id: ID of the endpoint to bind.
            resource_type: Type of resource (e.g., "scorer_job").
            resource_id: Unique identifier for the resource instance.
            created_by: Username of the creator.

        Returns:
            GatewayEndpointBinding entity.

        Raises:
            MlflowException: If the endpoint is not found (RESOURCE_DOES_NOT_EXIST).
        """
        with self.ManagedSessionMaker() as session:
            self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )

            current_time = get_current_time_millis()

            sql_binding = SqlGatewayEndpointBinding(
                endpoint_id=endpoint_id,
                resource_type=resource_type,
                resource_id=resource_id,
                created_at=current_time,
                last_updated_at=current_time,
                created_by=created_by,
                last_updated_by=created_by,
            )

            session.add(sql_binding)
            session.flush()
            session.refresh(sql_binding)

            self._invalidate_secret_cache()
            return sql_binding.to_mlflow_entity()

    def delete_endpoint_binding(
        self, endpoint_id: str, resource_type: str, resource_id: str
    ) -> None:
        """
        Delete an endpoint binding.

        Args:
            endpoint_id: ID of the endpoint.
            resource_type: Type of resource bound to the endpoint.
            resource_id: ID of the resource.

        Raises:
            MlflowException: If the binding is not found (RESOURCE_DOES_NOT_EXIST).
        """
        with self.ManagedSessionMaker() as session:
            sql_binding = self._get_entity_or_raise(
                session,
                SqlGatewayEndpointBinding,
                {
                    "endpoint_id": endpoint_id,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                },
                "GatewayEndpointBinding",
            )

            session.delete(sql_binding)
            self._invalidate_secret_cache()

    def list_endpoint_bindings(
        self,
        endpoint_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[GatewayEndpointBinding]:
        """
        List endpoint bindings with optional filtering.

        Args:
            endpoint_id: Optional filter by endpoint ID.
            resource_type: Optional filter by resource type.
            resource_id: Optional filter by resource ID.

        Returns:
            List of GatewayEndpointBinding entities (with endpoint_name and
            model_mappings populated).
        """
        with self.ManagedSessionMaker() as session:
            query = session.query(SqlGatewayEndpointBinding).options(
                joinedload(SqlGatewayEndpointBinding.endpoint).joinedload(
                    SqlGatewayEndpoint.model_mappings
                )
            )

            if endpoint_id is not None:
                query = query.filter(SqlGatewayEndpointBinding.endpoint_id == endpoint_id)
            if resource_type is not None:
                query = query.filter(SqlGatewayEndpointBinding.resource_type == resource_type)
            if resource_id is not None:
                query = query.filter(SqlGatewayEndpointBinding.resource_id == resource_id)

            bindings = query.all()
            return [binding.to_mlflow_entity() for binding in bindings]

    def set_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        tag: GatewayEndpointTag,
    ) -> None:
        """
        Set a tag on an endpoint.

        Args:
            endpoint_id: ID of the endpoint to tag.
            tag: GatewayEndpointTag with key and value to set.
        """
        with self.ManagedSessionMaker() as session:
            self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )
            session.merge(
                SqlGatewayEndpointTag(
                    endpoint_id=endpoint_id,
                    key=tag.key,
                    value=tag.value,
                )
            )

    def delete_gateway_endpoint_tag(
        self,
        endpoint_id: str,
        key: str,
    ) -> None:
        """
        Delete a tag from an endpoint.

        Args:
            endpoint_id: ID of the endpoint.
            key: Tag key to delete.
        """
        with self.ManagedSessionMaker() as session:
            self._get_entity_or_raise(
                session, SqlGatewayEndpoint, {"endpoint_id": endpoint_id}, "GatewayEndpoint"
            )
            session.query(SqlGatewayEndpointTag).filter(
                SqlGatewayEndpointTag.endpoint_id == endpoint_id,
                SqlGatewayEndpointTag.key == key,
            ).delete()

"""REST Gateway Store Mixin - Gateway API implementation for REST-based tracking stores."""

from __future__ import annotations

import json
from typing import Any

from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayEndpointTag,
    GatewayModelDefinition,
    GatewayResourceType,
    GatewaySecretInfo,
)
from mlflow.protos.service_pb2 import (
    AttachModelToGatewayEndpoint,
    CreateGatewayEndpoint,
    CreateGatewayEndpointBinding,
    CreateGatewayModelDefinition,
    CreateGatewaySecret,
    DeleteGatewayEndpoint,
    DeleteGatewayEndpointBinding,
    DeleteGatewayEndpointTag,
    DeleteGatewayModelDefinition,
    DeleteGatewaySecret,
    DetachModelFromGatewayEndpoint,
    GetGatewayEndpoint,
    GetGatewayModelDefinition,
    GetGatewaySecretInfo,
    ListGatewayEndpointBindings,
    ListGatewayEndpoints,
    ListGatewayModelDefinitions,
    ListGatewaySecretInfos,
    SetGatewayEndpointTag,
    UpdateGatewayEndpoint,
    UpdateGatewayModelDefinition,
    UpdateGatewaySecret,
)
from mlflow.utils.proto_json_utils import message_to_json


class RestGatewayStoreMixin:
    """Mixin class providing Gateway API implementation for REST-based tracking stores.

    This mixin adds Gateway functionality to REST tracking stores, enabling
    management of secrets, model definitions, endpoints, and bindings
    for the MLflow AI Gateway via REST API calls.

    The mixin expects the implementing class to provide:
    - _call_endpoint(api, json_body): Method to make REST API calls
    """

    # Set of v3 Gateway APIs (secrets, endpoints, model definitions, bindings)
    _V3_GATEWAY_APIS = {
        CreateGatewaySecret,
        GetGatewaySecretInfo,
        UpdateGatewaySecret,
        DeleteGatewaySecret,
        ListGatewaySecretInfos,
        CreateGatewayEndpoint,
        GetGatewayEndpoint,
        UpdateGatewayEndpoint,
        DeleteGatewayEndpoint,
        ListGatewayEndpoints,
        CreateGatewayModelDefinition,
        GetGatewayModelDefinition,
        ListGatewayModelDefinitions,
        UpdateGatewayModelDefinition,
        DeleteGatewayModelDefinition,
        AttachModelToGatewayEndpoint,
        DetachModelFromGatewayEndpoint,
        CreateGatewayEndpointBinding,
        DeleteGatewayEndpointBinding,
        ListGatewayEndpointBindings,
        SetGatewayEndpointTag,
        DeleteGatewayEndpointTag,
    }

    # ========== Secrets Management APIs ==========

    def create_gateway_secret(
        self,
        secret_name: str,
        secret_value: dict[str, str],
        provider: str | None = None,
        auth_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> GatewaySecretInfo:
        """
        Create a new secret for secure credential storage.

        Args:
            secret_name: Name to identify the secret.
            secret_value: The secret value(s) to encrypt and store as key-value pairs.
                For simple API keys: {"api_key": "sk-xxx"}
                For compound credentials: {"aws_access_key_id": "...",
                  "aws_secret_access_key": "..."}
            provider: Optional provider name (e.g., "openai", "anthropic").
            auth_config: Optional dict with authentication configuration. For providers
                with multiple auth modes, include "auth_mode" key (e.g.,
                {"auth_mode": "access_keys", "aws_region_name": "us-east-1"}).
            created_by: Optional identifier of the user creating the secret.

        Returns:
            The created GatewaySecretInfo object with masked value.
        """
        auth_config_json = json.dumps(auth_config) if auth_config is not None else None
        req_body = message_to_json(
            CreateGatewaySecret(
                secret_name=secret_name,
                secret_value=secret_value,
                provider=provider,
                auth_config_json=auth_config_json,
                created_by=created_by,
            )
        )
        response_proto = self._call_endpoint(CreateGatewaySecret, req_body)
        return GatewaySecretInfo.from_proto(response_proto.secret)

    def get_secret_info(
        self, secret_id: str | None = None, secret_name: str | None = None
    ) -> GatewaySecretInfo:
        """
        Retrieve information about a secret (value will be masked).

        Args:
            secret_id: The unique identifier of the secret.
            secret_name: The name of the secret.

        Returns:
            The GatewaySecretInfo object with masked value.
        """
        req_body = message_to_json(
            GetGatewaySecretInfo(secret_id=secret_id, secret_name=secret_name)
        )
        response_proto = self._call_endpoint(GetGatewaySecretInfo, req_body)
        return GatewaySecretInfo.from_proto(response_proto.secret)

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
            secret_id: The unique identifier of the secret to update.
            secret_value: Optional new secret value(s) for key rotation as key-value pairs,
                or None to leave unchanged.
                For simple API keys: {"api_key": "sk-xxx"}
                For compound credentials: {"aws_access_key_id": "...",
                  "aws_secret_access_key": "..."}
            auth_config: Optional dict with authentication configuration.
            updated_by: Optional identifier of the user updating the secret.

        Returns:
            The updated GatewaySecretInfo object with masked value.
        """
        auth_config_json = json.dumps(auth_config) if auth_config is not None else None
        req_body = message_to_json(
            UpdateGatewaySecret(
                secret_id=secret_id,
                secret_value=secret_value or {},
                auth_config_json=auth_config_json,
                updated_by=updated_by,
            )
        )
        response_proto = self._call_endpoint(UpdateGatewaySecret, req_body)
        return GatewaySecretInfo.from_proto(response_proto.secret)

    def delete_gateway_secret(self, secret_id: str) -> None:
        """
        Delete a secret.

        Args:
            secret_id: The unique identifier of the secret to delete.
        """
        req_body = message_to_json(DeleteGatewaySecret(secret_id=secret_id))
        self._call_endpoint(DeleteGatewaySecret, req_body)

    def list_secret_infos(self, provider: str | None = None) -> list[GatewaySecretInfo]:
        """
        List all secret metadata, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter secrets.

        Returns:
            List of GatewaySecretInfo objects with masked values.
        """
        req_body = message_to_json(ListGatewaySecretInfos(provider=provider))
        response_proto = self._call_endpoint(ListGatewaySecretInfos, req_body)
        return [GatewaySecretInfo.from_proto(s) for s in response_proto.secrets]

    # ========== Endpoints Management APIs ==========

    def create_gateway_endpoint(
        self,
        name: str,
        model_definition_ids: list[str],
        created_by: str | None = None,
    ) -> GatewayEndpoint:
        """
        Create a new endpoint with associated model definitions.

        Args:
            name: Name to identify the endpoint.
            model_definition_ids: List of model definition IDs to attach.
            created_by: Optional identifier of the user creating the endpoint.

        Returns:
            The created GatewayEndpoint object with associated model mappings.
        """
        req_body = message_to_json(
            CreateGatewayEndpoint(
                name=name,
                model_definition_ids=model_definition_ids,
                created_by=created_by,
            )
        )
        response_proto = self._call_endpoint(CreateGatewayEndpoint, req_body)
        return GatewayEndpoint.from_proto(response_proto.endpoint)

    def get_gateway_endpoint(
        self, endpoint_id: str | None = None, name: str | None = None
    ) -> GatewayEndpoint:
        """
        Retrieve an endpoint with its model configurations.

        Args:
            endpoint_id: The unique identifier of the endpoint.
            name: The name of the endpoint.

        Returns:
            The GatewayEndpoint object with associated models.
        """
        req_body = message_to_json(GetGatewayEndpoint(endpoint_id=endpoint_id, name=name))
        response_proto = self._call_endpoint(GetGatewayEndpoint, req_body)
        return GatewayEndpoint.from_proto(response_proto.endpoint)

    def update_gateway_endpoint(
        self,
        endpoint_id: str,
        name: str | None = None,
        updated_by: str | None = None,
    ) -> GatewayEndpoint:
        """
        Update an endpoint's metadata.

        Args:
            endpoint_id: The unique identifier of the endpoint to update.
            name: Optional new name for the endpoint.
            updated_by: Optional identifier of the user updating the endpoint.

        Returns:
            The updated GatewayEndpoint object.
        """
        req_body = message_to_json(
            UpdateGatewayEndpoint(
                endpoint_id=endpoint_id,
                name=name,
                updated_by=updated_by,
            )
        )
        response_proto = self._call_endpoint(UpdateGatewayEndpoint, req_body)
        return GatewayEndpoint.from_proto(response_proto.endpoint)

    def delete_gateway_endpoint(self, endpoint_id: str) -> None:
        """
        Delete an endpoint and all its associated models and bindings.

        Args:
            endpoint_id: The unique identifier of the endpoint to delete.
        """
        req_body = message_to_json(DeleteGatewayEndpoint(endpoint_id=endpoint_id))
        self._call_endpoint(DeleteGatewayEndpoint, req_body)

    def list_gateway_endpoints(self, provider: str | None = None) -> list[GatewayEndpoint]:
        """
        List all endpoints, optionally filtered by provider.

        Args:
            provider: Optional provider name to filter endpoints.

        Returns:
            List of GatewayEndpoint objects with their associated models.
        """
        req_body = message_to_json(ListGatewayEndpoints(provider=provider))
        response_proto = self._call_endpoint(ListGatewayEndpoints, req_body)
        return [GatewayEndpoint.from_proto(e) for e in response_proto.endpoints]

    # ========== Model Definitions Management APIs ==========

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
            name: User-friendly name for the model definition.
            secret_id: ID of the secret containing API credentials.
            provider: Provider name (e.g., "openai", "anthropic").
            model_name: Name of the model (e.g., "gpt-4", "claude-3-5-sonnet").
            created_by: Optional identifier of the user creating the definition.

        Returns:
            The created GatewayModelDefinition object.
        """
        req_body = message_to_json(
            CreateGatewayModelDefinition(
                name=name,
                secret_id=secret_id,
                provider=provider,
                model_name=model_name,
                created_by=created_by,
            )
        )
        response_proto = self._call_endpoint(CreateGatewayModelDefinition, req_body)
        return GatewayModelDefinition.from_proto(response_proto.model_definition)

    def get_gateway_model_definition(self, model_definition_id: str) -> GatewayModelDefinition:
        """
        Retrieve a model definition by ID.

        Args:
            model_definition_id: The unique identifier of the model definition.

        Returns:
            The GatewayModelDefinition object.
        """
        req_body = message_to_json(
            GetGatewayModelDefinition(model_definition_id=model_definition_id)
        )
        response_proto = self._call_endpoint(GetGatewayModelDefinition, req_body)
        return GatewayModelDefinition.from_proto(response_proto.model_definition)

    def list_gateway_model_definitions(
        self,
        provider: str | None = None,
        secret_id: str | None = None,
    ) -> list[GatewayModelDefinition]:
        """
        List all model definitions, optionally filtered.

        Args:
            provider: Optional provider name to filter definitions.
            secret_id: Optional secret ID to filter definitions.

        Returns:
            List of GatewayModelDefinition objects.
        """
        req_body = message_to_json(
            ListGatewayModelDefinitions(provider=provider, secret_id=secret_id)
        )
        response_proto = self._call_endpoint(ListGatewayModelDefinitions, req_body)
        return [GatewayModelDefinition.from_proto(m) for m in response_proto.model_definitions]

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
            model_definition_id: The unique identifier of the model definition.
            name: Optional new name.
            secret_id: Optional new secret ID.
            model_name: Optional new model name.
            updated_by: Optional identifier of the user updating the definition.
            provider: Optional new provider.

        Returns:
            The updated GatewayModelDefinition object.
        """
        req_body = message_to_json(
            UpdateGatewayModelDefinition(
                model_definition_id=model_definition_id,
                name=name,
                secret_id=secret_id,
                model_name=model_name,
                updated_by=updated_by,
                provider=provider,
            )
        )
        response_proto = self._call_endpoint(UpdateGatewayModelDefinition, req_body)
        return GatewayModelDefinition.from_proto(response_proto.model_definition)

    def delete_gateway_model_definition(self, model_definition_id: str) -> None:
        """
        Delete a model definition (fails if in use by any endpoint).

        Args:
            model_definition_id: The unique identifier of the model definition.
        """
        req_body = message_to_json(
            DeleteGatewayModelDefinition(model_definition_id=model_definition_id)
        )
        self._call_endpoint(DeleteGatewayModelDefinition, req_body)

    # ========== Endpoint Model Mappings Management APIs ==========

    def attach_model_to_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
        weight: float = 1.0,
        created_by: str | None = None,
    ) -> GatewayEndpointModelMapping:
        """
        Attach a model definition to an endpoint.

        Args:
            endpoint_id: The unique identifier of the endpoint.
            model_definition_id: The unique identifier of the model definition.
            weight: Optional routing weight (default 1).
            created_by: Optional identifier of the user creating the mapping.

        Returns:
            The created GatewayEndpointModelMapping object.
        """
        req_body = message_to_json(
            AttachModelToGatewayEndpoint(
                endpoint_id=endpoint_id,
                model_definition_id=model_definition_id,
                weight=weight,
                created_by=created_by,
            )
        )
        response_proto = self._call_endpoint(AttachModelToGatewayEndpoint, req_body)
        return GatewayEndpointModelMapping.from_proto(response_proto.mapping)

    def detach_model_from_endpoint(
        self,
        endpoint_id: str,
        model_definition_id: str,
    ) -> None:
        """
        Detach a model definition from an endpoint.

        Args:
            endpoint_id: The unique identifier of the endpoint.
            model_definition_id: The unique identifier of the model definition.
        """
        req_body = message_to_json(
            DetachModelFromGatewayEndpoint(
                endpoint_id=endpoint_id,
                model_definition_id=model_definition_id,
            )
        )
        self._call_endpoint(DetachModelFromGatewayEndpoint, req_body)

    # ========== Endpoint Bindings Management APIs ==========

    def create_endpoint_binding(
        self,
        endpoint_id: str,
        resource_type: GatewayResourceType,
        resource_id: str,
        created_by: str | None = None,
    ) -> GatewayEndpointBinding:
        """
        Create a binding between an endpoint and a resource.

        Args:
            endpoint_id: The unique identifier of the endpoint.
            resource_type: Type of resource to bind.
            resource_id: The unique identifier of the resource.
            created_by: Optional identifier of the user creating the binding.

        Returns:
            The created GatewayEndpointBinding object.
        """
        req_body = message_to_json(
            CreateGatewayEndpointBinding(
                endpoint_id=endpoint_id,
                resource_type=resource_type.value,
                resource_id=resource_id,
                created_by=created_by,
            )
        )
        response_proto = self._call_endpoint(CreateGatewayEndpointBinding, req_body)
        return GatewayEndpointBinding.from_proto(response_proto.binding)

    def delete_endpoint_binding(
        self, endpoint_id: str, resource_type: str, resource_id: str
    ) -> None:
        """
        Delete a binding between an endpoint and a resource.

        Args:
            endpoint_id: ID of the endpoint.
            resource_type: Type of resource bound to the endpoint.
            resource_id: ID of the resource.
        """
        req_body = message_to_json(
            DeleteGatewayEndpointBinding(
                endpoint_id=endpoint_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        )
        self._call_endpoint(DeleteGatewayEndpointBinding, req_body)

    def list_endpoint_bindings(
        self,
        endpoint_id: str | None = None,
        resource_type: GatewayResourceType | None = None,
        resource_id: str | None = None,
    ) -> list[GatewayEndpointBinding]:
        """
        List endpoint bindings with optional server-side filtering.

        Args:
            endpoint_id: Optional endpoint ID to filter bindings.
            resource_type: Optional resource type to filter bindings.
            resource_id: Optional resource ID to filter bindings.

        Returns:
            List of GatewayEndpointBinding objects matching the filters.
        """
        req_body = message_to_json(
            ListGatewayEndpointBindings(
                endpoint_id=endpoint_id,
                resource_type=resource_type.value if resource_type else None,
                resource_id=resource_id,
            )
        )
        response_proto = self._call_endpoint(ListGatewayEndpointBindings, req_body)
        return [GatewayEndpointBinding.from_proto(b) for b in response_proto.bindings]

    def set_gateway_endpoint_tag(self, endpoint_id: str, tag: GatewayEndpointTag) -> None:
        """
        Set a tag on an endpoint.

        Args:
            endpoint_id: ID of the endpoint to tag.
            tag: GatewayEndpointTag with key and value to set.
        """
        req_body = message_to_json(
            SetGatewayEndpointTag(
                endpoint_id=endpoint_id,
                key=tag.key,
                value=tag.value,
            )
        )
        self._call_endpoint(SetGatewayEndpointTag, req_body)

    def delete_gateway_endpoint_tag(self, endpoint_id: str, key: str) -> None:
        """
        Delete a tag from an endpoint.

        Args:
            endpoint_id: ID of the endpoint.
            key: Tag key to delete.
        """
        req_body = message_to_json(
            DeleteGatewayEndpointTag(
                endpoint_id=endpoint_id,
                key=key,
            )
        )
        self._call_endpoint(DeleteGatewayEndpointTag, req_body)

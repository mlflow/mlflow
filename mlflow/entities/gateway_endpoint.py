from dataclasses import dataclass, field
from enum import Enum

from mlflow.entities._mlflow_object import _MlflowObject


class GatewayResourceType(str, Enum):
    """Valid MLflow resource types that can use gateway endpoints."""

    SCORER_JOB = "scorer_job"


@dataclass
class GatewayModelDefinition(_MlflowObject):
    """
    Represents a reusable LLM model configuration.

    Model definitions can be shared across multiple endpoints, enabling
    centralized management of model configurations and API credentials.

    Args:
        model_definition_id: Unique identifier for this model definition.
        name: User-friendly name for identification and reuse.
        secret_id: ID of the secret containing authentication credentials.
        secret_name: Name of the secret for display/reference purposes.
        provider: LLM provider (e.g., "openai", "anthropic", "cohere", "bedrock").
        model_name: Provider-specific model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
        created_at: Timestamp (milliseconds) when the model definition was created.
        last_updated_at: Timestamp (milliseconds) when the model definition was last updated.
        created_by: User ID who created the model definition.
        last_updated_by: User ID who last updated the model definition.
    """

    model_definition_id: str
    name: str
    secret_id: str
    secret_name: str
    provider: str
    model_name: str
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None


@dataclass
class GatewayEndpointModelMapping(_MlflowObject):
    """
    Represents a mapping between an endpoint and a model definition.

    This is a junction entity that links endpoints to model definitions,
    enabling many-to-many relationships and traffic routing configuration.

    Args:
        mapping_id: Unique identifier for this mapping.
        endpoint_id: ID of the endpoint.
        model_definition_id: ID of the model definition.
        model_definition: The full model definition (populated via JOIN).
        weight: Routing weight for traffic distribution (default 1).
        created_at: Timestamp (milliseconds) when the mapping was created.
        created_by: User ID who created the mapping.
    """

    mapping_id: str
    endpoint_id: str
    model_definition_id: str
    model_definition: GatewayModelDefinition | None
    weight: int
    created_at: int
    created_by: str | None = None


@dataclass
class GatewayEndpoint(_MlflowObject):
    """
    Represents an LLM gateway endpoint with its associated model configurations.

    Args:
        endpoint_id: Unique identifier for this endpoint.
        name: User-friendly name for the endpoint.
        created_at: Timestamp (milliseconds) when the endpoint was created.
        last_updated_at: Timestamp (milliseconds) when the endpoint was last updated.
        model_mappings: List of model mappings bound to this endpoint.
        created_by: User ID who created the endpoint.
        last_updated_by: User ID who last updated the endpoint.
    """

    endpoint_id: str
    name: str
    created_at: int
    last_updated_at: int
    model_mappings: list[GatewayEndpointModelMapping] = field(default_factory=list)
    created_by: str | None = None
    last_updated_by: str | None = None


@dataclass
class GatewayEndpointBinding(_MlflowObject):
    """
    Represents a binding between an endpoint and an MLflow resource.

    Bindings track which MLflow resources (e.g., scorer jobs) are configured to use
    which endpoints. The composite key (endpoint_id, resource_type, resource_id) uniquely
    identifies each binding.

    Args:
        endpoint_id: ID of the endpoint this binding references.
        resource_type: Type of MLflow resource (e.g., "scorer_job").
        resource_id: ID of the specific resource instance.
        created_at: Timestamp (milliseconds) when the binding was created.
        last_updated_at: Timestamp (milliseconds) when the binding was last updated.
        created_by: User ID who created the binding.
        last_updated_by: User ID who last updated the binding.
    """

    endpoint_id: str
    resource_type: GatewayResourceType
    resource_id: str
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None

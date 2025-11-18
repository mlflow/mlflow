from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject

if TYPE_CHECKING:
    from mlflow.entities.endpoint_model import EndpointModel
    from mlflow.entities.endpoint_tag import EndpointTag


@dataclass
class Endpoint(_MlflowObject):
    """
    MLflow entity representing an Endpoint.

    Endpoints are routing configurations that can direct traffic to multiple models.
    Each model in the endpoint has its own secret (API key) for authentication.

    This entity contains only metadata about the endpoint - cryptographic fields
    (encrypted_model_config, wrapped_model_config_dek) are never exposed outside the store layer.

    Args:
        endpoint_id: String containing endpoint ID (UUID).
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        name: String containing optional display name, or None.
        description: String containing optional user-provided description, or None.
        created_by: String containing the user ID who created the endpoint, or None.
        last_updated_by: String containing the user ID who last updated the endpoint, or None.
        models: List of EndpointModel objects associated with this endpoint.
        tags: List of EndpointTag objects associated with this endpoint.
    """

    endpoint_id: str
    created_at: int
    last_updated_at: int
    name: str | None = None
    description: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    models: list[EndpointModel] = field(default_factory=list)
    tags: list[EndpointTag] = field(default_factory=list)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlflow.entities._mlflow_object import _MlflowObject

if TYPE_CHECKING:
    from mlflow.entities.secret_route_tag import SecretRouteTag


@dataclass
class SecretRoute(_MlflowObject):
    """
    MLflow entity representing a SecretRoute.

    SecretRoutes map API keys (secrets) to specific model configurations for LLM providers.
    One secret (API key) can be used with multiple models through different routes.

    This entity contains only metadata about the route - cryptographic fields
    (encrypted_model_config, wrapped_model_config_dek) are never exposed outside the store layer.

    Args:
        route_id: String containing route ID (UUID).
        secret_id: String containing the secret ID this route uses.
        model_name: String containing the model identifier.
            E.g., "claude-3-5-sonnet-20241022", "gpt-4-turbo", "gemini-2.5-pro".
        name: String containing optional display name, or None.
            If not provided, model_name is used for display.
        description: String containing optional user-provided description, or None.
        created_at: Creation timestamp in milliseconds since the UNIX epoch.
        last_updated_at: Last update timestamp in milliseconds since the UNIX epoch.
        created_by: String containing the user ID who created the route, or None.
        last_updated_by: String containing the user ID who last updated the route, or None.
        tags: List of SecretRouteTag objects associated with this route.
    """

    route_id: str
    secret_id: str
    model_name: str
    created_at: int
    last_updated_at: int
    name: str | None = None
    description: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    tags: list[SecretRouteTag] = field(default_factory=list)


@dataclass
class SecretRouteListItem(SecretRoute):
    """
    SecretRoute with additional display information for list responses.

    Extends SecretRoute with human-readable fields populated via JOIN
    with the secrets table. Used by list_secret_routes() to provide
    UI-friendly data without additional API calls.

    Args:
        secret_name: User-friendly secret name (e.g., "company_openai_key").
        provider: LLM provider (e.g., "openai", "anthropic", "google").
    """

    secret_name: str = ""
    provider: str = ""

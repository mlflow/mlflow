import importlib.util
from typing import Any, TypedDict

from typing_extensions import NotRequired

# Check if the provider backend is available without importing it (import is slow ~2.7s)
_PROVIDER_BACKEND_AVAILABLE = importlib.util.find_spec("litellm") is not None


class SecretFieldDict(TypedDict):
    """Schema for a secret field in auth configuration."""

    name: str
    type: str
    description: str
    required: bool


class ConfigFieldDict(TypedDict):
    """Schema for a non-secret config field in auth configuration."""

    name: str
    type: str
    description: str
    required: bool
    default: NotRequired[str | None]


class AuthModeDict(TypedDict):
    """Schema for an authentication mode."""

    display_name: str
    description: str
    credential_name: str
    secret_fields: list[SecretFieldDict]
    config_fields: list[ConfigFieldDict]
    completion_params: list[str]
    default: NotRequired[bool]
    runtime_auth: NotRequired[str]


class ProviderConfigResponse(TypedDict):
    """Response type for get_provider_config_response."""

    auth_modes: list[dict[str, Any]]
    default_mode: str


def _get_model_cost():
    """Lazy import of model_cost from the provider backend."""
    from litellm import model_cost

    return model_cost


def _get_provider_fields():
    """Lazy import of get_provider_fields from the provider backend."""
    from litellm import get_provider_fields

    return get_provider_fields


# Auth modes for providers with multiple authentication options.
# Each mode defines:
#   - display_name: Human-readable name for UI
#   - description: Help text explaining this auth method
#   - credential_name: Display name of the primary secret (for UI)
#   - secret_fields: Fields stored encrypted in encrypted_value JSON
#   - config_fields: Non-secret fields stored in auth_config JSON
#   - completion_params: Parameter names passed to completion() calls
#   - default: True if this is the default auth mode for the provider
#   - runtime_auth: Optional runtime auth handler name
#
# Configuration sourced from LiteLLM documentation and provider APIs:
#   - AWS Bedrock: https://docs.litellm.ai/docs/providers/bedrock
#   - Azure OpenAI: https://docs.litellm.ai/docs/providers/azure
#   - Vertex AI: https://docs.litellm.ai/docs/providers/vertex
#   - Databricks: https://docs.litellm.ai/docs/providers/databricks
#
# Only user-provided modes are included (no server-provided modes like
# managed identity, IRSA, or ADC that require specific hosting environments).
_PROVIDER_AUTH_MODES: dict[str, dict[str, AuthModeDict]] = {
    "bedrock": {
        "access_keys": {
            "display_name": "Access Keys",
            "description": "Use AWS Access Key ID and Secret Access Key",
            "credential_name": "AWS_ACCESS_KEY_ID",
            "secret_fields": [
                {
                    "name": "aws_access_key_id",
                    "type": "string",
                    "description": "AWS Access Key ID",
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "type": "string",
                    "description": "AWS Secret Access Key",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "aws_region_name",
                    "type": "string",
                    "description": "AWS Region (e.g., us-east-1)",
                    "required": False,
                },
            ],
            "completion_params": [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_region_name",
            ],
            "default": True,
        },
        "iam_role": {
            "display_name": "IAM Role Assumption",
            "description": "Assume an IAM role using base credentials (for cross-account access)",
            "credential_name": "AWS_ACCESS_KEY_ID",
            "secret_fields": [
                {
                    "name": "aws_access_key_id",
                    "type": "string",
                    "description": "AWS Access Key ID (for assuming role)",
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "type": "string",
                    "description": "AWS Secret Access Key",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "aws_role_name",
                    "type": "string",
                    "description": "IAM Role ARN to assume",
                    "required": True,
                },
                {
                    "name": "aws_session_name",
                    "type": "string",
                    "description": "Session name for assumed role",
                    "required": False,
                },
                {
                    "name": "aws_region_name",
                    "type": "string",
                    "description": "AWS Region (e.g., us-east-1)",
                    "required": False,
                },
            ],
            "completion_params": [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_role_name",
                "aws_session_name",
                "aws_region_name",
            ],
        },
        "session_token": {
            "display_name": "Session Token (STS)",
            "description": "Use temporary credentials with session token",
            "credential_name": "AWS_ACCESS_KEY_ID",
            "secret_fields": [
                {
                    "name": "aws_access_key_id",
                    "type": "string",
                    "description": "AWS Access Key ID",
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "type": "string",
                    "description": "AWS Secret Access Key",
                    "required": True,
                },
                {
                    "name": "aws_session_token",
                    "type": "string",
                    "description": "AWS Session Token",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "aws_region_name",
                    "type": "string",
                    "description": "AWS Region (e.g., us-east-1)",
                    "required": False,
                },
            ],
            "completion_params": [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
            ],
        },
    },
    "azure": {
        "api_key": {
            "display_name": "API Key",
            "description": "Use Azure OpenAI API Key",
            "credential_name": "AZURE_API_KEY",
            "secret_fields": [
                {
                    "name": "api_key",
                    "type": "string",
                    "description": "Azure OpenAI API Key",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "api_base",
                    "type": "string",
                    "description": "Azure OpenAI endpoint URL",
                    "required": True,
                },
                {
                    "name": "api_version",
                    "type": "string",
                    "description": "API version (e.g., 2024-02-01)",
                    "required": False,
                    "default": "2024-02-01",
                },
            ],
            "completion_params": ["api_key", "api_base", "api_version"],
            "default": True,
        },
        "service_principal": {
            "display_name": "Service Principal",
            "description": "Use Azure AD Service Principal (client credentials)",
            "credential_name": "AZURE_CLIENT_SECRET",
            "secret_fields": [
                {
                    "name": "client_secret",
                    "type": "string",
                    "description": "Azure AD Client Secret",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "api_base",
                    "type": "string",
                    "description": "Azure OpenAI endpoint URL",
                    "required": True,
                },
                {
                    "name": "client_id",
                    "type": "string",
                    "description": "Azure AD Application (Client) ID",
                    "required": True,
                },
                {
                    "name": "tenant_id",
                    "type": "string",
                    "description": "Azure AD Tenant ID",
                    "required": True,
                },
                {
                    "name": "api_version",
                    "type": "string",
                    "description": "API version (e.g., 2024-02-01)",
                    "required": False,
                    "default": "2024-02-01",
                },
            ],
            "completion_params": [
                "api_base",
                "api_version",
            ],
            "runtime_auth": "azure_service_principal",
        },
    },
    "vertex_ai": {
        "service_account_json": {
            "display_name": "Service Account JSON",
            "description": "Use GCP Service Account credentials (JSON key file contents)",
            "credential_name": "GOOGLE_APPLICATION_CREDENTIALS",
            "secret_fields": [
                {
                    "name": "vertex_credentials",
                    "type": "json",
                    "description": "Service Account JSON key file contents",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "vertex_project",
                    "type": "string",
                    "description": "GCP Project ID",
                    "required": True,
                },
                {
                    "name": "vertex_location",
                    "type": "string",
                    "description": "GCP Region (e.g., us-central1)",
                    "required": False,
                    "default": "us-central1",
                },
            ],
            "completion_params": ["vertex_credentials", "vertex_project", "vertex_location"],
            "default": True,
        },
    },
    "databricks": {
        "pat_token": {
            "display_name": "Personal Access Token",
            "description": "Use Databricks Personal Access Token",
            "credential_name": "DATABRICKS_TOKEN",
            "secret_fields": [
                {
                    "name": "api_key",
                    "type": "string",
                    "description": "Databricks Personal Access Token",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "api_base",
                    "type": "string",
                    "description": "Databricks workspace URL",
                    "required": True,
                },
            ],
            "completion_params": ["api_key", "api_base"],
            "default": True,
        },
        "oauth_m2m": {
            "display_name": "OAuth M2M (Service Principal)",
            "description": "Use OAuth machine-to-machine authentication",
            "credential_name": "DATABRICKS_CLIENT_SECRET",
            "secret_fields": [
                {
                    "name": "client_secret",
                    "type": "string",
                    "description": "OAuth Client Secret",
                    "required": True,
                },
            ],
            "config_fields": [
                {
                    "name": "api_base",
                    "type": "string",
                    "description": "Databricks workspace URL",
                    "required": True,
                },
                {
                    "name": "client_id",
                    "type": "string",
                    "description": "OAuth Client ID",
                    "required": True,
                },
            ],
            "completion_params": ["api_base"],
            "runtime_auth": "databricks_oauth_m2m",
        },
    },
}


def _get_credential_fields(provider: str) -> list[dict[str, Any]]:
    """
    Get credential fields for a provider from LiteLLM.

    Args:
        provider: The LiteLLM provider name

    Returns:
        List of credential field dictionaries
    """
    if not _PROVIDER_BACKEND_AVAILABLE:
        return []

    get_provider_fields = _get_provider_fields()
    provider_fields = get_provider_fields(provider)

    if provider_fields and len(provider_fields) > 0:
        return [
            {
                "name": field["field_name"],
                "type": field.get("field_type", "string"),
                "description": field.get("field_description", ""),
                "required": True,
            }
            for field in provider_fields
        ]

    return []


def get_provider_config_response(provider: str) -> ProviderConfigResponse:
    """
    Get provider configuration formatted for API response.

    For providers with multiple auth modes (bedrock, azure, vertex_ai, databricks),
    returns the full auth_modes structure. For simple API key providers, returns
    a single default auth mode.

    Args:
        provider: The LiteLLM provider name (e.g., 'openai', 'anthropic', 'databricks')

    Returns:
        dict with keys:
            - auth_modes: List of available authentication modes, each containing:
                - mode: Auth mode identifier (e.g., 'access_keys', 'api_key')
                - display_name: Human-readable name
                - description: Help text
                - credential_name: Display name of primary secret
                - secret_fields: Fields to store encrypted
                - config_fields: Non-secret config fields
            - default_mode: The recommended default auth mode

    Raises:
        ImportError: If litellm is not installed
        ValueError: If provider is not valid or not provided
    """
    if not provider:
        raise ValueError("Provider parameter is required")

    # Check if provider has defined auth modes
    if provider in _PROVIDER_AUTH_MODES:
        auth_modes = []
        default_mode = None
        for mode_id, mode_config in _PROVIDER_AUTH_MODES[provider].items():
            auth_modes.append(
                {
                    "mode": mode_id,
                    "display_name": mode_config["display_name"],
                    "description": mode_config["description"],
                    "credential_name": mode_config["credential_name"],
                    "secret_fields": mode_config["secret_fields"],
                    "config_fields": mode_config["config_fields"],
                }
            )
            if mode_config.get("default"):
                default_mode = mode_id
        return {
            "auth_modes": auth_modes,
            "default_mode": default_mode or auth_modes[0]["mode"],
        }

    # For simple providers, create a single auth mode from credential fields
    if credential_fields := _get_credential_fields(provider):
        primary_field = credential_fields[0]
        auth_mode = {
            "mode": "api_key",
            "display_name": "API Key",
            "description": f"Use {provider.title()} API Key",
            "credential_name": primary_field["name"],
            "secret_fields": [
                {
                    "name": "api_key",
                    "type": "string",
                    "description": primary_field.get("description", "API Key"),
                    "required": True,
                }
            ],
            "config_fields": [
                {
                    "name": field["name"],
                    "type": field.get("type", "string"),
                    "description": field.get("description", ""),
                    "required": field.get("required", True),
                    "default": field.get("default"),
                }
                for field in credential_fields[1:]
            ],
        }
        return {
            "auth_modes": [auth_mode],
            "default_mode": "api_key",
        }

    # Fallback for unknown providers
    return {
        "auth_modes": [
            {
                "mode": "api_key",
                "display_name": "API Key",
                "description": f"Use {provider.title()} API Key",
                "credential_name": f"{provider.upper()}_API_KEY",
                "secret_fields": [
                    {
                        "name": "api_key",
                        "type": "string",
                        "description": f"{provider.title()} API Key",
                        "required": True,
                    }
                ],
                "config_fields": [],
            }
        ],
        "default_mode": "api_key",
    }


def get_all_providers() -> list[str]:
    """
    Get a list of all LiteLLM providers that have chat, completion, or embedding capabilities.

    Only returns providers that have at least one chat, completion, or embedding model,
    excluding providers that only offer image generation, audio, or other non-text services.

    Returns:
        List of provider names that support chat/completion/embedding

    Raises:
        ImportError: If litellm is not installed
    """
    if not _PROVIDER_BACKEND_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    model_cost = _get_model_cost()
    providers = set()
    for _, info in model_cost.items():
        mode = info.get("mode")
        # Include providers with chat/completion/embedding models
        # mode=None indicates legacy completion models for backwards compatibility
        if mode in ("chat", "completion", "embedding", None):
            if provider := info.get("litellm_provider"):
                providers.add(provider)

    return list(providers)


def get_models(provider: str | None = None) -> list[dict[str, Any]]:
    """
    Get a list of models from LiteLLM, optionally filtered by provider.

    Returns models that support chat, completion, or embedding capabilities,
    excluding image generation, audio, and other non-text services.

    Args:
        provider: Optional provider name to filter by (e.g., 'openai', 'anthropic')

    Returns:
        List of model dictionaries with keys:
            - model: Model name
            - provider: Provider name
            - mode: Model mode (e.g., 'chat', 'completion', 'embedding')
            - supports_function_calling: Whether model supports tool/function calling
            - supports_vision: Whether model supports image/vision input
            - supports_reasoning: Whether model supports extended thinking (o1-style)
            - supports_prompt_caching: Whether model supports prompt caching
            - max_input_tokens: Maximum input context window size
            - max_output_tokens: Maximum output token limit
            - input_cost_per_token: Cost per input token (USD)
            - output_cost_per_token: Cost per output token (USD)

    Raises:
        ImportError: If litellm is not installed
    """
    if not _PROVIDER_BACKEND_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    model_cost = _get_model_cost()
    models = []
    for model_name, info in model_cost.items():
        if provider and info.get("litellm_provider") != provider:
            continue

        # Include chat/completion/embedding models
        # mode=None indicates legacy completion models for backwards compatibility
        # Exclude image_generation, audio, etc.
        mode = info.get("mode")
        if mode not in ("chat", "completion", "embedding", None):
            continue

        models.append(
            {
                "model": model_name,
                "provider": info.get("litellm_provider"),
                "mode": mode,
                "supports_function_calling": info.get("supports_function_calling", False),
                "supports_vision": info.get("supports_vision", False),
                "supports_reasoning": info.get("supports_reasoning", False),
                "supports_prompt_caching": info.get("supports_prompt_caching", False),
                "max_input_tokens": info.get("max_input_tokens"),
                "max_output_tokens": info.get("max_output_tokens"),
                "input_cost_per_token": info.get("input_cost_per_token"),
                "output_cost_per_token": info.get("output_cost_per_token"),
            }
        )

    return models

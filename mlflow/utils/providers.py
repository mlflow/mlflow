import importlib.util
from typing import Any, TypedDict

from typing_extensions import NotRequired

_PROVIDER_BACKEND_AVAILABLE = importlib.util.find_spec("litellm") is not None

_SUPPORTED_MODEL_MODES = ("chat", "completion", "embedding", None)


class FieldDict(TypedDict):
    name: str
    description: str
    secret: bool
    required: bool
    default: NotRequired[str | None]


class AuthModeDict(TypedDict):
    display_name: str
    description: str
    fields: list[FieldDict]
    default: NotRequired[bool]
    runtime_auth: NotRequired[str]


class ResponseFieldDict(TypedDict):
    name: str
    type: str
    description: str
    required: bool
    default: NotRequired[str | None]


class AuthModeResponseDict(TypedDict):
    mode: str
    display_name: str
    description: str
    secret_fields: list[ResponseFieldDict]
    config_fields: list[ResponseFieldDict]


class ProviderConfigResponse(TypedDict):
    auth_modes: list[AuthModeResponseDict]
    default_mode: str


def _get_model_cost():
    from litellm import model_cost

    return model_cost


# Auth modes for providers with multiple authentication options.
# Each mode defines:
#   - display_name: Human-readable name for UI
#   - description: Help text explaining this auth method
#   - fields: List of fields with secret flag indicating if encrypted
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
            "default": True,
            "fields": [
                {
                    "name": "aws_access_key_id",
                    "description": "AWS Access Key ID",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "description": "AWS Secret Access Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region (e.g., us-east-1)",
                    "secret": False,
                    "required": False,
                },
            ],
        },
        "iam_role": {
            "display_name": "IAM Role Assumption",
            "description": "Assume an IAM role using base credentials (for cross-account access)",
            "fields": [
                {
                    "name": "aws_access_key_id",
                    "description": "AWS Access Key ID (for assuming role)",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "description": "AWS Secret Access Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_role_name",
                    "description": "IAM Role ARN to assume",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "aws_session_name",
                    "description": "Session name for assumed role",
                    "secret": False,
                    "required": False,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region (e.g., us-east-1)",
                    "secret": False,
                    "required": False,
                },
            ],
        },
        "session_token": {
            "display_name": "Session Token (STS)",
            "description": "Use temporary credentials with session token",
            "fields": [
                {
                    "name": "aws_access_key_id",
                    "description": "AWS Access Key ID",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_secret_access_key",
                    "description": "AWS Secret Access Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_session_token",
                    "description": "AWS Session Token",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region (e.g., us-east-1)",
                    "secret": False,
                    "required": False,
                },
            ],
        },
    },
    "azure": {
        "api_key": {
            "display_name": "API Key",
            "description": "Use Azure OpenAI API Key",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Azure OpenAI API Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Azure OpenAI endpoint URL",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "api_version",
                    "description": "API version (e.g., 2024-02-01)",
                    "secret": False,
                    "required": False,
                    "default": "2024-02-01",
                },
            ],
        },
        "service_principal": {
            "display_name": "Service Principal",
            "description": "Use Azure AD Service Principal (client credentials)",
            "runtime_auth": "azure_service_principal",
            "fields": [
                {
                    "name": "client_secret",
                    "description": "Azure AD Client Secret",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Azure OpenAI endpoint URL",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "client_id",
                    "description": "Azure AD Application (Client) ID",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "tenant_id",
                    "description": "Azure AD Tenant ID",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "api_version",
                    "description": "API version (e.g., 2024-02-01)",
                    "secret": False,
                    "required": False,
                    "default": "2024-02-01",
                },
            ],
        },
    },
    "vertex_ai": {
        "service_account_json": {
            "display_name": "Service Account JSON",
            "description": "Use GCP Service Account credentials (JSON key file contents)",
            "default": True,
            "fields": [
                {
                    "name": "vertex_credentials",
                    "description": "Service Account JSON key file contents",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "vertex_project",
                    "description": "GCP Project ID",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "vertex_location",
                    "description": "GCP Region (e.g., us-central1)",
                    "secret": False,
                    "required": False,
                    "default": "us-central1",
                },
            ],
        },
    },
    "databricks": {
        "pat_token": {
            "display_name": "Personal Access Token",
            "description": "Use Databricks Personal Access Token",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Databricks Personal Access Token",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Databricks workspace URL",
                    "secret": False,
                    "required": True,
                },
            ],
        },
        "oauth_m2m": {
            "display_name": "OAuth M2M (Service Principal)",
            "description": "Use OAuth machine-to-machine authentication",
            "runtime_auth": "databricks_oauth_m2m",
            "fields": [
                {
                    "name": "client_secret",
                    "description": "OAuth Client Secret",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "api_base",
                    "description": "Databricks workspace URL",
                    "secret": False,
                    "required": True,
                },
                {
                    "name": "client_id",
                    "description": "OAuth Client ID",
                    "secret": False,
                    "required": True,
                },
            ],
        },
    },
}


def _build_response_field(field: FieldDict) -> ResponseFieldDict:
    response: ResponseFieldDict = {
        "name": field["name"],
        "type": "string",
        "description": field.get("description", ""),
        "required": field.get("required", True),
    }
    if "default" in field:
        response["default"] = field["default"]
    return response


def _build_auth_mode_response(mode_id: str, mode_config: AuthModeDict) -> AuthModeResponseDict:
    secret_fields: list[ResponseFieldDict] = []
    config_fields: list[ResponseFieldDict] = []

    for field in mode_config["fields"]:
        response_field = _build_response_field(field)
        if field.get("secret"):
            secret_fields.append(response_field)
        else:
            config_fields.append(response_field)

    return {
        "mode": mode_id,
        "display_name": mode_config["display_name"],
        "description": mode_config["description"],
        "secret_fields": secret_fields,
        "config_fields": config_fields,
    }


def _build_simple_api_key_mode(provider: str, description: str | None = None) -> AuthModeDict:
    return {
        "display_name": "API Key",
        "description": description or f"Use {provider.title()} API Key",
        "default": True,
        "fields": [
            {
                "name": "api_key",
                "description": f"{provider.title()} API Key",
                "secret": True,
                "required": True,
            },
        ],
    }


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
                - secret_fields: Fields to store encrypted
                - config_fields: Non-secret config fields
            - default_mode: The recommended default auth mode
    """
    if not provider:
        raise ValueError("Provider parameter is required")

    if provider in _PROVIDER_AUTH_MODES:
        auth_modes: list[AuthModeResponseDict] = []
        default_mode: str | None = None
        for mode_id, mode_config in _PROVIDER_AUTH_MODES[provider].items():
            auth_modes.append(_build_auth_mode_response(mode_id, mode_config))
            if mode_config.get("default"):
                default_mode = mode_id
        return {
            "auth_modes": auth_modes,
            "default_mode": default_mode or auth_modes[0]["mode"],
        }

    simple_mode = _build_simple_api_key_mode(provider)
    return {
        "auth_modes": [_build_auth_mode_response("api_key", simple_mode)],
        "default_mode": "api_key",
    }


def get_all_providers() -> list[str]:
    """
    Get a list of all LiteLLM providers that have chat, completion, or embedding capabilities.

    Only returns providers that have at least one chat, completion, or embedding model,
    excluding providers that only offer image generation, audio, or other non-text services.

    Returns:
        List of provider names that support chat/completion/embedding
    """
    if not _PROVIDER_BACKEND_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    model_cost = _get_model_cost()
    providers = set()
    for _, info in model_cost.items():
        mode = info.get("mode")
        if mode in _SUPPORTED_MODEL_MODES:
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
    """
    if not _PROVIDER_BACKEND_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    model_cost = _get_model_cost()
    models = []
    for model_name, info in model_cost.items():
        if provider and info.get("litellm_provider") != provider:
            continue

        mode = info.get("mode")
        if mode not in _SUPPORTED_MODEL_MODES:
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

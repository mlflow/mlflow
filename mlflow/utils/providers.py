import functools
import importlib.resources
import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import cachetools
from typing_extensions import NotRequired

from mlflow.environment_variables import MLFLOW_MODEL_CATALOG_CACHE_TTL, MLFLOW_MODEL_CATALOG_URI
from mlflow.exceptions import MlflowException
from mlflow.utils.provider_filter import (
    filter_providers,
    is_provider_allowed,
    normalize_provider_name,
)
from mlflow.utils.request_utils import cloud_storage_http_request

_logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_MODES = ("chat", "completion", "embedding", None)

_REMOTE_FETCH_MAX_RETRIES = 3
_REMOTE_FETCH_TIMEOUT = 5

# Retry codes for catalog fetches. Extends the standard transient codes with 404
# because GitHub Releases assets can briefly return 404 during the --clobber
# re-upload window or CDN propagation delay.
_CATALOG_RETRY_CODES = frozenset([404, 408, 429, 500, 502, 503, 504])

# Per-provider TTL cache for remote catalog fetches.
# Initialized lazily in _get_remote_cache() so the TTL reads the env var at first use.
_remote_cache: cachetools.TTLCache | None = None


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


class ModelInfo(TypedDict, total=False):
    """Flat model info used internally by cost_per_token, get_models, etc.

    All fields are optional (total=False) since not every model has every field.
    For example, embedding models may lack output pricing, and some models may
    not have deprecation dates or cache pricing.
    """

    mode: str | None
    supports_function_calling: bool
    supports_vision: bool
    supports_reasoning: bool
    supports_prompt_caching: bool
    supports_response_schema: bool
    max_input_tokens: int
    max_output_tokens: int
    max_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float
    cache_creation_input_token_cost: float
    deprecation_date: str


class ModelDict(TypedDict):
    """Model dictionary returned by get_models() and the gateway API.

    All fields are always present (some may be None).
    """

    model: str
    provider: str | None
    mode: str | None
    supports_function_calling: bool
    supports_vision: bool
    supports_reasoning: bool
    supports_prompt_caching: bool
    supports_response_schema: bool
    max_input_tokens: int | None
    max_output_tokens: int | None
    input_cost_per_token: float | None
    output_cost_per_token: float | None
    deprecation_date: str | None


# --- MLflow-native catalog schema TypedDicts (matches per-provider JSON files) ---


class CatalogContextWindow(TypedDict, total=False):
    max_input: int
    max_output: int
    max_tokens: int


class CatalogPricingTier(TypedDict, total=False):
    input_per_million_tokens: float
    output_per_million_tokens: float
    cache_read_per_million_tokens: float
    cache_write_per_million_tokens: float


class CatalogLongContextTier(CatalogPricingTier, total=False):
    threshold_tokens: int


class CatalogPricing(CatalogPricingTier, total=False):
    service_tiers: dict[str, CatalogPricingTier]
    long_context: list[CatalogLongContextTier]


class CatalogCapabilities(TypedDict, total=False):
    function_calling: bool
    vision: bool
    reasoning: bool
    prompt_caching: bool
    response_schema: bool


class CatalogModelEntry(TypedDict, total=False):
    mode: str
    context_window: CatalogContextWindow
    pricing: CatalogPricing
    capabilities: CatalogCapabilities
    deprecation_date: str


class CatalogFile(TypedDict):
    schema_version: str
    models: dict[str, CatalogModelEntry]


def _flatten_catalog_entry(entry: CatalogModelEntry) -> ModelInfo:
    """Convert an MLflow-native catalog entry to the flat ModelInfo format."""
    info: ModelInfo = {"mode": entry.get("mode")}

    if cw := entry.get("context_window"):
        if (v := cw.get("max_input")) is not None:
            info["max_input_tokens"] = v
        if (v := cw.get("max_output")) is not None:
            info["max_output_tokens"] = v
        if (v := cw.get("max_tokens")) is not None:
            info["max_tokens"] = v

    if pricing := entry.get("pricing"):
        if (v := pricing.get("input_per_million_tokens")) is not None:
            info["input_cost_per_token"] = v / 1_000_000
        if (v := pricing.get("output_per_million_tokens")) is not None:
            info["output_cost_per_token"] = v / 1_000_000
        if (v := pricing.get("cache_read_per_million_tokens")) is not None:
            info["cache_read_input_token_cost"] = v / 1_000_000
        if (v := pricing.get("cache_write_per_million_tokens")) is not None:
            info["cache_creation_input_token_cost"] = v / 1_000_000

    if caps := entry.get("capabilities"):
        info["supports_function_calling"] = caps.get("function_calling", False)
        info["supports_vision"] = caps.get("vision", False)
        info["supports_reasoning"] = caps.get("reasoning", False)
        info["supports_prompt_caching"] = caps.get("prompt_caching", False)
        info["supports_response_schema"] = caps.get("response_schema", False)

    if dep := entry.get("deprecation_date"):
        info["deprecation_date"] = dep

    return info


def _catalog_pkg() -> Path:
    return importlib.resources.files(__package__).joinpath("model_catalog")


@functools.lru_cache(maxsize=1)
def _list_provider_names() -> list[str]:
    """Return provider names available in the bundled catalog (cheap directory listing)."""
    try:
        return [p.stem for p in _catalog_pkg().glob("*.json") if p.is_file()]
    except (FileNotFoundError, TypeError):
        return []


def _parse_catalog_models(catalog: CatalogFile) -> dict[str, ModelInfo]:
    return {
        name: _flatten_catalog_entry(entry) for name, entry in catalog.get("models", {}).items()
    }


def _get_remote_cache() -> cachetools.TTLCache:
    global _remote_cache
    if _remote_cache is None:
        _remote_cache = cachetools.TTLCache(maxsize=256, ttl=MLFLOW_MODEL_CATALOG_CACHE_TTL.get())
    return _remote_cache


def _fetch_remote_provider(provider: str) -> dict[str, ModelInfo] | None:
    """Try to fetch a single provider's catalog from the configured URL with TTL caching.

    Supports ``http(s)://`` URLs (GitHub Releases, CDNs) and ``file://`` paths
    (for air-gapped / mirrored environments). Set ``MLFLOW_MODEL_CATALOG_URI``
    to an empty string to disable remote fetch entirely.
    """
    base_url = MLFLOW_MODEL_CATALOG_URI.get()
    if not base_url:
        return None

    cache = _get_remote_cache()
    if provider in cache:
        return cache[provider] or None

    url = f"{base_url.rstrip('/')}/{provider}.json"
    parsed = urllib.parse.urlparse(url)

    match parsed.scheme:
        case "file":
            result = _fetch_local_provider(provider, Path(urllib.request.url2pathname(parsed.path)))
        case "http" | "https":
            result = _fetch_http_provider(provider, url)
        case _:
            raise ValueError(
                f"Unsupported MLFLOW_MODEL_CATALOG_URI scheme: {parsed.scheme!r}. "
                f"Expected 'http', 'https', or 'file'. Got URI: {base_url}"
            )

    # Cache failures as empty dict so we don't retry on every call within the TTL
    cache[provider] = result or {}
    return result


def _fetch_local_provider(provider: str, path: Path) -> dict[str, ModelInfo] | None:
    try:
        catalog: CatalogFile = json.loads(path.read_text("utf-8"))
        return _parse_catalog_models(catalog)
    except Exception:
        _logger.debug("Failed to read local catalog for %s", provider, exc_info=True)
        return None


def _fetch_http_provider(provider: str, url: str) -> dict[str, ModelInfo] | None:
    try:
        resp = cloud_storage_http_request(
            "GET",
            url,
            max_retries=_REMOTE_FETCH_MAX_RETRIES,
            backoff_factor=1,
            retry_codes=_CATALOG_RETRY_CODES,
            timeout=_REMOTE_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        catalog: CatalogFile = resp.json()
        return _parse_catalog_models(catalog)
    except Exception:
        _logger.debug("Failed to fetch remote catalog for %s", provider, exc_info=True)
        return None


@functools.lru_cache(maxsize=128)
def _load_bundled_provider(provider: str) -> dict[str, ModelInfo]:
    """Load a single provider's catalog from the bundled package resources."""
    resource = _catalog_pkg().joinpath(f"{provider}.json")
    try:
        with importlib.resources.as_file(resource) as path, path.open(encoding="utf-8") as f:
            catalog: CatalogFile = json.load(f)
            return _parse_catalog_models(catalog)
    except (FileNotFoundError, TypeError):
        return {}


def _load_provider(provider: str) -> dict[str, ModelInfo]:
    """Load a provider's model catalog, trying remote first then bundled fallback."""
    if remote := _fetch_remote_provider(provider):
        return remote
    return _load_bundled_provider(provider)


def _lookup_model_info(model: str, custom_llm_provider: str | None = None) -> ModelInfo | None:
    """Look up model cost info, loading only the relevant provider file."""
    bare_model = model.split("/", 1)[-1]

    if custom_llm_provider:
        return _load_provider(custom_llm_provider).get(bare_model)

    # No provider given — scan bundled providers only (no remote fetch)
    # to avoid O(N) network requests across all providers.
    fallback = None
    for provider in _list_provider_names():
        if info := _load_bundled_provider(provider).get(bare_model):
            if info.get("input_cost_per_token"):
                return info
            if fallback is None:
                fallback = info
    return fallback


def cost_per_token(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    custom_llm_provider: str | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> tuple[float, float] | None:
    """Calculate cost per token using the bundled model price data.

    Returns:
        A tuple of (input_cost, output_cost) in USD, or None if the model is not found.
    """
    info = _lookup_model_info(model, custom_llm_provider)
    if info is None:
        return None

    input_cost_per_token = info.get("input_cost_per_token", 0.0)
    output_cost_per_token = info.get("output_cost_per_token", 0.0)

    # In this function, prompt_tokens is expected to include cache tokens, so we subtract
    # cache_read and cache_creation to get the regular (non-cached) portion, then price each
    # category at its own rate.
    cache_read = cache_read_input_tokens or 0
    cache_creation = cache_creation_input_tokens or 0
    regular_input_tokens = max(prompt_tokens - cache_read - cache_creation, 0)

    input_cost = regular_input_tokens * input_cost_per_token
    if cache_read > 0:
        input_cost += cache_read * info.get("cache_read_input_token_cost", input_cost_per_token)
    if cache_creation > 0:
        input_cost += cache_creation * info.get(
            "cache_creation_input_token_cost", input_cost_per_token
        )
    output_cost = completion_tokens * output_cost_per_token

    return input_cost, output_cost


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
# Includes both user-provided credential modes and a default credential chain mode
# that uses ambient server credentials (instance profile, IRSA, ECS task role, etc.).
_PROVIDER_AUTH_MODES: dict[str, dict[str, AuthModeDict]] = {
    "bedrock": {
        "api_key": {
            "display_name": "API Key",
            "description": "Use Amazon Bedrock API Key (bearer token)",
            "default": True,
            "fields": [
                {
                    "name": "api_key",
                    "description": "Amazon Bedrock API Key",
                    "secret": True,
                    "required": True,
                },
                {
                    "name": "aws_region_name",
                    "description": "AWS Region",
                    "secret": False,
                    "required": True,
                },
            ],
        },
        "access_keys": {
            "display_name": "Access Keys",
            "description": "Use AWS Access Key ID and Secret Access Key",
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
            "description": "Assume an IAM role using the server's ambient credentials "
            "(instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)",
            "fields": [
                {
                    "name": "aws_role_name",
                    "description": "IAM Role ARN to assume",
                    "secret": False,
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
        "default_chain": {
            "display_name": "Default Credential Chain",
            "description": "Use the server's default AWS credentials "
            "(instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)",
            "fields": [
                {
                    "name": "aws_role_name",
                    "description": "IAM Role ARN to assume (optional, for cross-account access)",
                    "secret": False,
                    "required": False,
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
                    "required": True,
                },
            ],
        },
        # TODO: uncomment this once it's supported by OpenAIConfig
        # "service_principal": {
        #     "display_name": "Service Principal",
        #     "description": "Use Azure AD Service Principal (client credentials)",
        #     "runtime_auth": "azure_service_principal",
        #     "fields": [
        #         {
        #             "name": "client_secret",
        #             "description": "Azure AD Client Secret",
        #             "secret": True,
        #             "required": True,
        #         },
        #         {
        #             "name": "api_base",
        #             "description": "Azure OpenAI endpoint URL",
        #             "secret": False,
        #             "required": True,
        #         },
        #         {
        #             "name": "client_id",
        #             "description": "Azure AD Application (Client) ID",
        #             "secret": False,
        #             "required": True,
        #         },
        #         {
        #             "name": "tenant_id",
        #             "description": "Azure AD Tenant ID",
        #             "secret": False,
        #             "required": True,
        #         },
        #         {
        #             "name": "api_version",
        #             "description": "API version (e.g., 2024-02-01)",
        #             "secret": False,
        #             "required": False,
        #             "default": "2024-02-01",
        #         },
        #     ],
        # },
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
    "sagemaker": {
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
                    "required": True,
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
                    "required": True,
                },
            ],
        },
        "default_chain": {
            "display_name": "Default Credential Chain",
            "description": "Use the server's default AWS credentials "
            "(instance profile, IRSA, ECS task role, ~/.aws/credentials, etc.)",
            "fields": [
                {
                    "name": "aws_role_name",
                    "description": "IAM Role ARN to assume (optional, for cross-account access)",
                    "secret": False,
                    "required": False,
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
    },
}

_BEDROCK_PROVIDERS = {"bedrock", "bedrock_converse"}


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
            {
                "name": "api_base",
                "description": f"{provider.title()} API Base URL",
                "secret": False,
                "required": False,
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

    if not is_provider_allowed(provider):
        _logger.debug(
            "Provider '%s' blocked by MLFLOW_GATEWAY_ALLOWED_PROVIDERS",
            provider,
        )
        raise MlflowException.invalid_parameter_value(
            f"Provider '{provider}' is not allowed by the current gateway provider policy."
        )

    provider = normalize_provider_name(provider.lower())
    config_provider = "bedrock" if provider in _BEDROCK_PROVIDERS else provider

    if config_provider in _PROVIDER_AUTH_MODES:
        auth_modes: list[AuthModeResponseDict] = []
        default_mode: str | None = None
        for mode_id, mode_config in _PROVIDER_AUTH_MODES[config_provider].items():
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


_EXCLUDED_PROVIDERS = {"bedrock_converse"}

# Providers that should be consolidated into a single provider.
# For example, vertex_ai-llama_models, vertex_ai-anthropic, etc. should all be
# consolidated into vertex_ai to be used by the AI Gateway.
_PROVIDER_CONSOLIDATION = {
    "vertex_ai": lambda p: p == "vertex_ai" or p.startswith("vertex_ai-"),
}


def _normalize_provider(provider: str) -> str:
    """
    Normalize provider name by consolidating variants into a single provider.

    For example, vertex_ai-llama_models -> vertex_ai
    """
    for normalized, matcher in _PROVIDER_CONSOLIDATION.items():
        if matcher(provider):
            return normalized
    return provider


def get_all_providers() -> list[str]:
    """
    Get a list of all providers.

    Provider variants are consolidated into a single provider (e.g., all vertex_ai-*
    variants are returned as just vertex_ai).
    """
    providers = set()
    for provider in _list_provider_names():
        if provider in _EXCLUDED_PROVIDERS:
            continue
        providers.add(_normalize_provider(provider))
    return filter_providers(list(providers))


def get_models(provider: str | None = None) -> list[ModelDict]:
    """
    Get a list of models from LiteLLM, optionally filtered by provider.

    Returns models that support chat, completion, or embedding capabilities,
    excluding image generation, audio, and other non-text services.

    Args:
        provider: Optional provider name to filter by (e.g., 'openai', 'anthropic').
                  When filtering by a consolidated provider (e.g., 'vertex_ai'),
                  all variant providers are included (e.g., 'vertex_ai-anthropic').

    Returns:
        List of model dictionaries with keys:
            - model: Model name
            - provider: Provider name (normalized, e.g., vertex_ai instead of vertex_ai-anthropic)
            - mode: Model mode (e.g., 'chat', 'completion', 'embedding')
            - supports_function_calling: Whether model supports tool/function calling
            - supports_vision: Whether model supports image/vision input
            - supports_reasoning: Whether model supports extended thinking (o1-style)
            - supports_prompt_caching: Whether model supports prompt caching
            - supports_response_schema: Whether model supports structured JSON output
            - max_input_tokens: Maximum input context window size
            - max_output_tokens: Maximum output token limit
            - input_cost_per_token: Cost per input token (USD)
            - output_cost_per_token: Cost per output token (USD)
            - deprecation_date: Date when model will be deprecated (if known)
    """
    if provider:
        # Fast path: only load provider files that match the filter
        matching = (
            p
            for p in _list_provider_names()
            if _normalize_provider(p) == provider and p not in _EXCLUDED_PROVIDERS
        )
    else:
        matching = (p for p in _list_provider_names() if p not in _EXCLUDED_PROVIDERS)

    entries = (
        (model_name, file_provider, info)
        for file_provider in matching
        for model_name, info in _load_provider(file_provider).items()
    )
    return _extract_models(entries, provider_filter=provider)


def _extract_models(
    entries: Iterator[tuple[str, str | None, ModelInfo]],
    provider_filter: str | None = None,
) -> list[ModelDict]:
    # Use dict to dedupe models by (provider, model_name) key
    models_dict: dict[tuple[str | None, str], ModelDict] = {}
    for model_name, entry_provider, info in entries:
        normalized_provider = _normalize_provider(entry_provider) if entry_provider else None

        # Filter by provider (matching against the normalized provider name)
        if provider_filter and normalized_provider != provider_filter:
            continue

        if normalized_provider and not is_provider_allowed(normalized_provider):
            continue

        mode = info.get("mode")
        if mode not in _SUPPORTED_MODEL_MODES:
            continue

        # Model names sometimes include the provider prefix, e.g. "gemini/gemini-2.5-flash"
        # Strip the normalized provider prefix if present
        if normalized_provider and model_name.startswith(f"{normalized_provider}/"):
            model_name = model_name.removeprefix(f"{normalized_provider}/")

        # Skip fine-tuned model variants (e.g. "ft:gpt-4o-2024-08-06:org::id")
        if model_name.startswith("ft:"):
            continue

        # Dedupe by (provider, model_name) - keep the first occurrence
        key = (normalized_provider, model_name)
        if key in models_dict:
            continue

        models_dict[key] = _build_model_dict(model_name, normalized_provider, mode, info)

    return list(models_dict.values())


def _build_model_dict(
    model_name: str, provider: str | None, mode: str | None, info: ModelInfo
) -> ModelDict:
    return {
        "model": model_name,
        "provider": provider,
        "mode": mode,
        "supports_function_calling": info.get("supports_function_calling", False),
        "supports_vision": info.get("supports_vision", False),
        "supports_reasoning": info.get("supports_reasoning", False),
        "supports_prompt_caching": info.get("supports_prompt_caching", False),
        "supports_response_schema": info.get("supports_response_schema", False),
        "max_input_tokens": info.get("max_input_tokens"),
        "max_output_tokens": info.get("max_output_tokens"),
        "input_cost_per_token": info.get("input_cost_per_token"),
        "output_cost_per_token": info.get("output_cost_per_token"),
        "deprecation_date": info.get("deprecation_date"),
    }


# Azure OpenAI environment variable names (matching litellm convention)
AZURE_API_KEY_ENV_VAR = "AZURE_API_KEY"
AZURE_API_BASE_ENV_VAR = "AZURE_API_BASE"
AZURE_API_VERSION_ENV_VAR = "AZURE_API_VERSION"

# Mapping of core providers to their environment variable names for credentials/config fields
_CORE_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "azure": {
        "api_key": AZURE_API_KEY_ENV_VAR,
        "api_base": AZURE_API_BASE_ENV_VAR,
        "api_version": AZURE_API_VERSION_ENV_VAR,
    },
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "bedrock": {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_session_token": "AWS_SESSION_TOKEN",  # Optional
    },
}

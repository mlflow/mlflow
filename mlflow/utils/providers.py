import importlib.util
from typing import Any

# Check if litellm is available without importing it (litellm import is slow ~2.7s)
LITELLM_AVAILABLE = importlib.util.find_spec("litellm") is not None


def _get_litellm():
    """Lazy import of litellm to avoid slow startup times."""
    import litellm

    return litellm


def _get_litellm_model_cost():
    """Lazy import of model_cost from litellm."""
    from litellm import model_cost

    return model_cost


def _get_litellm_provider_fields():
    """Lazy import of get_provider_fields from litellm."""
    from litellm import get_provider_fields

    return get_provider_fields


PROVIDER_CREDENTIAL_MAPPING = {
    "databricks": [
        {
            "name": "DATABRICKS_API_KEY",
            "type": "string",
            "description": "Databricks API Key",
            "required": True,
        },
        {
            "name": "DATABRICKS_API_BASE",
            "type": "string",
            "description": "Databricks API Base URL",
            "required": True,
        },
    ],
    "openai": [
        {
            "name": "OPENAI_API_KEY",
            "type": "string",
            "description": "OpenAI API Key",
            "required": True,
        },
    ],
    "anthropic": [
        {
            "name": "ANTHROPIC_API_KEY",
            "type": "string",
            "description": "Anthropic API Key",
            "required": True,
        },
    ],
    "bedrock": [
        {
            "name": "AWS_ACCESS_KEY_ID",
            "type": "string",
            "description": "AWS Access Key ID",
            "required": True,
        },
        {
            "name": "AWS_SECRET_ACCESS_KEY",
            "type": "string",
            "description": "AWS Secret Access Key",
            "required": True,
        },
        {
            "name": "AWS_REGION_NAME",
            "type": "string",
            "description": "AWS Region Name",
            "required": False,
        },
    ],
    "azure": [
        {
            "name": "AZURE_API_KEY",
            "type": "string",
            "description": "Azure API Key",
            "required": True,
        },
        {
            "name": "AZURE_API_BASE",
            "type": "string",
            "description": "Azure API Base URL",
            "required": True,
        },
        {
            "name": "AZURE_API_VERSION",
            "type": "string",
            "description": "Azure API Version",
            "required": False,
        },
    ],
    "cohere": [
        {
            "name": "COHERE_API_KEY",
            "type": "string",
            "description": "Cohere API Key",
            "required": True,
        },
    ],
    "groq": [
        {
            "name": "GROQ_API_KEY",
            "type": "string",
            "description": "Groq API Key",
            "required": True,
        },
    ],
    "mistral": [
        {
            "name": "MISTRAL_API_KEY",
            "type": "string",
            "description": "Mistral API Key",
            "required": True,
        },
    ],
    "together_ai": [
        {
            "name": "TOGETHERAI_API_KEY",
            "type": "string",
            "description": "Together AI API Key",
            "required": True,
        },
    ],
    "fireworks_ai": [
        {
            "name": "FIREWORKS_API_KEY",
            "type": "string",
            "description": "Fireworks AI API Key",
            "required": True,
        },
    ],
    "replicate": [
        {
            "name": "REPLICATE_API_KEY",
            "type": "string",
            "description": "Replicate API Key",
            "required": True,
        },
    ],
    "huggingface": [
        {
            "name": "HUGGINGFACE_API_KEY",
            "type": "string",
            "description": "Hugging Face API Key",
            "required": True,
        },
    ],
    "ai21": [
        {
            "name": "AI21_API_KEY",
            "type": "string",
            "description": "AI21 API Key",
            "required": True,
        },
    ],
    "palm": [
        {
            "name": "PALM_API_KEY",
            "type": "string",
            "description": "PaLM API Key",
            "required": True,
        },
    ],
    "perplexity": [
        {
            "name": "PERPLEXITYAI_API_KEY",
            "type": "string",
            "description": "Perplexity AI API Key",
            "required": True,
        },
    ],
    "anyscale": [
        {
            "name": "ANYSCALE_API_KEY",
            "type": "string",
            "description": "Anyscale API Key",
            "required": True,
        },
    ],
    "deepinfra": [
        {
            "name": "DEEPINFRA_API_KEY",
            "type": "string",
            "description": "DeepInfra API Key",
            "required": True,
        },
    ],
    "nvidia_nim": [
        {
            "name": "NVIDIA_NIM_API_KEY",
            "type": "string",
            "description": "NVIDIA NIM API Key",
            "required": True,
        },
    ],
    "cerebras": [
        {
            "name": "CEREBRAS_API_KEY",
            "type": "string",
            "description": "Cerebras API Key",
            "required": True,
        },
    ],
    "vertex_ai": [
        {
            "name": "VERTEX_PROJECT",
            "type": "string",
            "description": "Vertex AI Project ID",
            "required": True,
        },
        {
            "name": "VERTEX_LOCATION",
            "type": "string",
            "description": "Vertex AI Location",
            "required": True,
        },
    ],
}


def get_provider_config_response(provider: str) -> dict[str, Any]:
    """
    Get provider configuration formatted for API response.

    Args:
        provider: The LiteLLM provider name (e.g., 'openai', 'anthropic', 'databricks')

    Returns:
        dict with keys:
            - credential_name: Primary credential field name (first field)
            - auth_fields: Additional auth fields (remaining fields)
            - capabilities: Provider capabilities

    Raises:
        ImportError: If litellm is not installed
        ValueError: If provider is not valid or not provided
    """
    if not provider:
        raise ValueError("Provider parameter is required")

    config_data = get_provider_config(provider)
    credential_fields = config_data.get("credential_fields", [])

    credential_name = None
    auth_fields = []

    if credential_fields:
        credential_name = credential_fields[0]["name"]
        auth_fields = [
            {
                "name": field["name"],
                "required": field.get("required", True),
                "type": field.get("type", "string"),
                "description": field.get("description", ""),
            }
            for field in credential_fields[1:]
        ]

    return {
        "credential_name": credential_name,
        "auth_fields": auth_fields,
        "capabilities": config_data.get("capabilities", {}),
    }


def get_provider_config(provider: str) -> dict[str, Any]:
    """
    Get configuration requirements for a LiteLLM provider.

    Args:
        provider: The LiteLLM provider name (e.g., 'openai', 'anthropic', 'databricks')

    Returns:
        dict with keys:
            - credential_fields: list of required credential fields
            - capabilities: dict of aggregated capabilities from model_cost
    """
    if not LITELLM_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    get_provider_fields = _get_litellm_provider_fields()
    provider_fields = get_provider_fields(provider)

    credential_fields = []
    if provider_fields and len(provider_fields) > 0:
        credential_fields = [
            {
                "name": field["field_name"],
                "type": field.get("field_type", "string"),
                "description": field.get("field_description", ""),
                "required": True,
            }
            for field in provider_fields
        ]
    elif provider in PROVIDER_CREDENTIAL_MAPPING:
        credential_fields = PROVIDER_CREDENTIAL_MAPPING[provider]
    else:
        api_key_name = f"{provider.upper()}_API_KEY"
        credential_fields = [
            {
                "name": api_key_name,
                "type": "string",
                "description": f"{provider.title()} API Key",
                "required": True,
            }
        ]

    capabilities = {}
    litellm = _get_litellm()
    if hasattr(litellm, "model_cost"):
        model_cost = _get_litellm_model_cost()
        for _, info in model_cost.items():
            if info.get("litellm_provider") == provider:
                for cap_key in [
                    "supports_function_calling",
                    "supports_vision",
                    "supports_audio_input",
                    "supports_audio_output",
                    "supports_system_messages",
                    "supports_prompt_caching",
                ]:
                    if cap_key in info and info[cap_key]:
                        capabilities[cap_key] = True

                if "max_tokens" in info and info["max_tokens"]:
                    current_max = capabilities.get("max_tokens", 0)
                    try:
                        model_max = int(info["max_tokens"])
                        capabilities["max_tokens"] = max(current_max, model_max)
                    except (ValueError, TypeError):
                        pass

    return {
        "credential_fields": credential_fields,
        "capabilities": capabilities,
    }


def get_all_providers() -> list[str]:
    """
    Get a list of all LiteLLM providers that have chat/completion capabilities.

    Only returns providers that have at least one chat or completion model,
    excluding providers that only offer embedding, image generation, or other
    non-chat services.

    Returns:
        List of provider names that support chat/completion

    Raises:
        ImportError: If litellm is not installed
    """
    if not LITELLM_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    litellm = _get_litellm()
    if not hasattr(litellm, "model_cost"):
        return []

    model_cost = _get_litellm_model_cost()
    providers = set()
    for _, info in model_cost.items():
        mode = info.get("mode")
        # Only include providers with chat/completion models
        if mode in ("chat", "completion", None):
            if provider := info.get("litellm_provider"):
                providers.add(provider)

    return list(providers)


def get_models(provider: str | None = None) -> list[dict[str, Any]]:
    """
    Get a list of chat-capable models from LiteLLM, optionally filtered by provider.

    Only returns models that support chat/completion (text generation), excluding
    vision-only, audio-only, embedding, and image generation models.

    Args:
        provider: Optional provider name to filter by (e.g., 'openai', 'anthropic')

    Returns:
        List of model dictionaries with keys:
            - model: Model name
            - provider: Provider name
            - mode: Model mode (e.g., 'chat', 'completion')
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
    if not LITELLM_AVAILABLE:
        raise ImportError("LiteLLM is not installed. Install it with: pip install 'mlflow[genai]'")

    litellm = _get_litellm()
    if not hasattr(litellm, "model_cost"):
        return []

    model_cost = _get_litellm_model_cost()
    models = []
    for model_name, info in model_cost.items():
        if provider and info.get("litellm_provider") != provider:
            continue

        # Only include chat/completion models (mode='chat' or 'completion')
        # Exclude embedding, image_generation, audio, etc.
        mode = info.get("mode")
        if mode not in ("chat", "completion", None):
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

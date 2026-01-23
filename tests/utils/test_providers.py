from unittest import mock

from mlflow.utils.providers import (
    _normalize_provider,
    get_all_providers,
    get_models,
)


def test_normalize_provider_normalizes_vertex_ai_variants():
    assert _normalize_provider("vertex_ai") == "vertex_ai"
    assert _normalize_provider("vertex_ai-anthropic") == "vertex_ai"
    assert _normalize_provider("vertex_ai-llama_models") == "vertex_ai"
    assert _normalize_provider("vertex_ai-mistral") == "vertex_ai"


def test_normalize_provider_does_not_normalize_other_providers():
    assert _normalize_provider("openai") == "openai"
    assert _normalize_provider("anthropic") == "anthropic"
    assert _normalize_provider("bedrock") == "bedrock"
    assert _normalize_provider("gemini") == "gemini"


@mock.patch("mlflow.utils.providers._get_model_cost")
def test_get_all_providers_consolidates_vertex_ai_variants(mock_model_cost):
    mock_model_cost.return_value = {
        "gpt-4o": {"litellm_provider": "openai", "mode": "chat"},
        "claude-3-5-sonnet": {"litellm_provider": "anthropic", "mode": "chat"},
        "gemini-1.5-pro": {"litellm_provider": "vertex_ai", "mode": "chat"},
        "vertex_ai/meta/llama-4-scout": {
            "litellm_provider": "vertex_ai-llama_models",
            "mode": "chat",
        },
        "vertex_ai/claude-3-5-sonnet": {
            "litellm_provider": "vertex_ai-anthropic",
            "mode": "chat",
        },
    }

    providers = get_all_providers()

    # vertex_ai-* variants should be consolidated into vertex_ai
    assert "vertex_ai" in providers
    assert "vertex_ai-llama_models" not in providers
    assert "vertex_ai-anthropic" not in providers
    assert "openai" in providers
    assert "anthropic" in providers


@mock.patch("mlflow.utils.providers._get_model_cost")
def test_get_models_normalizes_vertex_ai_provider_and_strips_prefix(mock_model_cost):
    mock_model_cost.return_value = {
        "vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas": {
            "litellm_provider": "vertex_ai-llama_models",
            "mode": "chat",
            "supports_function_calling": True,
        },
        "vertex_ai/claude-3-5-sonnet": {
            "litellm_provider": "vertex_ai-anthropic",
            "mode": "chat",
            "supports_function_calling": True,
        },
        "gemini-1.5-pro": {
            "litellm_provider": "vertex_ai",
            "mode": "chat",
            "supports_function_calling": True,
        },
    }

    models = get_models(provider="vertex_ai")

    assert len(models) == 3

    # Check that all providers are normalized to vertex_ai
    for model in models:
        assert model["provider"] == "vertex_ai"

    # Check that vertex_ai/ prefix is stripped from model names
    model_names = [m["model"] for m in models]
    assert "meta/llama-4-scout-17b-16e-instruct-maas" in model_names
    assert "claude-3-5-sonnet" in model_names
    assert "gemini-1.5-pro" in model_names

    # Ensure the original prefixed names are not present
    assert "vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas" not in model_names
    assert "vertex_ai/claude-3-5-sonnet" not in model_names


@mock.patch("mlflow.utils.providers._get_model_cost")
def test_get_models_filters_by_consolidated_provider(mock_model_cost):
    mock_model_cost.return_value = {
        "gpt-4o": {"litellm_provider": "openai", "mode": "chat"},
        "vertex_ai/meta/llama-4-scout": {
            "litellm_provider": "vertex_ai-llama_models",
            "mode": "chat",
        },
    }

    # Filtering by vertex_ai should include vertex_ai-* variants
    vertex_models = get_models(provider="vertex_ai")
    assert len(vertex_models) == 1
    assert vertex_models[0]["model"] == "meta/llama-4-scout"

    # Filtering by openai should not include vertex_ai models
    openai_models = get_models(provider="openai")
    assert len(openai_models) == 1
    assert openai_models[0]["model"] == "gpt-4o"


@mock.patch("mlflow.utils.providers._get_model_cost")
def test_get_models_does_not_modify_other_providers(mock_model_cost):
    mock_model_cost.return_value = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
        },
        "claude-3-5-sonnet": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_function_calling": True,
        },
    }

    models = get_models()

    openai_model = next(m for m in models if m["provider"] == "openai")
    assert openai_model["model"] == "gpt-4o"

    anthropic_model = next(m for m in models if m["provider"] == "anthropic")
    assert anthropic_model["model"] == "claude-3-5-sonnet"


@mock.patch("mlflow.utils.providers._get_model_cost")
def test_get_models_dedupes_models_after_normalization(mock_model_cost):
    # Same model appearing under different vertex_ai variants should be deduped
    mock_model_cost.return_value = {
        "gemini-3-flash-preview": {
            "litellm_provider": "vertex_ai",
            "mode": "chat",
            "supports_function_calling": True,
        },
        "vertex_ai/gemini-3-flash-preview": {
            "litellm_provider": "vertex_ai-chat-models",
            "mode": "chat",
            "supports_function_calling": True,
        },
    }

    models = get_models(provider="vertex_ai")

    # Should only have one gemini-3-flash-preview, not two
    model_names = [m["model"] for m in models]
    assert model_names.count("gemini-3-flash-preview") == 1
    assert len(models) == 1

from unittest import mock

import pytest

from mlflow.utils.providers import (
    _normalize_provider,
    cost_per_token,
    get_all_providers,
    get_models,
    get_provider_config_response,
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


def test_get_all_providers_consolidates_vertex_ai_variants():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_model_cost:
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


def test_get_models_normalizes_vertex_ai_provider_and_strips_prefix():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_model_cost:
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


def test_get_models_filters_by_consolidated_provider():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_model_cost:
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


def test_get_models_does_not_modify_other_providers():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_model_cost:
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


def test_get_models_dedupes_models_after_normalization():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_model_cost:
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


def test_get_provider_config_bedrock_has_default_chain():
    config = get_provider_config_response("bedrock")
    modes = {m["mode"] for m in config["auth_modes"]}
    assert "default_chain" in modes

    default_chain = next(m for m in config["auth_modes"] if m["mode"] == "default_chain")
    assert default_chain["display_name"] == "Default Credential Chain"
    assert default_chain["secret_fields"] == []
    assert all(not f["required"] for f in default_chain["config_fields"])


def test_get_provider_config_sagemaker_has_default_chain():
    config = get_provider_config_response("sagemaker")
    modes = {m["mode"] for m in config["auth_modes"]}
    assert "default_chain" in modes


def test_cost_per_token_basic():
    input_cost, output_cost = cost_per_token(
        model="gpt-4o", prompt_tokens=1000, completion_tokens=500
    )
    # gpt-4o: input_cost_per_token=2.5e-6, output_cost_per_token=1e-5
    assert input_cost == pytest.approx(0.0025)
    assert output_cost == pytest.approx(0.005)


def test_cost_per_token_with_provider_prefix():
    input_cost, output_cost = cost_per_token(
        model="gpt-4o", prompt_tokens=1000, completion_tokens=500, custom_llm_provider="openai"
    )
    assert input_cost == pytest.approx(0.0025)
    assert output_cost == pytest.approx(0.005)


def test_cost_per_token_cache_read_tokens():
    input_cost, output_cost = cost_per_token(
        model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=200,
    )
    # gpt-4o: cache_read_input_token_cost=1.25e-6
    # regular: (1000-200) * 2.5e-6 = 0.002
    # cache_read: 200 * 1.25e-6 = 0.00025
    assert input_cost == pytest.approx(0.00225)
    assert output_cost == pytest.approx(0.005)


def test_cost_per_token_cache_creation_tokens():
    input_cost, output_cost = cost_per_token(
        model="claude-sonnet-4-20250514",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_creation_input_tokens=300,
    )
    assert input_cost > 0
    assert output_cost > 0


def test_cost_per_token_zero_tokens():
    input_cost, output_cost = cost_per_token(model="gpt-4o", prompt_tokens=0, completion_tokens=0)
    assert input_cost == 0.0
    assert output_cost == 0.0


def test_cost_per_token_unknown_model_raises():
    with pytest.raises(ValueError, match="not found"):
        cost_per_token(model="totally-unknown-model", prompt_tokens=100)


def test_cost_per_token_unknown_model_with_provider_raises():
    with pytest.raises(ValueError, match="not found"):
        cost_per_token(
            model="totally-unknown-model",
            prompt_tokens=100,
            custom_llm_provider="unknown-provider",
        )


def test_cost_per_token_no_cache_cost_falls_back_to_input_rate():
    with mock.patch("mlflow.utils.providers._get_model_cost") as mock_cost:
        mock_cost.return_value = {
            "test-model": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
            }
        }
        input_cost, output_cost = cost_per_token(
            model="test-model",
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_input_tokens=200,
        )
        # No cache_read_input_token_cost, falls back to input_cost_per_token
        # regular: 800 * 1e-6 = 0.0008
        # cache_read: 200 * 1e-6 = 0.0002 (same rate as regular)
        assert input_cost == pytest.approx(0.001)
        assert output_cost == pytest.approx(0.001)

import json
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.providers import (
    _fetch_remote_provider,
    _flatten_catalog_entry,
    _get_remote_cache,
    _list_provider_names,
    _load_bundled_provider,
    _load_provider,
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


def test_list_provider_names_returns_bundled_providers():
    _list_provider_names.cache_clear()
    providers = _list_provider_names()
    assert len(providers) > 0
    assert "openai" in providers
    assert "anthropic" in providers
    assert "bedrock" in providers


def test_list_provider_names_excludes_non_json():
    _list_provider_names.cache_clear()
    providers = _list_provider_names()
    # __init__.py should not appear
    assert "__init__" not in providers
    for p in providers:
        assert not p.endswith(".py")


def test_load_provider_returns_models(monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", "")
    _load_bundled_provider.cache_clear()
    models = _load_provider("openai")
    assert len(models) > 0
    assert "gpt-4o" in models
    info = models["gpt-4o"]
    assert info["mode"] == "chat"
    assert "input_cost_per_token" in info
    assert info["input_cost_per_token"] > 0


def test_load_provider_returns_empty_for_unknown(monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", "")
    _load_bundled_provider.cache_clear()
    assert _load_provider("nonexistent_provider_xyz") == {}


def test_load_provider_flattens_pricing(monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", "")
    _load_bundled_provider.cache_clear()
    models = _load_provider("anthropic")
    model = next(iter(models.values()))
    # Should have flat ModelInfo keys, not nested pricing/capabilities
    assert "input_cost_per_token" in model or "mode" in model
    assert "pricing" not in model
    assert "context_window" not in model


def _mock_catalog(provider_data):
    """Context manager that mocks the per-provider catalog with the given data.

    ``provider_data`` is a dict mapping provider names to ``{model_name: info}`` dicts.
    """
    return (
        mock.patch(
            "mlflow.utils.providers._load_provider",
            side_effect=lambda p: provider_data.get(p, {}),
        ),
        mock.patch(
            "mlflow.utils.providers._list_provider_names",
            return_value=list(provider_data.keys()),
        ),
    )


def test_get_all_providers_consolidates_vertex_ai_variants():
    data = {
        "openai": {"gpt-4o": {"mode": "chat"}},
        "anthropic": {"claude-3-5-sonnet": {"mode": "chat"}},
        "vertex_ai": {"gemini-1.5-pro": {"mode": "chat"}},
        "vertex_ai-llama_models": {"meta/llama-4-scout": {"mode": "chat"}},
        "vertex_ai-anthropic": {"claude-3-5-sonnet": {"mode": "chat"}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        providers = get_all_providers()

        assert "vertex_ai" in providers
        assert "vertex_ai-llama_models" not in providers
        assert "vertex_ai-anthropic" not in providers
        assert "openai" in providers
        assert "anthropic" in providers


def test_get_models_normalizes_vertex_ai_provider_and_strips_prefix():
    data = {
        "vertex_ai-llama_models": {
            "meta/llama-4-scout-17b-16e-instruct-maas": {
                "mode": "chat",
                "supports_function_calling": True,
            }
        },
        "vertex_ai-anthropic": {
            "claude-3-5-sonnet": {"mode": "chat", "supports_function_calling": True}
        },
        "vertex_ai": {"gemini-1.5-pro": {"mode": "chat", "supports_function_calling": True}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        models = get_models(provider="vertex_ai")

        assert len(models) == 3
        for model in models:
            assert model["provider"] == "vertex_ai"

        model_names = [m["model"] for m in models]
        assert "meta/llama-4-scout-17b-16e-instruct-maas" in model_names
        assert "claude-3-5-sonnet" in model_names
        assert "gemini-1.5-pro" in model_names


def test_get_models_filters_by_consolidated_provider():
    data = {
        "openai": {"gpt-4o": {"mode": "chat"}},
        "vertex_ai-llama_models": {"meta/llama-4-scout": {"mode": "chat"}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        vertex_models = get_models(provider="vertex_ai")
        assert len(vertex_models) == 1
        assert vertex_models[0]["model"] == "meta/llama-4-scout"

        openai_models = get_models(provider="openai")
        assert len(openai_models) == 1
        assert openai_models[0]["model"] == "gpt-4o"


def test_get_models_does_not_modify_other_providers():
    data = {
        "openai": {"gpt-4o": {"mode": "chat", "supports_function_calling": True}},
        "anthropic": {"claude-3-5-sonnet": {"mode": "chat", "supports_function_calling": True}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        models = get_models()

        openai_model = next(m for m in models if m["provider"] == "openai")
        assert openai_model["model"] == "gpt-4o"

        anthropic_model = next(m for m in models if m["provider"] == "anthropic")
        assert anthropic_model["model"] == "claude-3-5-sonnet"


def test_get_models_dedupes_models_after_normalization():
    data = {
        "vertex_ai": {
            "gemini-3-flash-preview": {"mode": "chat", "supports_function_calling": True}
        },
        "vertex_ai-chat-models": {
            "gemini-3-flash-preview": {"mode": "chat", "supports_function_calling": True}
        },
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        models = get_models(provider="vertex_ai")

        model_names = [m["model"] for m in models]
        assert model_names.count("gemini-3-flash-preview") == 1
        assert len(models) == 1


def test_get_all_providers_with_allowed_filter(monkeypatch):
    data = {
        "openai": {"gpt-4o": {"mode": "chat"}},
        "anthropic": {"claude-3-5-sonnet": {"mode": "chat"}},
        "gemini": {"gemini-1.5-pro": {"mode": "chat"}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai,anthropic")
        providers = get_all_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" not in providers


def test_get_models_filters_with_allowed_providers(monkeypatch):
    data = {
        "openai": {"gpt-4o": {"mode": "chat", "supports_function_calling": True}},
        "anthropic": {"claude-3-5-sonnet": {"mode": "chat", "supports_function_calling": True}},
        "gemini": {"gemini-1.5-pro": {"mode": "chat", "supports_function_calling": True}},
    }
    with _mock_catalog(data)[0], _mock_catalog(data)[1]:
        monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
        models = get_models()
        providers_in_result = {m["provider"] for m in models}
        assert providers_in_result == {"openai"}


def test_get_provider_config_rejects_provider_not_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "anthropic")
    with pytest.raises(MlflowException, match="not allowed"):
        get_provider_config_response("openai")


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


_MOCK_PROVIDER_DATA = {
    "test_provider": {
        "test-model": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 5e-7,
            "cache_creation_input_token_cost": 3e-6,
        },
    },
    "openai": {
        "test-provider-model": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
        },
    },
}


def _mock_load_provider(provider):
    return _MOCK_PROVIDER_DATA.get(provider, {})


@pytest.fixture
def mock_model_cost():
    with (
        mock.patch(
            "mlflow.utils.providers._load_provider", side_effect=_mock_load_provider
        ) as m_load,
        mock.patch(
            "mlflow.utils.providers._load_bundled_provider", side_effect=_mock_load_provider
        ),
        mock.patch(
            "mlflow.utils.providers._list_provider_names",
            return_value=list(_MOCK_PROVIDER_DATA.keys()),
        ),
    ):
        yield m_load


def test_cost_per_token_basic(mock_model_cost):
    input_cost, output_cost = cost_per_token(
        model="test-model", prompt_tokens=1000, completion_tokens=500
    )
    # input: 1000 * 1e-6 = 0.001, output: 500 * 2e-6 = 0.001
    assert input_cost == pytest.approx(0.001)
    assert output_cost == pytest.approx(0.001)


def test_cost_per_token_with_provider_prefix(mock_model_cost):
    # "test-provider-model" only exists under "openai/" prefix, so provider lookup is exercised
    input_cost, output_cost = cost_per_token(
        model="test-provider-model",
        prompt_tokens=1000,
        completion_tokens=500,
        custom_llm_provider="openai",
    )
    assert input_cost == pytest.approx(0.001)
    assert output_cost == pytest.approx(0.001)


def test_cost_per_token_strips_provider_prefix(mock_model_cost):
    # "openai/test-model" should resolve to "test-model" by stripping the prefix
    input_cost, output_cost = cost_per_token(
        model="openai/test-model", prompt_tokens=1000, completion_tokens=500
    )
    assert input_cost == pytest.approx(0.001)
    assert output_cost == pytest.approx(0.001)


def test_cost_per_token_cache_read_tokens(mock_model_cost):
    input_cost, output_cost = cost_per_token(
        model="test-model",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_read_input_tokens=200,
    )
    # regular: (1000-200) * 1e-6 = 0.0008
    # cache_read: 200 * 5e-7 = 0.0001
    assert input_cost == pytest.approx(0.0009)
    assert output_cost == pytest.approx(0.001)


def test_cost_per_token_cache_creation_tokens(mock_model_cost):
    input_cost, output_cost = cost_per_token(
        model="test-model",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_creation_input_tokens=300,
    )
    # regular: (1000-300) * 1e-6 = 0.0007
    # cache_creation: 300 * 3e-6 = 0.0009
    assert input_cost == pytest.approx(0.0016)
    assert output_cost == pytest.approx(0.001)


def test_cost_per_token_zero_tokens(mock_model_cost):
    input_cost, output_cost = cost_per_token(
        model="test-model", prompt_tokens=0, completion_tokens=0
    )
    assert input_cost == 0.0
    assert output_cost == 0.0


def test_cost_per_token_unknown_model_returns_none(mock_model_cost):
    assert cost_per_token(model="totally-unknown-model", prompt_tokens=100) is None


def test_cost_per_token_unknown_model_with_provider_returns_none(mock_model_cost):
    assert (
        cost_per_token(
            model="totally-unknown-model",
            prompt_tokens=100,
            custom_llm_provider="unknown-provider",
        )
        is None
    )


def test_cost_per_token_no_cache_cost_falls_back_to_input_rate():
    no_cache_data = {
        "nocache_provider": {
            "test-model": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
            }
        }
    }
    with (
        mock.patch(
            "mlflow.utils.providers._load_provider",
            side_effect=lambda p: no_cache_data.get(p, {}),
        ),
        mock.patch(
            "mlflow.utils.providers._load_bundled_provider",
            side_effect=lambda p: no_cache_data.get(p, {}),
        ),
        mock.patch(
            "mlflow.utils.providers._list_provider_names",
            return_value=list(no_cache_data.keys()),
        ),
    ):
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


def test_flatten_catalog_entry():
    entry = {
        "mode": "chat",
        "context_window": {"max_input": 128000, "max_output": 16384},
        "pricing": {
            "input_per_million_tokens": 2.5,
            "output_per_million_tokens": 10.0,
            "cache_read_per_million_tokens": 1.25,
            "cache_write_per_million_tokens": 5.0,
        },
        "capabilities": {
            "function_calling": True,
            "vision": True,
            "reasoning": False,
            "prompt_caching": True,
            "response_schema": True,
        },
        "deprecation_date": "2026-01-01",
    }
    info = _flatten_catalog_entry(entry)
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 16384
    assert info["input_cost_per_token"] == pytest.approx(2.5e-6)
    assert info["output_cost_per_token"] == pytest.approx(1e-5)
    assert info["cache_read_input_token_cost"] == pytest.approx(1.25e-6)
    assert info["cache_creation_input_token_cost"] == pytest.approx(5e-6)
    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True
    assert info["supports_reasoning"] is False
    assert info["deprecation_date"] == "2026-01-01"


def test_load_bundled_provider_returns_data():
    _load_bundled_provider.cache_clear()
    result = _load_bundled_provider("openai")
    assert len(result) > 0
    assert "gpt-4o" in result
    info = result["gpt-4o"]
    assert info["mode"] == "chat"
    assert "input_cost_per_token" in info


def test_load_provider_uses_remote_when_available():
    remote_data = {"test-model": {"mode": "chat", "input_cost_per_token": 1e-6}}
    with mock.patch(
        "mlflow.utils.providers._fetch_remote_provider", return_value=remote_data
    ) as mock_remote:
        result = _load_provider("openai")
        mock_remote.assert_called_once_with("openai")
        assert result is remote_data


def test_load_provider_falls_back_to_bundled_when_remote_fails():
    with mock.patch(
        "mlflow.utils.providers._fetch_remote_provider", return_value=None
    ) as mock_remote:
        result = _load_provider("openai")
        mock_remote.assert_called_once_with("openai")
        assert len(result) > 0
        assert "gpt-4o" in result


def test_fetch_remote_provider_disabled_when_url_empty(monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", "")
    assert _fetch_remote_provider("openai") is None


def test_fetch_remote_provider_supports_file_url(tmp_path, monkeypatch):
    catalog = {
        "schema_version": "1.0",
        "models": {"test-model": {"mode": "chat", "pricing": {"input_per_million_tokens": 1.0}}},
    }
    (tmp_path / "test_provider.json").write_text(json.dumps(catalog))
    monkeypatch.setenv("MLFLOW_MODEL_CATALOG_URI", tmp_path.as_uri())
    _get_remote_cache().clear()
    result = _fetch_remote_provider("test_provider")
    assert result is not None
    assert "test-model" in result

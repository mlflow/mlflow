import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from update_model_catalog import (
    _extract_long_context_pricing,
    _extract_modality_pricing,
    _extract_service_tiers,
    _extract_tool_pricing,
    _is_deprecated,
    _normalize_provider,
    _transform_entry,
    convert,
)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("vertex_ai", "vertex_ai"),
        ("vertex_ai-anthropic", "vertex_ai"),
        ("vertex_ai-llama_models", "vertex_ai"),
        ("vertex_ai-chat-models", "vertex_ai"),
        ("bedrock", "bedrock"),
    ],
)
def test_normalize_provider(provider, expected):
    assert _normalize_provider(provider) == expected


def test_transform_entry_chat_model():
    info = {
        "mode": "chat",
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 1.5e-5,
        "cache_read_input_token_cost": 3e-7,
        "cache_creation_input_token_cost": 3.75e-6,
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        "max_tokens": 64000,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_prompt_caching": True,
        "supports_response_schema": True,
    }
    result = _transform_entry(info)
    assert result == {
        "mode": "chat",
        "context_window": {"max_input": 200000, "max_output": 64000, "max_tokens": 64000},
        "pricing": {
            "input_per_million_tokens": 3.0,
            "output_per_million_tokens": 15.0,
            "cache_read_per_million_tokens": 0.3,
            "cache_write_per_million_tokens": 3.75,
        },
        "capabilities": {
            "function_calling": True,
            "vision": True,
            "reasoning": True,
            "prompt_caching": True,
            "response_schema": True,
        },
    }


def test_transform_entry_skips_image_generation():
    info = {"mode": "image_generation", "input_cost_per_token": 1e-6}
    assert _transform_entry(info) is None


def test_transform_entry_includes_future_deprecation_date():
    info = {
        "mode": "chat",
        "deprecation_date": "2099-01-01",
    }
    result = _transform_entry(info)
    assert result is not None
    assert result["deprecation_date"] == "2099-01-01"


def test_transform_entry_skips_past_deprecation_date():
    info = {
        "mode": "chat",
        "deprecation_date": "2020-01-01",
    }
    assert _transform_entry(info) is None


def test_is_deprecated_past_date():
    assert _is_deprecated({"deprecation_date": "2020-01-01"}) is True


def test_is_deprecated_future_date():
    assert _is_deprecated({"deprecation_date": "2099-01-01"}) is False


def test_is_deprecated_no_date():
    assert _is_deprecated({}) is False


def test_is_deprecated_invalid_date():
    assert _is_deprecated({"deprecation_date": "not-a-date"}) is False


def test_transform_entry_with_service_tiers():
    info = {
        "mode": "chat",
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 8e-6,
        "cache_read_input_token_cost": 5e-7,
        "input_cost_per_token_flex": 1e-6,
        "output_cost_per_token_flex": 4e-6,
        "cache_read_input_token_cost_flex": 2.5e-7,
        "input_cost_per_token_priority": 3.5e-6,
        "output_cost_per_token_priority": 1.4e-5,
        "input_cost_per_token_batches": 1e-6,
        "output_cost_per_token_batches": 4e-6,
    }
    result = _transform_entry(info)
    tiers = result["pricing"]["service_tiers"]
    assert tiers["flex"] == {
        "input_per_million_tokens": 1.0,
        "output_per_million_tokens": 4.0,
        "cache_read_per_million_tokens": 0.25,
    }
    assert tiers["priority"] == {
        "input_per_million_tokens": 3.5,
        "output_per_million_tokens": 14.0,
    }
    assert tiers["batch"] == {
        "input_per_million_tokens": 1.0,
        "output_per_million_tokens": 4.0,
    }


def test_transform_entry_with_long_context():
    info = {
        "mode": "chat",
        "input_cost_per_token": 1.25e-6,
        "output_cost_per_token": 1e-5,
        "input_cost_per_token_above_200k_tokens": 2.5e-6,
        "output_cost_per_token_above_200k_tokens": 1.5e-5,
        "cache_read_input_token_cost_above_200k_tokens": 2.5e-7,
    }
    result = _transform_entry(info)
    long_ctx = result["pricing"]["long_context"]
    assert len(long_ctx) == 1
    assert long_ctx[0] == {
        "threshold_tokens": 200000,
        "input_per_million_tokens": 2.5,
        "output_per_million_tokens": 15.0,
        "cache_read_per_million_tokens": 0.25,
    }


def test_extract_long_context_multiple_thresholds():
    info = {
        "input_cost_per_token_above_128k_tokens": 1e-6,
        "output_cost_per_token_above_128k_tokens": 2e-6,
        "input_cost_per_token_above_256k_tokens": 2e-6,
        "output_cost_per_token_above_256k_tokens": 4e-6,
    }
    result = _extract_long_context_pricing(info)
    assert len(result) == 2
    assert result[0]["threshold_tokens"] == 128000
    assert result[1]["threshold_tokens"] == 256000


def test_extract_service_tiers_empty_when_no_tiers():
    info = {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
    assert _extract_service_tiers(info) == {}


def test_extract_modality_pricing():
    info = {
        "input_cost_per_audio_token": 7e-7,
        "output_cost_per_audio_token": 1.1e-6,
        "cache_read_input_audio_token_cost": 2e-7,
        "cache_creation_input_audio_token_cost": 4e-7,
    }
    assert _extract_modality_pricing(info) == {
        "audio": {
            "input_per_million_tokens": 0.7,
            "output_per_million_tokens": 1.1,
            "cache_read_per_million_tokens": 0.2,
            "cache_write_per_million_tokens": 0.4,
        }
    }


def test_extract_modality_pricing_skips_reasoning():
    info = {
        "input_cost_per_audio_token": 7e-7,
        "output_cost_per_reasoning_token": 4e-7,
    }
    assert _extract_modality_pricing(info) == {"audio": {"input_per_million_tokens": 0.7}}


def test_extract_tool_pricing():
    info = {
        "computer_use_input_cost_per_1k_tokens": 0.00225,
        "computer_use_output_cost_per_1k_tokens": 0.009,
        "search_context_cost_per_query": {
            "search_context_size_low": 0.01,
            "search_context_size_medium": 0.01,
            "search_context_size_high": 0.01,
        },
        "tool_use_system_prompt_tokens": 159,
    }
    assert _extract_tool_pricing(info) == {
        "computer_use": {
            "input_per_million_tokens": 2.25,
            "output_per_million_tokens": 9.0,
        },
        "search_context_per_query": {
            "search_context_size_low": 0.01,
            "search_context_size_medium": 0.01,
            "search_context_size_high": 0.01,
        },
        "tool_use_system_prompt_tokens": 159,
    }


def test_transform_entry_with_modality_and_tool_pricing():
    info = {
        "mode": "chat",
        "input_cost_per_token": 1e-7,
        "output_cost_per_token": 4e-7,
        "input_cost_per_audio_token": 7e-7,
        "computer_use_input_cost_per_1k_tokens": 0.00225,
        "tool_use_system_prompt_tokens": 159,
    }
    result = _transform_entry(info)
    assert result["pricing"]["modality"] == {"audio": {"input_per_million_tokens": 0.7}}
    assert result["pricing"]["tooling"] == {
        "computer_use": {"input_per_million_tokens": 2.25},
        "tool_use_system_prompt_tokens": 159,
    }


def test_convert_end_to_end(tmp_path):
    input_data = {
        "sample_spec": {"mode": "chat"},
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2.5e-6,
            "output_cost_per_token": 1e-5,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "supports_function_calling": True,
            "supports_vision": True,
        },
        "openai/gpt-4o-mini": {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1.5e-7,
            "output_cost_per_token": 6e-7,
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
        },
        "claude-3-5-sonnet": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 1.5e-5,
        },
        "dall-e-3": {
            "litellm_provider": "openai",
            "mode": "image_generation",
        },
        "ft:gpt-4o:org::id": {
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "bedrock_converse/model": {
            "litellm_provider": "bedrock_converse",
            "mode": "chat",
        },
    }

    output_dir = tmp_path / "output"

    stats = convert(input_data, output_dir)

    assert stats == {"anthropic": 1, "bedrock": 1, "openai": 2}
    assert (output_dir / "openai.json").exists()
    assert (output_dir / "anthropic.json").exists()
    assert (output_dir / "bedrock.json").exists()
    assert not (output_dir / "bedrock_converse.json").exists()

    openai_catalog = json.loads((output_dir / "openai.json").read_text())
    assert openai_catalog["schema_version"] == "1.0"
    assert "gpt-4o" in openai_catalog["models"]
    assert "gpt-4o-mini" in openai_catalog["models"]
    # Fine-tuned and image_generation should be excluded
    assert "ft:gpt-4o:org::id" not in openai_catalog["models"]
    assert "dall-e-3" not in openai_catalog["models"]


def test_convert_preserves_existing_models(tmp_path):

    input_data = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 2.5e-6,
        },
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Pre-populate with a manually-added model
    existing_catalog = {
        "schema_version": "1.0",
        "models": {
            "custom-model": {
                "mode": "chat",
                "pricing": {"input_per_million_tokens": 1.0},
                "capabilities": {
                    "function_calling": False,
                    "vision": False,
                    "reasoning": False,
                    "prompt_caching": False,
                    "response_schema": False,
                },
            }
        },
    }
    (output_dir / "openai.json").write_text(json.dumps(existing_catalog))

    stats = convert(input_data, output_dir)

    catalog = json.loads((output_dir / "openai.json").read_text())
    # Both upstream and manually-added models should be present
    assert "gpt-4o" in catalog["models"]
    assert "custom-model" in catalog["models"]
    assert stats["openai"] == 2


def test_convert_preserves_community_provider(tmp_path):

    input_data = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Pre-populate with a community-maintained provider
    community_catalog = {
        "schema_version": "1.0",
        "models": {
            "my-model": {
                "mode": "chat",
                "capabilities": {
                    "function_calling": False,
                    "vision": False,
                    "reasoning": False,
                    "prompt_caching": False,
                    "response_schema": False,
                },
            }
        },
    }
    (output_dir / "custom_provider.json").write_text(json.dumps(community_catalog))

    stats = convert(input_data, output_dir)

    # Community provider should be preserved
    assert (output_dir / "custom_provider.json").exists()
    assert "custom_provider" in stats
    assert stats["custom_provider"] == 1


def test_convert_skips_deprecated_models(tmp_path):
    input_data = {
        "old-model": {
            "litellm_provider": "openai",
            "mode": "chat",
            "deprecation_date": "2020-01-01",
        },
        "new-model": {
            "litellm_provider": "openai",
            "mode": "chat",
            "deprecation_date": "2099-12-31",
        },
    }

    output_dir = tmp_path / "output"

    stats = convert(input_data, output_dir)

    catalog = json.loads((output_dir / "openai.json").read_text())
    assert "old-model" not in catalog["models"]
    assert "new-model" in catalog["models"]
    assert stats["openai"] == 1


def test_convert_upstream_overrides_existing_model(tmp_path):

    input_data = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 9.99e-6,
        },
    }

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Pre-populate with old pricing
    existing_catalog = {
        "schema_version": "1.0",
        "models": {
            "gpt-4o": {
                "mode": "chat",
                "pricing": {"input_per_million_tokens": 1.0},
                "capabilities": {
                    "function_calling": False,
                    "vision": False,
                    "reasoning": False,
                    "prompt_caching": False,
                    "response_schema": False,
                },
            }
        },
    }
    (output_dir / "openai.json").write_text(json.dumps(existing_catalog))

    convert(input_data, output_dir)

    catalog = json.loads((output_dir / "openai.json").read_text())
    # Upstream price should win
    assert catalog["models"]["gpt-4o"]["pricing"]["input_per_million_tokens"] == pytest.approx(9.99)

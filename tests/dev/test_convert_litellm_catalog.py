import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from convert_litellm_catalog import _normalize_provider, _transform_entry, convert


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
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_prompt_caching": True,
        "supports_response_schema": True,
    }
    result = _transform_entry(info)
    assert result == {
        "mode": "chat",
        "context_window": {"max_input": 200000, "max_output": 64000},
        "pricing": {
            "input_per_token": 3e-6,
            "output_per_token": 1.5e-5,
            "cache_read_per_token": 3e-7,
            "cache_write_per_token": 3.75e-6,
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


def test_transform_entry_includes_deprecation_date():
    info = {
        "mode": "chat",
        "deprecation_date": "2026-01-01",
    }
    result = _transform_entry(info)
    assert result["deprecation_date"] == "2026-01-01"


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

    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(input_data))
    output_dir = tmp_path / "output"

    stats = convert(input_path, output_dir)

    assert stats == {"anthropic": 1, "openai": 2}
    assert (output_dir / "openai.json").exists()
    assert (output_dir / "anthropic.json").exists()
    assert not (output_dir / "bedrock_converse.json").exists()

    openai_catalog = json.loads((output_dir / "openai.json").read_text())
    assert openai_catalog["schema_version"] == "1.0"
    assert "gpt-4o" in openai_catalog["models"]
    assert "gpt-4o-mini" in openai_catalog["models"]
    # Fine-tuned and image_generation should be excluded
    assert "ft:gpt-4o:org::id" not in openai_catalog["models"]
    assert "dall-e-3" not in openai_catalog["models"]


def test_convert_removes_stale_files(tmp_path):
    input_data = {
        "gpt-4o": {
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(input_data))
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create a stale file
    (output_dir / "stale_provider.json").write_text("{}")

    convert(input_path, output_dir)

    assert not (output_dir / "stale_provider.json").exists()
    assert (output_dir / "openai.json").exists()

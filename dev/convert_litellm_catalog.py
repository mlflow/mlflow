"""Convert LiteLLM model_prices_and_context_window.json to MLflow-native per-provider catalog.

Usage:
    uv run python dev/convert_litellm_catalog.py [--input PATH] [--output-dir PATH]

Reads the LiteLLM JSON (default: mlflow/utils/model_prices_and_context_window.json),
transforms it into the MLflow-native schema, and writes one JSON file per provider
into the output directory (default: mlflow/utils/model_catalog/).
"""

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Modes that MLflow cares about for gateway / cost tracking
_SUPPORTED_MODES = {"chat", "completion", "embedding"}

# Providers that should be consolidated into a canonical name
_PROVIDER_CONSOLIDATION = {
    "vertex_ai-anthropic": "vertex_ai",
    "vertex_ai-llama_models": "vertex_ai",
    "vertex_ai-mistral": "vertex_ai",
    "vertex_ai-chat-models": "vertex_ai",
    "vertex_ai-text-models": "vertex_ai",
    "vertex_ai-code-chat-models": "vertex_ai",
    "vertex_ai-code-text-models": "vertex_ai",
    "vertex_ai-embedding-models": "vertex_ai",
    "vertex_ai-vision-models": "vertex_ai",
}

# Providers to exclude from the catalog entirely
_EXCLUDED_PROVIDERS = {"bedrock_converse"}


def _normalize_provider(provider: str) -> str:
    if provider in _PROVIDER_CONSOLIDATION:
        return _PROVIDER_CONSOLIDATION[provider]
    if provider.startswith("vertex_ai-"):
        return "vertex_ai"
    return provider


def _transform_entry(info: dict) -> dict | None:
    """Transform a single LiteLLM model entry into MLflow-native schema."""
    mode = info.get("mode")
    if mode not in _SUPPORTED_MODES:
        return None

    pricing = {}
    if (v := info.get("input_cost_per_token")) is not None:
        pricing["input_per_token"] = v
    if (v := info.get("output_cost_per_token")) is not None:
        pricing["output_per_token"] = v
    if (v := info.get("cache_read_input_token_cost")) is not None:
        pricing["cache_read_per_token"] = v
    if (v := info.get("cache_creation_input_token_cost")) is not None:
        pricing["cache_write_per_token"] = v

    capabilities = {
        "function_calling": info.get("supports_function_calling", False),
        "vision": info.get("supports_vision", False),
        "reasoning": info.get("supports_reasoning", False),
        "prompt_caching": info.get("supports_prompt_caching", False),
        "response_schema": info.get("supports_response_schema", False),
    }

    context_window = {}
    if (v := info.get("max_input_tokens")) is not None:
        context_window["max_input"] = v
    if (v := info.get("max_output_tokens")) is not None:
        context_window["max_output"] = v

    entry = {"mode": mode}
    if context_window:
        entry["context_window"] = context_window
    if pricing:
        entry["pricing"] = pricing
    entry["capabilities"] = capabilities
    if dep := info.get("deprecation_date"):
        entry["deprecation_date"] = dep

    return entry


def convert(input_path: Path, output_dir: Path) -> dict[str, int]:
    """Convert LiteLLM JSON to per-provider MLflow catalog files.

    Returns a dict mapping provider names to model counts.
    """
    with input_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Group by provider
    providers: dict[str, dict[str, dict]] = {}
    seen: set[tuple[str, str]] = set()

    for key, info in raw.items():
        if key == "sample_spec":
            continue

        provider = info.get("litellm_provider")
        if not provider:
            continue
        if provider in _EXCLUDED_PROVIDERS:
            continue

        provider = _normalize_provider(provider)
        model_name = key.split("/", 1)[-1]

        # Skip fine-tuned variants
        if model_name.startswith("ft:"):
            continue

        # Dedupe by (provider, model_name)
        dedup_key = (provider, model_name)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entry = _transform_entry(info)
        if entry is None:
            continue

        providers.setdefault(provider, {})[model_name] = entry

    # Write per-provider files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale provider files
    existing_files = {p.stem for p in output_dir.glob("*.json")}
    new_files = set(providers.keys())
    for stale in existing_files - new_files:
        (output_dir / f"{stale}.json").unlink()

    stats = {}
    for provider, models in sorted(providers.items()):
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "models": dict(sorted(models.items())),
        }
        out_path = output_dir / f"{provider}.json"
        out_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        stats[provider] = len(models)

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("mlflow/utils/model_prices_and_context_window.json"),
        help="Path to LiteLLM model prices JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mlflow/utils/model_catalog"),
        help="Output directory for per-provider JSON files",
    )
    args = parser.parse_args()

    stats = convert(args.input, args.output_dir)
    total = sum(stats.values())
    print(f"Converted {total} models across {len(stats)} providers:")
    for provider, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {provider}: {count}")


if __name__ == "__main__":
    main()

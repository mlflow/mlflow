"""Update the MLflow model catalog from upstream data sources.

Usage:
    uv run python dev/update_model_catalog.py [--output-dir PATH]

Fetches the LiteLLM model_prices_and_context_window.json from GitHub,
transforms it into the MLflow-native schema, and merges the results into the
per-provider catalog files in the output directory (default: mlflow/utils/model_catalog/).

Models present in the upstream source always take precedence over existing entries
(upstream wins). Models not present in the upstream source are preserved, allowing
community additions to coexist with automated upstream syncs.
Models with a deprecation_date in the past are dropped during conversion.
"""

import argparse
import json
import re
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
    "bedrock_converse": "bedrock",
}


def _normalize_provider(provider: str) -> str:
    if provider in _PROVIDER_CONSOLIDATION:
        return _PROVIDER_CONSOLIDATION[provider]
    if provider.startswith("vertex_ai-"):
        return "vertex_ai"
    return provider


_PER_MILLION = 1_000_000
_PER_THOUSAND = 1_000


def _to_per_million(cost_per_token: float) -> float:
    return round(cost_per_token * _PER_MILLION, 10)


def _extract_base_pricing(info: dict[str, Any]) -> dict[str, Any]:
    """Extract base pricing fields from a LiteLLM entry (converted to per-million-tokens)."""
    pricing = {}
    if (v := info.get("input_cost_per_token")) is not None:
        pricing["input_per_million_tokens"] = _to_per_million(v)
    if (v := info.get("output_cost_per_token")) is not None:
        pricing["output_per_million_tokens"] = _to_per_million(v)
    if (v := info.get("cache_read_input_token_cost")) is not None:
        pricing["cache_read_per_million_tokens"] = _to_per_million(v)
    if (v := info.get("cache_creation_input_token_cost")) is not None:
        pricing["cache_write_per_million_tokens"] = _to_per_million(v)
    return pricing


_MODALITY_INPUT = re.compile(r"^input_cost_per_([a-z0-9_]+)_token$")
_MODALITY_OUTPUT = re.compile(r"^output_cost_per_([a-z0-9_]+)_token$")
_MODALITY_CACHE_READ = re.compile(r"^cache_read_input_([a-z0-9_]+)_token_cost$")
_MODALITY_CACHE_WRITE = re.compile(r"^cache_creation_input_([a-z0-9_]+)_token_cost$")
_MODALITY_CACHE_READ_ALT = re.compile(r"^cache_read_input_token_cost_per_([a-z0-9_]+)_token$")
_EXCLUDED_MODALITIES = {"reasoning"}


def _extract_modality_pricing(info: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract modality-specific pricing (audio/image/etc) as per-million-token rates."""
    modalities: dict[str, dict[str, float]] = {}
    for k, v in info.items():
        if m := _MODALITY_INPUT.match(k):
            modality = m.group(1)
            if modality in _EXCLUDED_MODALITIES:
                continue
            modalities.setdefault(modality, {})["input_per_million_tokens"] = _to_per_million(v)
        elif m := _MODALITY_OUTPUT.match(k):
            modality = m.group(1)
            if modality in _EXCLUDED_MODALITIES:
                continue
            modalities.setdefault(modality, {})["output_per_million_tokens"] = _to_per_million(v)
        elif m := _MODALITY_CACHE_READ.match(k):
            modality = m.group(1)
            if modality in _EXCLUDED_MODALITIES:
                continue
            modality_entry = modalities.setdefault(modality, {})
            modality_entry["cache_read_per_million_tokens"] = _to_per_million(v)
        elif m := _MODALITY_CACHE_WRITE.match(k):
            modality = m.group(1)
            if modality in _EXCLUDED_MODALITIES:
                continue
            modality_entry = modalities.setdefault(modality, {})
            modality_entry["cache_write_per_million_tokens"] = _to_per_million(v)
        elif m := _MODALITY_CACHE_READ_ALT.match(k):
            modality = m.group(1)
            if modality in _EXCLUDED_MODALITIES:
                continue
            modality_entry = modalities.setdefault(modality, {})
            modality_entry["cache_read_per_million_tokens"] = _to_per_million(v)

    return modalities


def _extract_tool_pricing(info: dict[str, Any]) -> dict[str, Any]:
    """Extract tool-related pricing and tool-use token overhead fields."""
    tool_pricing: dict[str, Any] = {}

    if (v := info.get("computer_use_input_cost_per_1k_tokens")) is not None:
        tool_pricing.setdefault("computer_use", {})["input_per_million_tokens"] = round(
            v * _PER_THOUSAND, 10
        )
    if (v := info.get("computer_use_output_cost_per_1k_tokens")) is not None:
        tool_pricing.setdefault("computer_use", {})["output_per_million_tokens"] = round(
            v * _PER_THOUSAND, 10
        )
    if (v := info.get("search_context_cost_per_query")) is not None:
        tool_pricing["search_context_per_query"] = v
    if (v := info.get("tool_use_system_prompt_tokens")) is not None:
        tool_pricing["tool_use_system_prompt_tokens"] = v

    return tool_pricing


# LiteLLM uses suffixes like _batches, _batch_requests, _flex, _priority
_TIER_PATTERNS = {
    "batch": re.compile(r"^(input|output)_cost_per_token_(batches|batch_requests)$"),
    "flex": re.compile(r"^(input|output)_cost_per_token_flex$"),
    "priority": re.compile(r"^(input|output)_cost_per_token_priority$"),
}

_TIER_CACHE_PATTERNS = {
    "batch": re.compile(r"^cache_read_input_token_cost_(batches|batch_requests)$"),
    "flex": re.compile(r"^cache_read_input_token_cost_flex$"),
    "priority": re.compile(r"^cache_read_input_token_cost_priority$"),
}


def _extract_service_tiers(info: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract service tier pricing overrides (batch, flex, priority)."""
    tiers: dict[str, dict[str, float]] = {}
    for tier_name, pattern in _TIER_PATTERNS.items():
        for k, v in info.items():
            if m := pattern.match(k):
                direction = m.group(1)  # "input" or "output"
                tiers.setdefault(tier_name, {})[f"{direction}_per_million_tokens"] = (
                    _to_per_million(v)
                )

    for tier_name, pattern in _TIER_CACHE_PATTERNS.items():
        for k, v in info.items():
            if pattern.match(k):
                tiers.setdefault(tier_name, {})["cache_read_per_million_tokens"] = _to_per_million(
                    v
                )

    return tiers


# Matches keys like input_cost_per_token_above_200k_tokens or
# cache_read_input_token_cost_above_128k_tokens
_LONG_CTX_INPUT = re.compile(r"^input_cost_per_token_above_(\d+[km]?)_tokens$")
_LONG_CTX_OUTPUT = re.compile(r"^output_cost_per_token_above_(\d+[km]?)_tokens$")
_LONG_CTX_CACHE_READ = re.compile(r"^cache_read_input_token_cost_above_(\d+[km]?)_tokens$")
_LONG_CTX_CACHE_WRITE = re.compile(r"^cache_creation_input_token_cost_above_(\d+[km]?)_tokens$")


def _parse_threshold(s: str) -> int:
    """Convert threshold string like '200k', '128k', or '1m' to token count."""
    s = s.lower()
    if s.endswith("m"):
        return int(s[:-1]) * 1_000_000
    if s.endswith("k"):
        return int(s[:-1]) * 1_000
    return int(s)


def _extract_long_context_pricing(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract long-context pricing tiers as a list of threshold overrides."""
    # Group by threshold
    thresholds: dict[int, dict[str, Any]] = {}
    for k, v in info.items():
        if m := _LONG_CTX_INPUT.match(k):
            t = _parse_threshold(m.group(1))
            thresholds.setdefault(t, {"threshold_tokens": t})["input_per_million_tokens"] = (
                _to_per_million(v)
            )
        elif m := _LONG_CTX_OUTPUT.match(k):
            t = _parse_threshold(m.group(1))
            thresholds.setdefault(t, {"threshold_tokens": t})["output_per_million_tokens"] = (
                _to_per_million(v)
            )
        elif m := _LONG_CTX_CACHE_READ.match(k):
            t = _parse_threshold(m.group(1))
            thresholds.setdefault(t, {"threshold_tokens": t})["cache_read_per_million_tokens"] = (
                _to_per_million(v)
            )
        elif m := _LONG_CTX_CACHE_WRITE.match(k):
            t = _parse_threshold(m.group(1))
            thresholds.setdefault(t, {"threshold_tokens": t})["cache_write_per_million_tokens"] = (
                _to_per_million(v)
            )

    return sorted(thresholds.values(), key=lambda x: x["threshold_tokens"])


def _is_deprecated(info: dict[str, Any]) -> bool:
    """Return True if the model's deprecation_date is in the past."""
    dep = info.get("deprecation_date")
    if not dep:
        return False
    try:
        return datetime.strptime(dep, "%Y-%m-%d").date() < date.today()
    except ValueError:
        return False


def _transform_entry(info: dict[str, Any]) -> dict[str, Any] | None:
    """Transform a single LiteLLM model entry into MLflow-native schema."""
    mode = info.get("mode")
    if mode not in _SUPPORTED_MODES:
        return None

    if _is_deprecated(info):
        return None

    pricing = _extract_base_pricing(info)

    if service_tiers := _extract_service_tiers(info):
        pricing["service_tiers"] = service_tiers

    if long_context := _extract_long_context_pricing(info):
        pricing["long_context"] = long_context

    if modality_pricing := _extract_modality_pricing(info):
        pricing["modality"] = modality_pricing

    if tool_pricing := _extract_tool_pricing(info):
        pricing["tooling"] = tool_pricing

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
    if (v := info.get("max_tokens")) is not None:
        context_window["max_tokens"] = v

    entry = {"mode": mode}
    if context_window:
        entry["context_window"] = context_window
    if pricing:
        entry["pricing"] = pricing
    entry["capabilities"] = capabilities
    if dep := info.get("deprecation_date"):
        entry["deprecation_date"] = dep

    return entry


_LEGACY_PRICING_KEY_MAP = {
    "input_per_token": "input_per_million_tokens",
    "output_per_token": "output_per_million_tokens",
    "cache_read_per_token": "cache_read_per_million_tokens",
    "cache_write_per_token": "cache_write_per_million_tokens",
}


def _migrate_pricing_block(pricing: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy *_per_token keys to *_per_million_tokens in a flat pricing block."""
    result = {}
    for k, v in pricing.items():
        if k in _LEGACY_PRICING_KEY_MAP:
            result[_LEGACY_PRICING_KEY_MAP[k]] = _to_per_million(v)
        else:
            result[k] = v
    return result


def _migrate_legacy_pricing(entry: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy *_per_token pricing keys to *_per_million_tokens in a catalog entry.

    Applies the migration at the top level of the pricing block and recursively within
    service_tiers, long_context, and modality sub-sections.
    """
    if "pricing" not in entry:
        return entry

    entry = {**entry}
    pricing = _migrate_pricing_block(entry["pricing"])

    if "service_tiers" in pricing:
        pricing["service_tiers"] = {
            tier: _migrate_pricing_block(tier_data)
            for tier, tier_data in pricing["service_tiers"].items()
        }

    if "long_context" in pricing:
        pricing["long_context"] = [_migrate_pricing_block(ctx) for ctx in pricing["long_context"]]

    if "modality" in pricing:
        pricing["modality"] = {
            mod: _migrate_pricing_block(mod_data) for mod, mod_data in pricing["modality"].items()
        }

    entry["pricing"] = pricing
    return entry


_LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)


def _fetch_litellm_catalog() -> dict[str, Any]:
    """Download the latest LiteLLM model catalog from GitHub."""
    print(f"Fetching {_LITELLM_URL} ...")
    with urllib.request.urlopen(_LITELLM_URL, timeout=30) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return data


def convert(raw: dict[str, Any], output_dir: Path) -> dict[str, int]:
    """Convert upstream catalog dict to per-provider MLflow catalog files.

    Returns a dict mapping provider names to model counts.
    """

    today = date.today().isoformat()

    # Load existing catalog files first so we can detect which entries have changed
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_catalogs: dict[str, dict[str, Any]] = {}
    for provider_file in output_dir.glob("*.json"):
        try:
            existing = json.loads(provider_file.read_text(encoding="utf-8"))
            existing_catalogs[provider_file.stem] = existing.get("models", {})
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read existing {provider_file.name}: {e}")

    # Group by provider
    providers: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()

    for key, info in raw.items():
        if key == "sample_spec":
            continue

        provider = info.get("litellm_provider")
        if not provider:
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

        # Determine last_updated_at: carry over existing date if entry is unchanged;
        # set today if no existing date (first-time backfill or new entry)
        existing_entry = existing_catalogs.get(provider, {}).get(model_name)
        if existing_entry is not None:
            existing_without_last_updated_at = {
                k: v for k, v in existing_entry.items() if k != "last_updated_at"
            }
            if entry == existing_without_last_updated_at:
                # Entry is unchanged; preserve existing last_updated_at or set today if absent
                entry["last_updated_at"] = existing_entry.get("last_updated_at") or today
            else:
                entry["last_updated_at"] = today
        else:
            entry["last_updated_at"] = today

        providers.setdefault(provider, {})[model_name] = entry

    # Merge with existing catalogs: preserve models not in upstream (community additions)
    for provider, existing_models in existing_catalogs.items():
        if provider not in providers:
            providers[provider] = {}
        for model_name, entry in existing_models.items():
            if model_name not in providers.get(provider, {}):
                migrated = _migrate_legacy_pricing(entry)
                if "last_updated_at" not in migrated:
                    migrated = {**migrated, "last_updated_at": today}
                providers.setdefault(provider, {})[model_name] = migrated

    stats = {}
    for provider, models in sorted(providers.items()):
        if not models:
            continue
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "models": dict(sorted(models.items())),
        }
        out_path = output_dir / f"{provider}.json"
        out_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        stats[provider] = len(models)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mlflow/utils/model_catalog"),
        help="Output directory for per-provider JSON files",
    )
    args = parser.parse_args()

    raw = _fetch_litellm_catalog()
    stats = convert(raw, args.output_dir)
    total = sum(stats.values())
    print(f"Converted {total} models across {len(stats)} providers:")
    for provider, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {provider}: {count}")


if __name__ == "__main__":
    main()

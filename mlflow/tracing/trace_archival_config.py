from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlflow.environment_variables import MLFLOW_TRACE_ARCHIVAL_CONFIG
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.utils.trace_archival import (
    _parse_trace_archival_long_retention_allowlist,
)
from mlflow.utils.validation import (
    _validate_trace_archival_repository_support,
    _validate_trace_archival_retention_string,
)

_TRACE_ARCHIVAL_CONFIG_KEY = "trace_archival"
_TRACE_ARCHIVAL_INTERVAL_SECONDS_DEFAULT = 300
_TRACE_ARCHIVAL_POSITIVE_INT_MIN = 1
_TRACE_ARCHIVAL_INTERVAL_SECONDS_MAX = 86400
_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_TTL_SECONDS = 5.0

_logger = logging.getLogger(__name__)
_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_LOCK = threading.Lock()
_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE: _TraceArchivalServerConfigCacheEntry | None = None


@dataclass(frozen=True, slots=True)
class TraceArchivalServerConfig:
    enabled: bool
    location: str
    retention: str
    long_retention_allowlist: tuple[str, ...] = ()
    interval_seconds: int = _TRACE_ARCHIVAL_INTERVAL_SECONDS_DEFAULT
    max_traces_per_pass: int | None = None


@dataclass(frozen=True, slots=True)
class _TraceArchivalServerConfigCacheEntry:
    config_path: str
    config: TraceArchivalServerConfig
    expires_at_monotonic: float


def get_trace_archival_server_config() -> TraceArchivalServerConfig | None:
    global _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE

    config_path = MLFLOW_TRACE_ARCHIVAL_CONFIG.get()
    if config_path is None or not config_path.strip():
        with _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_LOCK:
            _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE = None
        return None

    normalized_config_path = str(Path(config_path))
    now = time.monotonic()
    with _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_LOCK:
        cached = _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE
        if (
            cached is not None
            and cached.config_path == normalized_config_path
            and now < cached.expires_at_monotonic
        ):
            return cached.config

        cache_was_for_same_path = (
            cached is not None and cached.config_path == normalized_config_path
        )
        previous_config = cached.config if cache_was_for_same_path else None

        try:
            config = load_trace_archival_server_config(normalized_config_path)
        except MlflowException:
            if previous_config is not None:
                _logger.warning(
                    "Failed to refresh trace archival config; continuing to use the last valid "
                    "config.",
                    exc_info=True,
                )
                _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE = _TraceArchivalServerConfigCacheEntry(
                    config_path=normalized_config_path,
                    config=previous_config,
                    expires_at_monotonic=now + _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_TTL_SECONDS,
                )
                return previous_config
            raise

        if previous_config is not None and previous_config != config:
            _logger.info(
                "Trace archival config changed; refreshed cached server settings.",
            )

        _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE = _TraceArchivalServerConfigCacheEntry(
            config_path=normalized_config_path,
            config=config,
            expires_at_monotonic=now + _TRACE_ARCHIVAL_SERVER_CONFIG_CACHE_TTL_SECONDS,
        )
        return config


def load_trace_archival_server_config(
    config_path: str | Path,
) -> TraceArchivalServerConfig:
    path = Path(config_path)
    raw_config = _load_yaml(path)
    trace_archival = _get_required_mapping(raw_config, path, _TRACE_ARCHIVAL_CONFIG_KEY)

    enabled = _get_required_bool(trace_archival, path, "enabled")
    location = _validate_trace_archival_repository_support(
        _get_required_value(trace_archival, path, "location"),
        parameter_name=f"{_TRACE_ARCHIVAL_CONFIG_KEY}.location",
    )
    retention = _validate_trace_archival_retention_string(
        _get_required_value(trace_archival, path, "retention"),
        parameter_name=f"{_TRACE_ARCHIVAL_CONFIG_KEY}.retention",
    )
    long_retention_allowlist = _parse_long_retention_allowlist(
        trace_archival.get("long_retention_allowlist"),
        path,
    )
    interval_seconds = _get_optional_bounded_positive_int(
        trace_archival,
        path,
        "interval_seconds",
        default=_TRACE_ARCHIVAL_INTERVAL_SECONDS_DEFAULT,
        maximum=_TRACE_ARCHIVAL_INTERVAL_SECONDS_MAX,
    )
    max_traces_per_pass = _get_optional_bounded_positive_int(
        trace_archival, path, "max_traces_per_pass"
    )

    return TraceArchivalServerConfig(
        enabled=enabled,
        location=location,
        retention=retention,
        long_retention_allowlist=long_retention_allowlist,
        interval_seconds=interval_seconds,
        max_traces_per_pass=max_traces_per_pass,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as e:
        raise _invalid_trace_archival_config(
            path,
            f"Failed to parse YAML: {e}",
        ) from e
    except OSError as e:
        raise _invalid_trace_archival_config(
            path,
            f"Failed to read config file: {e}",
        ) from e

    if not isinstance(payload, dict):
        raise _invalid_trace_archival_config(
            path,
            "Top-level YAML value must be a mapping containing 'trace_archival'.",
        )

    return payload


def _get_required_mapping(payload: dict[str, Any], path: Path, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise _invalid_trace_archival_config(
            path,
            f"Missing required '{key}' mapping.",
        )
    return value


def _get_required_value(payload: dict[str, Any], path: Path, key: str) -> Any:
    if key not in payload:
        raise _invalid_trace_archival_config(
            path,
            f"Missing required '{_TRACE_ARCHIVAL_CONFIG_KEY}.{key}' value.",
        )
    return payload[key]


def _get_required_bool(payload: dict[str, Any], path: Path, key: str) -> bool:
    value = _get_required_value(payload, path, key)
    if isinstance(value, bool):
        return value
    raise _invalid_trace_archival_config(
        path,
        f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.{key}' must be a boolean.",
    )


def _get_optional_bounded_positive_int(
    payload: dict[str, Any],
    path: Path,
    key: str,
    *,
    default: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if key not in payload:
        return default

    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_trace_archival_config(
            path,
            f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.{key}' must be a positive integer.",
        )
    if value < _TRACE_ARCHIVAL_POSITIVE_INT_MIN:
        raise _invalid_trace_archival_config(
            path,
            f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.{key}' must be a positive integer.",
        )
    if maximum is not None and value > maximum:
        raise _invalid_trace_archival_config(
            path,
            f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.{key}' must be <= {maximum}.",
        )
    return value


def _parse_long_retention_allowlist(value: Any, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _invalid_trace_archival_config(
            path,
            f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.long_retention_allowlist' must be a list.",
        )

    normalized_entries = []
    for idx, entry in enumerate(value):
        if isinstance(entry, (dict, list)):
            raise _invalid_trace_archival_config(
                path,
                f"'{_TRACE_ARCHIVAL_CONFIG_KEY}.long_retention_allowlist[{idx}]' "
                "must be a scalar experiment ID.",
            )
        normalized_entries.append(str(entry))

    try:
        return tuple(_parse_trace_archival_long_retention_allowlist(",".join(normalized_entries)))
    except MlflowException as e:
        raise _invalid_trace_archival_config(path, e.message) from e


def _invalid_trace_archival_config(path: Path, message: str) -> MlflowException:
    return MlflowException.invalid_parameter_value(
        f"Invalid trace archival config file '{path}': {message}"
    )

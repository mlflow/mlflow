import logging
import time
from datetime import datetime, timezone
from threading import Lock

import requests
from cachetools import LRUCache

from mlflow.server.auth.oauth.config import ExternalAuthzConfig

_logger = logging.getLogger(__name__)


class _TTLEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: dict[str, object], ttl: float):
        self.value = value
        self.expires_at = time.monotonic() + ttl


class ExternalAuthzClient:
    def __init__(self, config: ExternalAuthzConfig):
        self._config = config
        self._cache: LRUCache = LRUCache(maxsize=config.cache_max_size)
        self._cache_lock = Lock()

    def check_permission(
        self,
        username: str,
        email: str,
        provider: str,
        resource_type: str,
        resource_id: str,
        action: str,
        access_token: str = "",
        ip_address: str = "",
        workspace: str = "default",
    ) -> dict[str, object] | None:
        if not self._config.enabled:
            return None

        # Check cache
        cache_key = (username, resource_type, resource_id, action)
        with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if time.monotonic() < entry.expires_at:
                    return entry.value
                # Expired entry, remove it
                del self._cache[cache_key]

        # Build request payload
        payload = {
            "subject": {
                "username": username,
                "email": email,
                "provider": provider,
            },
            "resource": {
                "type": resource_type,
                "id": resource_id,
                "workspace": workspace,
            },
            "action": action,
            "context": {
                "ip_address": ip_address,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        headers = {"Content-Type": "application/json"}
        if self._config.forward_token and access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        headers["X-MLflow-Service"] = "mlflow"

        # Parse additional static headers
        if self._config.headers:
            for pair in self._config.headers.split(","):
                pair = pair.strip()
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    headers[k.strip()] = v.strip()

        # Make request with retries
        last_error = None
        for attempt in range(1 + self._config.max_retries):
            if attempt > 0:
                time.sleep(self._config.retry_backoff_seconds * attempt)

            try:
                resp = requests.post(
                    self._config.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._config.timeout_seconds,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    decision = {
                        "allowed": result.get(self._config.allowed_field, False),
                        "permission": result.get(self._config.permission_field, ""),
                        "is_admin": result.get(self._config.admin_field, False),
                        "reason": result.get("reason", ""),
                    }

                    # Cache with TTL from response or default
                    ttl = result.get("cache_ttl_seconds", self._config.cache_ttl_seconds)
                    if ttl > 0:
                        with self._cache_lock:
                            self._cache[cache_key] = _TTLEntry(decision, ttl)

                    return decision

                if resp.status_code == 404:
                    # Resource type not recognized, fall through to MLflow RBAC
                    return None

                if resp.status_code in (401, 403):
                    _logger.error(
                        "External authz service auth failure: %s %s",
                        resp.status_code,
                        resp.text,
                    )
                    return self._handle_error("auth_failure")

                if resp.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                    continue

                _logger.warning("Unexpected status from external authz: %s", resp.status_code)
                return self._handle_error("unexpected_status")

            except requests.exceptions.Timeout:
                last_error = "timeout"
                continue
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue

        _logger.error("External authz service failed after retries: %s", last_error)
        return self._handle_error("retries_exhausted")

    def _handle_error(self, reason: str) -> dict[str, object] | None:
        match self._config.on_error:
            case "deny":
                return {"allowed": False, "permission": "", "is_admin": False, "reason": reason}
            case "fallback_to_default":
                return None
            case "allow":
                return {"allowed": True, "permission": "", "is_admin": False, "reason": ""}
            case _:
                return {"allowed": False, "permission": "", "is_admin": False, "reason": reason}

    def invalidate_cache_for_user(self, username: str):
        with self._cache_lock:
            keys_to_remove = [k for k in self._cache if k[0] == username]
            for k in keys_to_remove:
                del self._cache[k]

    def invalidate_cache_for_resource(self, resource_type: str, resource_id: str):
        with self._cache_lock:
            keys_to_remove = [
                k for k in self._cache if k[1] == resource_type and k[2] == resource_id
            ]
            for k in keys_to_remove:
                del self._cache[k]

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

"""
Server-side encrypted cache for secrets management.

Implements time-bucketed ephemeral encryption for cached secrets to provide defense-in-depth
and satisfy CWE-316 (https://cwe.mitre.org/data/definitions/316.html).

Security Model and Limitations:

This cache protects against accidental exposure of secrets in logs, debug output, or error
messages. It also provides forward secrecy for expired cache entries since bucket keys are
randomly generated and deleted after expiration rather than derived from a base key.

This cache does not protect against attackers with real-time memory access to the running
process. During the TTL window (default 60s), both the encrypted secrets and their bucket
keys exist in process memory. A memory dump during this window captures both, allowing
decryption. Root-level attackers who can attach debuggers or read process memory can extract
secrets while they are cached.

The protection is that expired bucket keys are deleted from memory, making historical secrets
permanently unrecoverable even with full memory access. For protection against attackers with
real-time memory access, hardware-backed key management (HSM, Intel SGX, AWS Nitro Enclaves)
is required. Software-only solutions cannot prevent memory inspection by privileged attackers.

Implementation:

Random 256-bit keys are generated per time bucket using os.urandom (NIST SP 800-90A). Keys
are stored in memory and deleted on expiration. Secrets are encrypted with AES-GCM-256
(NIST SP 800-175B). After bucket expiration, keys are purged and old secrets become
permanently unrecoverable.

Performance overhead is approximately 10 microseconds per operation compared to 1-5ms for
database queries. Cache entries have a configurable TTL (default 60s, max 300s) and max
size (default 1000 entries).
"""

import json
import os
import time
from collections import OrderedDict
from threading import RLock, Thread
from typing import Any

from mlflow.utils.crypto import _encrypt_with_aes_gcm, decrypt_with_aes_gcm

_MIN_TTL = 10
_MAX_TTL = 300

_DEFAULT_CACHE_TTL = 60
_DEFAULT_CACHE_MAX_SIZE = 1000

SECRETS_CACHE_TTL_ENV_VAR = "MLFLOW_SERVER_SECRETS_CACHE_TTL"
SECRETS_CACHE_MAX_SIZE_ENV_VAR = "MLFLOW_SERVER_SECRETS_CACHE_MAX_SIZE"


class EphemeralCacheEncryption:
    """
    Time-bucketed ephemeral encryption with forward secrecy.

    Generates random 256-bit keys per time bucket (os.urandom per NIST SP 800-90A). Keys are stored
    in memory only and deleted when expired. Secrets encrypted with AES-GCM-256 + 96-bit nonce
    (NIST SP 800-38D). Expired bucket keys are purged from memory, making decryption of old cached
    secrets impossible even with full memory access (NIST SP 800-57 Section 8.2.3).

    Unlike key derivation schemes, this approach ensures true forward secrecy: once a bucket key
    is deleted, there is no computational path to recover it - the randomness is gone.

    A background daemon thread proactively purges expired keys, ensuring deterministic cleanup
    within TTL seconds of expiration regardless of cache activity.

    Args:
        ttl_seconds: Time-to-live and key rotation interval in seconds. Key rotation always
                    matches TTL to ensure cache entries expire when keys become unreadable.
    """

    def __init__(self, ttl_seconds: int = 60):
        self._key_rotation_seconds = ttl_seconds
        self._active_bucket: int | None = None
        self._active_key: bytes | None = None
        self._previous_bucket: int | None = None
        self._previous_key: bytes | None = None
        self._lock = RLock()
        self._shutdown = False

        # Start background cleanup thread
        self._cleanup_thread = Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="EphemeralCacheEncryption-cleanup",
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background thread that proactively purges expired bucket keys."""
        while not self._shutdown:
            time.sleep(self._key_rotation_seconds)
            self._purge_expired_keys()

    def _purge_expired_keys(self) -> None:
        """Purge any bucket keys that are more than 1 bucket old."""
        with self._lock:
            current_bucket = self._get_time_bucket()

            # Purge active key if it's now stale
            if self._active_bucket is not None:
                if abs(current_bucket - self._active_bucket) > 1:
                    self._active_bucket = None
                    self._active_key = None

            # Purge previous key if it's now more than 1 bucket old
            if self._previous_bucket is not None:
                if abs(current_bucket - self._previous_bucket) > 1:
                    self._previous_bucket = None
                    self._previous_key = None

    def _get_time_bucket(self) -> int:
        return int(time.time() // self._key_rotation_seconds)

    def _get_bucket_key(self, time_bucket: int) -> bytes | None:
        """
        Get or create bucket key, with lazy cleanup of expired keys.

        Keys are generated randomly per bucket (not derived), so once deleted they are
        permanently unrecoverable. This provides true forward secrecy against memory dumps.
        """
        with self._lock:
            current_bucket = self._get_time_bucket()

            # Rotate keys if we've moved to a new bucket
            if self._active_bucket is not None and self._active_bucket != current_bucket:
                # Keep previous bucket key for 1-bucket tolerance on decryption
                if self._active_bucket == current_bucket - 1:
                    self._previous_bucket = self._active_bucket
                    self._previous_key = self._active_key
                else:
                    # More than 1 bucket old - purge completely
                    self._previous_bucket = None
                    self._previous_key = None
                self._active_bucket = None
                self._active_key = None

            # Purge previous key if it's now more than 1 bucket old
            if self._previous_bucket is not None:
                if abs(current_bucket - self._previous_bucket) > 1:
                    self._previous_bucket = None
                    self._previous_key = None

            # Return existing key if available (for decryption of recent entries)
            if time_bucket == self._active_bucket and self._active_key is not None:
                return self._active_key
            if time_bucket == self._previous_bucket and self._previous_key is not None:
                return self._previous_key

            # Only create new keys for current bucket (not for expired buckets)
            if time_bucket == current_bucket:
                self._active_bucket = current_bucket
                self._active_key = os.urandom(32)
                return self._active_key

            # Bucket key was already purged - decryption impossible
            return None

    def encrypt(self, plaintext: str) -> tuple[bytes, int]:
        bucket = self._get_time_bucket()
        bucket_key = self._get_bucket_key(bucket)

        result = _encrypt_with_aes_gcm(
            plaintext.encode("utf-8"),
            bucket_key,
        )

        blob = result.nonce + result.ciphertext
        return (blob, bucket)

    def decrypt(self, blob: bytes, time_bucket: int) -> str | None:
        current_bucket = self._get_time_bucket()

        # NB: 1-bucket tolerance handles edge cases where encryption/decryption
        # happen across bucket boundary
        if abs(current_bucket - time_bucket) > 1:
            return None

        bucket_key = self._get_bucket_key(time_bucket)
        if bucket_key is None:
            return None

        try:
            plaintext_bytes = decrypt_with_aes_gcm(blob, bucket_key, aad=None)
            return plaintext_bytes.decode("utf-8")
        except Exception:
            return None


class SecretCache:
    """
    Thread-safe LRU cache for encrypted secrets satisfying CWE-316.

    Cache keys follow pattern "{resource_type}:{resource_id}". Entries expire via lazy TTL
    checks and LRU eviction. Full cache clear on mutations for simplicity (mutations rare vs reads).

    Args:
        ttl_seconds: Time-to-live in seconds (10-300s range). Default 60s.
        max_size: Max entries before LRU eviction. Default 1000.
    """

    def __init__(
        self,
        ttl_seconds: int = 60,
        max_size: int = 1000,
    ):
        if ttl_seconds < _MIN_TTL or ttl_seconds > _MAX_TTL:
            raise ValueError(
                f"Cache TTL must be between {_MIN_TTL} and {_MAX_TTL} seconds. "
                f"Got: {ttl_seconds}. "
                f"Lower values (10-30s) are more secure but impact performance. "
                f"Higher values (120-300s) improve performance but increase exposure window."
            )

        self._ttl = ttl_seconds
        self._max_size = max_size
        self._crypto = EphemeralCacheEncryption(ttl_seconds=ttl_seconds)
        self._cache: OrderedDict[str, tuple[bytes, int, float]] = OrderedDict()
        self._lock = RLock()

    def get(self, cache_key: str) -> str | dict[str, Any] | None:
        with self._lock:
            if cache_key not in self._cache:
                return None

            blob, time_bucket, expiry = self._cache[cache_key]

            if time.time() > expiry:
                del self._cache[cache_key]
                return None

            self._cache.move_to_end(cache_key)

            plaintext = self._crypto.decrypt(blob, time_bucket)
            if plaintext is None:
                del self._cache[cache_key]
                return None

            if plaintext.startswith("{") and plaintext.endswith("}"):
                try:
                    return json.loads(plaintext)
                except json.JSONDecodeError:
                    pass
            return plaintext

    def set(self, cache_key: str, value: str | dict[str, Any]) -> None:
        with self._lock:
            plaintext = json.dumps(value) if isinstance(value, dict) else value
            blob, time_bucket = self._crypto.encrypt(plaintext)
            expiry = time.time() + self._ttl

            self._cache[cache_key] = (blob, time_bucket, expiry)
            self._cache.move_to_end(cache_key)

            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

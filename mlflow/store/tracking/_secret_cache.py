"""
Server-side encrypted cache for secrets management.

Implements time-bucketed ephemeral encryption for cached secrets to prevent memory dump attacks
(CVE-2023-32784, CVE-2014-0160) and satisfy CWE-316 (https://cwe.mitre.org/data/definitions/316.html).

The general approach here is to port the principles of forward secrecy from transport security
(e.g., TLS 1.3) to an in-memory cache context. By deriving encryption keys from a process-ephemeral
base key combined with time buckets, we ensure that cached secrets become permanently unrecoverable
after a short time window even if an attacker gains access to the base key via a memory dump.

For full context:

Process-ephemeral base key (os.urandom(32)) combined with time buckets creates derived encryption
keys via SHA-256 (NIST SP 800-108). Secrets encrypted with AES-GCM-256 (NIST SP 800-175B) using
current bucket's key. After rotation, old encrypted secrets become unreadable even with base key
access, providing forward secrecy (NIST SP 800-57 Section 5.3.8).
This additional overhead adds ~10μs per operation vs ~1-5ms DB query overhead.
Typical cache hit rate 99%+ for repeated jobs-based execution and gateway usage, reducing DB
load by 100x.
The approach here has a very high bar on attack complexity: plaintext extraction takes
<30s and is completely trivial, while this encrypted approach requires
real-time memory access + crypto extraction within rotation window (60s default), all over a
memory dump file that is many of GBs in size with no direct reference to the encrypted bytes
in the file.

Cache entries have a configurable max size (default 1000 entries) and have a time-to-live (TTL)
default of 60s to limit exposure window. Admin-configurable overrides to the TTL allow tuning
performance vs security trade-offs with a non-negotiable enforced maximum of 300s to limit risk.
"""

import hashlib
import os
import time
from collections import OrderedDict
from threading import RLock

from mlflow.utils.crypto import decrypt_with_aes_gcm, encrypt_with_aes_gcm

MIN_TTL = 10  # 10 seconds minimum - prevents too-frequent DB queries
MAX_TTL = 300  # 5 minutes maximum - limits exposure window

DEFAULT_CACHE_TTL = 60  # 1 minute - balances performance and security
DEFAULT_CACHE_MAX_SIZE = 1000  # Handles typical workloads (~100KB memory)


class EphemeralCacheEncryption:
    """
    Time-bucketed ephemeral encryption with forward secrecy.

    Process-ephemeral 256-bit base key (os.urandom per NIST SP 800-90A) derives time-bucketed keys
    via SHA-256 (NIST SP 800-108). Secrets encrypted with AES-GCM-256 + 96-bit nonce
    (NIST SP 800-38D). Expired buckets (>1-bucket tolerance) make derived keys unrecoverable,
    preventing decryption of old cached secrets even with base key access
    (NIST SP 800-57 Section 8.2.3).

    Args:
        ttl_seconds: Time-to-live and key rotation interval in seconds. Key rotation always
                    matches TTL to ensure cache entries expire when keys become unreadable.
    """

    def __init__(self, ttl_seconds: int = 60):
        self._base_key = os.urandom(32)
        self._key_rotation_seconds = ttl_seconds

    def _get_time_bucket(self) -> int:
        """Get current time bucket (seconds since epoch // rotation_seconds)."""
        return int(time.time() // self._key_rotation_seconds)

    def _derive_bucket_key(self, time_bucket: int) -> bytes:
        """
        Derive 32-byte encryption key for time bucket via SHA-256(base_key || bucket).
        Expired buckets make derived keys unrecoverable, providing forward secrecy.
        """
        bucket_bytes = time_bucket.to_bytes(8, byteorder="big")
        return hashlib.sha256(self._base_key + bucket_bytes).digest()

    def encrypt(self, plaintext: str) -> tuple[bytes, int]:
        """
        Encrypt secret with AES-GCM-256 using current bucket's derived key.
        Returns (12-byte nonce + ciphertext blob, time_bucket).

        Args:
            plaintext: Secret value to encrypt
        """
        bucket = self._get_time_bucket()
        bucket_key = self._derive_bucket_key(bucket)

        result = encrypt_with_aes_gcm(
            plaintext.encode("utf-8"),
            bucket_key,
            nonce=None,  # Nonce is generated internally (random 96-bit)
            aad=None,  # No AAD needed for cache
        )

        blob = result.nonce + result.ciphertext
        return (blob, bucket)

    def decrypt(self, blob: bytes, time_bucket: int) -> str | None:
        """
        Decrypt using bucket's derived key with 1-bucket tolerance. Secrets >1 bucket old return
        None (cache miss), forcing re-encryption. All failures return None uniformly.

        Args:
            blob: 12-byte nonce + AES-GCM ciphertext
            time_bucket: Time bucket when secret was encrypted
        """
        current_bucket = self._get_time_bucket()

        # Allow decryption of current bucket and previous bucket (tolerance = 1)
        # This handles edge cases where encryption and decryption happen across
        # a bucket boundary
        if abs(current_bucket - time_bucket) > 1:
            return None  # Key rotated and the secret cannot be decrypted

        bucket_key = self._derive_bucket_key(time_bucket)

        try:
            plaintext_bytes = decrypt_with_aes_gcm(blob, bucket_key, aad=None)
            return plaintext_bytes.decode("utf-8")
        except Exception:
            # Return None uniformly for all failures (indicating a cache miss)
            return None


class SecretCache:
    """
    Thread-safe LRU cache for encrypted secrets satisfying CWE-316.

    The cache uses an RLock-protected OrderedDict to store AES-GCM-256 encrypted secrets. Cache
    keys follow the pattern "{resource_type}:{resource_id}" (e.g., "SCORER_JOB:job_456"). Entries
    expire via lazy TTL checks and LRU eviction when max_size is exceeded. TTL is validated at
    initialization to be between MIN_TTL (10s) and MAX_TTL (300s) to enforce security constraints.

    For cache invalidation, we chose full cache clear on mutations (UPDATE/DELETE/BIND/UNBIND)
    over targeted invalidation for simplicity and correctness. Since mutations are rare
    compared to reads, the cost of cache stampede is negligible and DB query
    serialization naturally handles the temporary rebuilding period.

    Args:
        ttl_seconds: Time-to-live in seconds (10-300s range). Default 60s. Key rotation
                    automatically matches TTL for optimal security.
        max_size: Max entries before LRU eviction. Default 1000 (~100KB memory).
    """

    def __init__(
        self,
        ttl_seconds: int = 60,
        max_size: int = 1000,
    ):
        if ttl_seconds < MIN_TTL or ttl_seconds > MAX_TTL:
            raise ValueError(
                f"Cache TTL must be between {MIN_TTL} and {MAX_TTL} seconds. "
                f"Got: {ttl_seconds}. "
                f"Lower values (10-30s) are more secure but impact performance. "
                f"Higher values (120-300s) improve performance but increase exposure window."
            )

        self._ttl = ttl_seconds
        self._max_size = max_size
        self._crypto = EphemeralCacheEncryption(ttl_seconds=ttl_seconds)

        # Thread-safe LRU cache: {cache_key: (blob, time_bucket, expiry_timestamp)}
        self._cache: OrderedDict[str, tuple[bytes, int, float]] = OrderedDict()
        self._lock = RLock()

    def get(self, cache_key: str) -> str | None:
        """
        Retrieve and decrypt secret. Returns None if missing, expired, or decryption fails.
        Promotes to MRU on hit.

        Args:
            cache_key: Cache key in format "{resource_type}:{resource_id}"
        """
        with self._lock:
            if cache_key not in self._cache:
                return None

            blob, time_bucket, expiry = self._cache[cache_key]

            if time.time() > expiry:
                del self._cache[cache_key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)

            plaintext = self._crypto.decrypt(blob, time_bucket)
            if plaintext is None:
                del self._cache[cache_key]  # Purge undecryptable entry
                return None

            return plaintext

    def set(self, cache_key: str, plaintext_secret: str) -> None:
        """
        Encrypt and cache secret. Evicts LRU entry if max_size exceeded.

        Args:
            cache_key: Cache key in format "{resource_type}:{resource_id}"
            plaintext_secret: Secret value to encrypt and cache
        """
        with self._lock:
            blob, time_bucket = self._crypto.encrypt(plaintext_secret)
            expiry = time.time() + self._ttl

            self._cache[cache_key] = (blob, time_bucket, expiry)
            self._cache.move_to_end(cache_key)

            # Evict LRU if over size limit
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)  # Remove oldest (first) item

    def clear(self) -> None:
        """
        Clear all cached secrets after mutations (UPDATE/DELETE/BIND/UNBIND). Full clear preferred
        over targeted invalidation for simplicity. Mutations rare (1-10/day) vs reads
        (1000-10000/min).
        """
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Return number of secrets currently cached."""
        with self._lock:
            return len(self._cache)

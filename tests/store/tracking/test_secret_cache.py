import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

# Commented out pending integration with rest branch:
# from mlflow.entities import SecretResourceType
from mlflow.store.tracking._secret_cache import (
    _MAX_TTL,
    _MIN_TTL,
    EphemeralCacheEncryption,
    SecretCache,
)

# Commented out pending integration with rest branch:
# from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def crypto():
    return EphemeralCacheEncryption(ttl_seconds=60)


@pytest.fixture
def cache():
    return SecretCache(ttl_seconds=60, max_size=100)


@pytest.mark.parametrize(
    "plaintext",
    [
        "my-secret-api-key-12345",
        json.dumps({"api_key": "key123", "region": "us-west-2"}),
        json.dumps({"openai_key": "sk-123", "anthropic_key": "claude-api"}),
    ],
)
def test_encrypt_decrypt_roundtrip(crypto, plaintext):
    blob, time_bucket = crypto.encrypt(plaintext)
    decrypted = crypto.decrypt(blob, time_bucket)
    assert decrypted == plaintext


def test_different_plaintexts_produce_different_ciphertexts(crypto):
    blob1, _ = crypto.encrypt("secret1")
    blob2, _ = crypto.encrypt("secret2")
    assert blob1 != blob2


def test_same_plaintext_produces_different_ciphertexts_due_to_nonce(crypto):
    blob1, bucket1 = crypto.encrypt("same-secret")
    blob2, bucket2 = crypto.encrypt("same-secret")
    assert blob1 != blob2
    assert bucket1 == bucket2


def test_decryption_fails_after_key_rotation():
    crypto = EphemeralCacheEncryption(ttl_seconds=1)
    plaintext = "secret"
    blob, time_bucket = crypto.encrypt(plaintext)
    time.sleep(2.5)
    decrypted = crypto.decrypt(blob, time_bucket)
    assert decrypted is None


def test_bucket_key_purged_after_expiration():
    crypto = EphemeralCacheEncryption(ttl_seconds=1)
    plaintext = "secret-that-should-be-unrecoverable"
    blob, time_bucket = crypto.encrypt(plaintext)

    assert crypto._active_bucket == time_bucket
    assert crypto._active_key is not None
    assert len(crypto._active_key) == 32

    time.sleep(2.5)
    crypto._get_bucket_key(crypto._get_time_bucket())

    # After 2.5 seconds with 1s TTL, the original bucket key should be purged
    # (it's more than 1 bucket old)
    assert crypto._active_bucket != time_bucket
    assert crypto._previous_bucket != time_bucket or crypto._previous_bucket is None
    assert crypto.decrypt(blob, time_bucket) is None


def test_decryption_succeeds_within_rotation_tolerance():
    crypto = EphemeralCacheEncryption(ttl_seconds=1)
    plaintext = "secret"
    blob, time_bucket = crypto.encrypt(plaintext)
    time.sleep(0.5)
    decrypted = crypto.decrypt(blob, time_bucket)
    assert decrypted == plaintext


def test_decryption_fails_with_corrupted_blob(crypto):
    plaintext = "secret"
    blob, time_bucket = crypto.encrypt(plaintext)
    corrupted_blob = b"corrupted" + blob[9:]
    decrypted = crypto.decrypt(corrupted_blob, time_bucket)
    assert decrypted is None


def test_process_ephemeral_keys_unique_per_instance():
    crypto1 = EphemeralCacheEncryption(ttl_seconds=60)
    crypto2 = EphemeralCacheEncryption(ttl_seconds=60)
    # Each instance generates independent random bucket keys
    # so one instance cannot decrypt what another encrypted
    blob, bucket = crypto1.encrypt("secret")
    decrypted_by_crypto2 = crypto2.decrypt(blob, bucket)
    assert decrypted_by_crypto2 is None


def test_cache_miss_returns_none(cache):
    result = cache.get("SCORER_JOB:job_123")
    assert result is None


def test_cache_hit_returns_value(cache):
    cache_key = "SCORER_JOB:job_123"
    secret = {"api_key": "secret123"}
    cache.set(cache_key, secret)
    result = cache.get(cache_key)
    assert result == secret


def test_cache_stores_multiple_entries(cache):
    cache.set("SCORER_JOB:job_1", {"key": "secret1"})
    cache.set("SCORER_JOB:job_2", {"key": "secret2"})
    cache.set("GLOBAL:workspace_1", {"key": "secret3"})
    assert cache.get("SCORER_JOB:job_1")["key"] == "secret1"
    assert cache.get("SCORER_JOB:job_2")["key"] == "secret2"
    assert cache.get("GLOBAL:workspace_1")["key"] == "secret3"


def test_lru_eviction_when_max_size_exceeded():
    cache = SecretCache(ttl_seconds=60, max_size=3)
    cache.set("key_1", {"value": "1"})
    cache.set("key_2", {"value": "2"})
    cache.set("key_3", {"value": "3"})
    assert cache.size() == 3
    cache.set("key_4", {"value": "4"})
    assert cache.size() == 3
    assert cache.get("key_1") is None
    assert cache.get("key_4") == {"value": "4"}


def test_lru_promotion_on_access():
    cache = SecretCache(ttl_seconds=60, max_size=3)
    cache.set("key_1", {"value": "1"})
    cache.set("key_2", {"value": "2"})
    cache.set("key_3", {"value": "3"})
    _ = cache.get("key_1")
    cache.set("key_4", {"value": "4"})
    assert cache.get("key_1") == {"value": "1"}
    assert cache.get("key_2") is None


def test_clear_removes_all_entries(cache):
    cache.set("key_1", {"value": "1"})
    cache.set("key_2", {"value": "2"})
    cache.set("key_3", {"value": "3"})
    assert cache.size() == 3
    cache.clear()
    assert cache.size() == 0
    assert cache.get("key_1") is None
    assert cache.get("key_2") is None
    assert cache.get("key_3") is None


@pytest.mark.parametrize(
    ("ttl", "should_raise"),
    [
        (_MIN_TTL - 1, True),
        (_MIN_TTL, False),
        (60, False),
        (_MAX_TTL, False),
        (_MAX_TTL + 1, True),
    ],
)
def test_ttl_validation(ttl, should_raise):
    if should_raise:
        match = f"Cache TTL must be between {_MIN_TTL} and {_MAX_TTL}"
        with pytest.raises(ValueError, match=match):
            SecretCache(ttl_seconds=ttl)
    else:
        cache = SecretCache(ttl_seconds=ttl)
        assert cache._ttl == ttl


def test_thread_safety_concurrent_reads(cache):
    cache.set("key_1", {"value": "secret"})

    def read_cache():
        for _ in range(100):
            result = cache.get("key_1")
            assert result == {"value": "secret"}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(read_cache) for _ in range(10)]
        for future in futures:
            future.result()


def test_thread_safety_concurrent_writes():
    cache = SecretCache(ttl_seconds=60, max_size=1000)

    def write_cache(thread_id):
        for i in range(50):
            cache.set(f"key_{thread_id}_{i}", {"value": f"{thread_id}_{i}"})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_cache, i) for i in range(10)]
        for future in futures:
            future.result()

    assert cache.size() <= 1000


def test_thread_safety_concurrent_clear():
    cache = SecretCache(ttl_seconds=60, max_size=1000)
    for i in range(100):
        cache.set(f"key_{i}", {"value": str(i)})

    def clear_cache():
        cache.clear()

    def read_cache():
        for _ in range(50):
            _ = cache.get("key_0")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(clear_cache) for _ in range(3)]
        futures += [executor.submit(read_cache) for _ in range(7)]
        for future in futures:
            future.result()


def test_cache_handles_unicode_secrets(cache):
    unicode_secret = {"key": "🔐🔑 secret with émojis and àccénts"}
    cache.set("unicode_key", unicode_secret)
    result = cache.get("unicode_key")
    assert result == unicode_secret
    assert result["key"] == "🔐🔑 secret with émojis and àccénts"


def test_cache_handles_large_secrets(cache):
    large_secret = {"key": "x" * 10000}
    cache.set("large_key", large_secret)
    result = cache.get("large_key")
    assert result == large_secret
    assert len(result["key"]) == 10000


def test_cache_roundtrip_with_complex_secret(cache):
    complex_secret = {
        "openai_key": "sk-1234567890",
        "anthropic_key": "claude-api-key",
        "config": {"region": "us-west-2", "timeout": 30},
    }
    cache.set("SCORER_JOB:complex_job", complex_secret)
    result = cache.get("SCORER_JOB:complex_job")
    assert result == complex_secret


def test_cache_isolation_between_resources(cache):
    cache.set("SCORER_JOB:job_1", {"key": "secret1"})
    cache.set("SCORER_JOB:job_2", {"key": "secret2"})
    cache.set("GLOBAL:workspace_1", {"key": "secret3"})
    cache.clear()
    assert cache.get("SCORER_JOB:job_1") is None


# Integration tests commented out pending rest branch integration:
"""
    assert cache.get("SCORER_JOB:job_2") is None
    assert cache.get("GLOBAL:workspace_1") is None


def test_sqlalchemy_store_uses_default_constants(tmp_path):
    db_uri = f"sqlite:///{tmp_path}/test.db"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root=str(tmp_path))
    assert store._secret_cache._ttl == DEFAULT_CACHE_TTL
    assert store._secret_cache._max_size == DEFAULT_CACHE_MAX_SIZE
    assert store._secret_cache._crypto._key_rotation_seconds == DEFAULT_CACHE_TTL


def test_sqlalchemy_store_respects_env_var_config(tmp_path, monkeypatch):
    monkeypatch.setenv(SECRETS_CACHE_TTL_ENV_VAR, "120")
    monkeypatch.setenv(SECRETS_CACHE_MAX_SIZE_ENV_VAR, "500")
    db_uri = f"sqlite:///{tmp_path}/test_env.db"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root=str(tmp_path))
    assert store._secret_cache._ttl == 120
    assert store._secret_cache._max_size == 500
    assert store._secret_cache._crypto._key_rotation_seconds == 120


def test_e2e_secret_cache_populated_on_first_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRETS_KEK_PASSPHRASE", "test-kek-passphrase-32chars-min")
    db_uri = f"sqlite:///{tmp_path}/test_e2e.db"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root=str(tmp_path))

    assert store._secret_cache.size() == 0

    store._create_and_bind_secret(
        secret_name="api_key",
        secret_value={"api_key": "sk-test-12345"},
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_123",
        field_name="OPENAI_API_KEY",
        is_shared=False,
        created_by="test@example.com",
    )

    assert store._secret_cache.size() == 0

    secrets = store._get_secrets_for_resource(SecretResourceType.SCORER_JOB, "job_123")
    assert secrets == {"OPENAI_API_KEY": "sk-test-12345"}
    assert store._secret_cache.size() == 1

    cache_key = f"{SecretResourceType.SCORER_JOB}:job_123"
    cached_value = store._secret_cache.get(cache_key)
    assert cached_value is not None
    assert json.loads(cached_value) == {"OPENAI_API_KEY": "sk-test-12345"}

    secrets_again = store._get_secrets_for_resource(SecretResourceType.SCORER_JOB, "job_123")
    assert secrets_again == {"OPENAI_API_KEY": "sk-test-12345"}
    assert store._secret_cache.size() == 1

    store._secret_cache.clear()
    assert store._secret_cache.size() == 0

    secrets_after_clear = store._get_secrets_for_resource(SecretResourceType.SCORER_JOB, "job_123")
    assert secrets_after_clear == {"OPENAI_API_KEY": "sk-test-12345"}
    assert store._secret_cache.size() == 1

    cached_value_after_clear = store._secret_cache.get(cache_key)
    assert cached_value_after_clear is not None
    assert json.loads(cached_value_after_clear) == {"OPENAI_API_KEY": "sk-test-12345"}


def test_e2e_cache_miss_on_key_rotation_falls_back_to_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRETS_KEK_PASSPHRASE", "test-kek-passphrase-32chars-min")
    db_uri = f"sqlite:///{tmp_path}/test_cache_miss.db"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root=str(tmp_path))

    store._create_and_bind_secret(
        secret_name="api_key",
        secret_value={"api_key": "sk-test-12345"},
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_123",
        field_name="OPENAI_API_KEY",
        is_shared=False,
        created_by="test@example.com",
    )

    secrets = store._get_secrets_for_resource(SecretResourceType.SCORER_JOB, "job_123")
    assert secrets == {"OPENAI_API_KEY": "sk-test-12345"}
    assert store._secret_cache.size() == 1

    cache_key = f"{SecretResourceType.SCORER_JOB}:job_123"
    cached_value = store._secret_cache.get(cache_key)
    assert cached_value is not None

    old_base_key = store._secret_cache._crypto._base_key

    store._secret_cache._crypto._base_key = os.urandom(32)

    new_base_key = store._secret_cache._crypto._base_key
    assert old_base_key != new_base_key

    cached_value_after_rotation = store._secret_cache.get(cache_key)
    assert cached_value_after_rotation is None

    secrets_after_rotation = store._get_secrets_for_resource(
        SecretResourceType.SCORER_JOB, "job_123"
    )
    assert secrets_after_rotation == {"OPENAI_API_KEY": "sk-test-12345"}
    assert store._secret_cache.size() == 1

    cached_value_repopulated = store._secret_cache.get(cache_key)
    assert cached_value_repopulated is not None
    assert json.loads(cached_value_repopulated) == {"OPENAI_API_KEY": "sk-test-12345"}


def _worker_process_fetch_secret(db_uri, worker_id, job_ids, result_queue):
    os.environ["MLFLOW_SECRETS_KEK_PASSPHRASE"] = "test-kek-passphrase-32chars-min"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root="/tmp")

    initial_cache_size = store._secret_cache.size()

    fetched_secrets = {}
    for job_id in job_ids:
        secrets = store._get_secrets_for_resource(SecretResourceType.SCORER_JOB, job_id)
        fetched_secrets[job_id] = secrets

    final_cache_size = store._secret_cache.size()

    cached_keys = {}
    for job_id in job_ids:
        cache_key = f"{SecretResourceType.SCORER_JOB}:{job_id}"
        cached_keys[job_id] = store._secret_cache.get(cache_key)

    base_key = store._secret_cache._crypto._base_key

    result_queue.put(
        {
            "worker_id": worker_id,
            "initial_cache_size": initial_cache_size,
            "final_cache_size": final_cache_size,
            "fetched_secrets": fetched_secrets,
            "cached_keys": cached_keys,
            "base_key": base_key,
        }
    )


def test_e2e_process_isolation_separate_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_SECRETS_KEK_PASSPHRASE", "test-kek-passphrase-32chars-min")
    db_uri = f"sqlite:///{tmp_path}/test_multiprocess.db"
    store = SqlAlchemyStore(db_uri=db_uri, default_artifact_root=str(tmp_path))

    store._create_and_bind_secret(
        secret_name="api_key_123",
        secret_value={"api_key": "sk-worker-0-secret"},
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_123",
        field_name="OPENAI_API_KEY",
        is_shared=False,
        created_by="test@example.com",
    )

    store._create_and_bind_secret(
        secret_name="api_key_456",
        secret_value={"api_key": "sk-worker-1-secret"},
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_456",
        field_name="ANTHROPIC_API_KEY",
        is_shared=False,
        created_by="test@example.com",
    )

    store._create_and_bind_secret(
        secret_name="shared_key",
        secret_value={"api_key": "sk-shared-secret"},
        resource_type=SecretResourceType.SCORER_JOB,
        resource_id="job_789",
        field_name="SHARED_KEY",
        is_shared=False,
        created_by="test@example.com",
    )

    result_queue = multiprocessing.Queue()

    worker_jobs = [
        ["job_123", "job_789"],
        ["job_456", "job_789"],
    ]

    processes = []
    for i in range(2):
        p = multiprocessing.Process(
            target=_worker_process_fetch_secret, args=(db_uri, i, worker_jobs[i], result_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    assert len(results) == 2

    worker_0 = next(r for r in results if r["worker_id"] == 0)
    worker_1 = next(r for r in results if r["worker_id"] == 1)

    assert worker_0["initial_cache_size"] == 0
    assert worker_0["final_cache_size"] == 2
    assert worker_0["fetched_secrets"]["job_123"] == {"OPENAI_API_KEY": "sk-worker-0-secret"}
    assert worker_0["fetched_secrets"]["job_789"] == {"SHARED_KEY": "sk-shared-secret"}
    assert worker_0["cached_keys"]["job_123"] is not None
    assert worker_0["cached_keys"]["job_789"] is not None

    assert worker_1["initial_cache_size"] == 0
    assert worker_1["final_cache_size"] == 2
    assert worker_1["fetched_secrets"]["job_456"] == {"ANTHROPIC_API_KEY": "sk-worker-1-secret"}
    assert worker_1["fetched_secrets"]["job_789"] == {"SHARED_KEY": "sk-shared-secret"}
    assert worker_1["cached_keys"]["job_456"] is not None
    assert worker_1["cached_keys"]["job_789"] is not None

    assert worker_0["base_key"] != worker_1["base_key"]
"""

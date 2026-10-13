import threading
import time

import pytest

from mlflow.prompt.registry_utils import (
    PromptCache,
    PromptCacheKey,
    _get_prompt_cache_namespace,
)
from mlflow.utils.workspace_context import WorkspaceContext

REGISTRY_URI = "https://registry.example.com"
OTHER_REGISTRY_URI = "https://other-registry.example.com"


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the prompt cache before and after each test."""
    PromptCache._reset_instance()
    yield
    PromptCache._reset_instance()


def test_singleton_pattern():
    cache1 = PromptCache.get_instance()
    cache2 = PromptCache.get_instance()
    assert cache1 is cache2


def test_set_and_get():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("test-prompt", version=1, registry_uri=REGISTRY_URI)
    cache.set(key, {"template": "Hello {{name}}"})
    assert cache.get(key) == {"template": "Hello {{name}}"}


def test_get_nonexistent():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("nonexistent", version=1, registry_uri=REGISTRY_URI)
    assert cache.get(key) is None


def test_ttl_expiration():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("test-prompt", version=1, registry_uri=REGISTRY_URI)
    cache.set(key, "value", ttl_seconds=0.01)
    time.sleep(0.02)
    assert cache.get(key) is None


def test_delete_prompt():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
    key2 = PromptCacheKey.from_parts("my-prompt", version=2, registry_uri=REGISTRY_URI)
    key3 = PromptCacheKey.from_parts("other-prompt", version=1, registry_uri=REGISTRY_URI)

    cache.set(key1, "value1")
    cache.set(key2, "value2")
    cache.set(key3, "value3")

    # Delete only version 1 of my-prompt
    cache.delete("my-prompt", version=1, registry_uri=REGISTRY_URI)

    assert cache.get(key1) is None
    assert cache.get(key2) == "value2"  # version 2 still cached
    assert cache.get(key3) == "value3"


def test_delete_prompt_by_alias():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("my-prompt", alias="production", registry_uri=REGISTRY_URI)
    key2 = PromptCacheKey.from_parts("my-prompt", alias="staging", registry_uri=REGISTRY_URI)

    cache.set(key1, "value1")
    cache.set(key2, "value2")

    # Delete only the production alias
    cache.delete("my-prompt", alias="production", registry_uri=REGISTRY_URI)

    assert cache.get(key1) is None
    assert cache.get(key2) == "value2"  # staging still cached


def test_delete_all_prompt_entries():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
    key2 = PromptCacheKey.from_parts("my-prompt", version=2, registry_uri=REGISTRY_URI)
    key3 = PromptCacheKey.from_parts("my-prompt", alias="latest", registry_uri=REGISTRY_URI)
    key4 = PromptCacheKey.from_parts("other-prompt", version=1, registry_uri=REGISTRY_URI)

    cache.set(key1, "value1")
    cache.set(key2, "value2")
    cache.set(key3, "value3")
    cache.set(key4, "value4")

    cache.delete_all("my-prompt", registry_uri=REGISTRY_URI)

    assert cache.get(key1) is None
    assert cache.get(key2) is None
    assert cache.get(key3) is None
    assert cache.get(key4) == "value4"


def test_clear():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("prompt1", version=1, registry_uri=REGISTRY_URI)
    key2 = PromptCacheKey.from_parts("prompt2", version=1, registry_uri=REGISTRY_URI)

    cache.set(key1, "value1")
    cache.set(key2, "value2")
    cache.clear()

    assert cache.get(key1) is None
    assert cache.get(key2) is None


def test_generate_cache_key_with_version():
    key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
    assert key.name == "my-prompt"
    assert key.version == 1
    assert key.alias is None


def test_generate_cache_key_with_alias():
    key = PromptCacheKey.from_parts("my-prompt", alias="production", registry_uri=REGISTRY_URI)
    assert key.name == "my-prompt"
    assert key.version is None
    assert key.alias == "production"


def test_generate_cache_key_with_neither():
    key = PromptCacheKey.from_parts("my-prompt", registry_uri=REGISTRY_URI)
    assert key.name == "my-prompt"
    assert key.version is None
    assert key.alias is None


def test_generate_cache_key_with_both_raises_error():
    with pytest.raises(ValueError, match="Cannot specify both version and alias"):
        PromptCacheKey.from_parts(
            "my-prompt", version=1, alias="production", registry_uri=REGISTRY_URI
        )


def test_generate_cache_key_version_zero():
    key = PromptCacheKey.from_parts("my-prompt", version=0, registry_uri=REGISTRY_URI)
    assert key.name == "my-prompt"
    assert key.version == 0
    assert key.alias is None


@pytest.mark.parametrize(
    "registry_uri",
    [
        "file:///tmp/mlruns",
        "sqlite:////tmp/mlflow.db",
        "databricks-uc://profile",
        "https://user:secret@registry.example.com",
    ],
)
def test_prompt_cache_namespace_is_deterministic_and_non_secret(registry_uri):
    namespace = _get_prompt_cache_namespace(registry_uri)
    key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=registry_uri)

    assert namespace == _get_prompt_cache_namespace(registry_uri)
    assert len(namespace) == 64
    assert all(char in "0123456789abcdef" for char in namespace)
    assert registry_uri not in repr(key)
    assert "user" not in repr(key)
    assert "secret" not in repr(key)


def test_prompt_cache_namespace_rejects_unresolved_registry_uri():
    with pytest.raises(ValueError, match="Registry URI must be resolved"):
        _get_prompt_cache_namespace(None)


def test_cache_is_isolated_by_registry_uri():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
    other_key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=OTHER_REGISTRY_URI)

    cache.set(key, "value")
    cache.set(other_key, "other-value")

    assert key != other_key
    assert cache.get(key) == "value"
    assert cache.get(other_key) == "other-value"

    cache.delete("my-prompt", version=1, registry_uri=REGISTRY_URI)
    assert cache.get(key) is None
    assert cache.get(other_key) == "other-value"


def test_cache_is_isolated_by_workspace():
    cache = PromptCache.get_instance()
    with WorkspaceContext("team-a"):
        key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
        cache.set(key, "team-a-value")

    with WorkspaceContext("team-b"):
        other_key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
        cache.set(other_key, "team-b-value")
        cache.delete("my-prompt", version=1, registry_uri=REGISTRY_URI)

    assert key != other_key
    assert cache.get(key) == "team-a-value"
    assert cache.get(other_key) is None


def test_delete_all_is_scoped_by_registry_uri():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
    alias_key = PromptCacheKey.from_parts(
        "my-prompt", alias="production", registry_uri=REGISTRY_URI
    )
    other_key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=OTHER_REGISTRY_URI)

    cache.set(key, "value")
    cache.set(alias_key, "alias-value")
    cache.set(other_key, "other-value")

    cache.delete_all("my-prompt", registry_uri=REGISTRY_URI)

    assert cache.get(key) is None
    assert cache.get(alias_key) is None
    assert cache.get(other_key) == "other-value"


def test_delete_all_is_scoped_by_workspace():
    cache = PromptCache.get_instance()
    with WorkspaceContext("team-a"):
        key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
        alias_key = PromptCacheKey.from_parts(
            "my-prompt", alias="production", registry_uri=REGISTRY_URI
        )
        cache.set(key, "team-a-value")
        cache.set(alias_key, "team-a-alias-value")

    with WorkspaceContext("team-b"):
        other_key = PromptCacheKey.from_parts("my-prompt", version=1, registry_uri=REGISTRY_URI)
        cache.set(other_key, "team-b-value")
        cache.delete_all("my-prompt", registry_uri=REGISTRY_URI)

    assert cache.get(key) == "team-a-value"
    assert cache.get(alias_key) == "team-a-alias-value"
    assert cache.get(other_key) is None


def test_concurrent_get_instance():
    instances = []
    errors = []

    def get_instance():
        try:
            instance = PromptCache.get_instance()
            instances.append(instance)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(name=f"prompt-cache-singleton-{i}", target=get_instance) for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert all(inst is instances[0] for inst in instances)


def test_concurrent_operations():
    cache = PromptCache.get_instance()
    errors = []

    def writer(thread_id):
        try:
            for i in range(50):
                key = PromptCacheKey.from_parts(
                    f"prompt-{thread_id}-{i}", version=1, registry_uri=REGISTRY_URI
                )
                cache.set(key, f"value-{thread_id}-{i}")
        except Exception as e:
            errors.append(e)

    def reader(thread_id):
        try:
            for i in range(50):
                key = PromptCacheKey.from_parts(
                    f"prompt-{thread_id}-{i}", version=1, registry_uri=REGISTRY_URI
                )
                cache.get(key)
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(name=f"prompt-cache-writer-{i}", target=writer, args=(i,)))
        threads.append(threading.Thread(name=f"prompt-cache-reader-{i}", target=reader, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_set_uses_default_ttl():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("test", version=1, registry_uri=REGISTRY_URI)
    cache.set(key, "value")
    assert cache.get(key) == "value"

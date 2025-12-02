import threading
import time

import pytest

from mlflow.prompt.registry_utils import PromptCache, PromptCacheKey


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
    key = PromptCacheKey.from_parts("test-prompt", version=1)
    cache.set(key, {"template": "Hello {{name}}"})
    assert cache.get(key) == {"template": "Hello {{name}}"}


def test_get_nonexistent():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("nonexistent", version=1)
    assert cache.get(key) is None


def test_ttl_expiration():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("test-prompt", version=1)
    cache.set(key, "value", ttl_seconds=0.01)
    time.sleep(0.02)
    assert cache.get(key) is None


def test_delete_prompt():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("my-prompt", version=1)
    key2 = PromptCacheKey.from_parts("my-prompt", version=2)
    key3 = PromptCacheKey.from_parts("other-prompt", version=1)

    cache.set(key1, "value1")
    cache.set(key2, "value2")
    cache.set(key3, "value3")

    # Delete only version 1 of my-prompt
    cache.delete("my-prompt", version=1)

    assert cache.get(key1) is None
    assert cache.get(key2) == "value2"  # version 2 still cached
    assert cache.get(key3) == "value3"


def test_delete_prompt_by_alias():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("my-prompt", alias="production")
    key2 = PromptCacheKey.from_parts("my-prompt", alias="staging")

    cache.set(key1, "value1")
    cache.set(key2, "value2")

    # Delete only the production alias
    cache.delete("my-prompt", alias="production")

    assert cache.get(key1) is None
    assert cache.get(key2) == "value2"  # staging still cached


def test_clear():
    cache = PromptCache.get_instance()
    key1 = PromptCacheKey.from_parts("prompt1", version=1)
    key2 = PromptCacheKey.from_parts("prompt2", version=1)

    cache.set(key1, "value1")
    cache.set(key2, "value2")
    cache.clear()

    assert cache.get(key1) is None
    assert cache.get(key2) is None


def test_generate_cache_key_with_version():
    key = PromptCacheKey.from_parts("my-prompt", version=1)
    assert key.name == "my-prompt"
    assert key.version == 1
    assert key.alias is None


def test_generate_cache_key_with_alias():
    key = PromptCacheKey.from_parts("my-prompt", alias="production")
    assert key.name == "my-prompt"
    assert key.version is None
    assert key.alias == "production"


def test_generate_cache_key_with_neither():
    key = PromptCacheKey.from_parts("my-prompt")
    assert key.name == "my-prompt"
    assert key.version is None
    assert key.alias is None


def test_generate_cache_key_with_both_raises_error():
    with pytest.raises(ValueError, match="Cannot specify both version and alias"):
        PromptCacheKey.from_parts("my-prompt", version=1, alias="production")


def test_generate_cache_key_version_zero():
    key = PromptCacheKey.from_parts("my-prompt", version=0)
    assert key.name == "my-prompt"
    assert key.version == 0
    assert key.alias is None


def test_concurrent_get_instance():
    instances = []
    errors = []

    def get_instance():
        try:
            instance = PromptCache.get_instance()
            instances.append(instance)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=get_instance) for _ in range(10)]
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
                key = PromptCacheKey.from_parts(f"prompt-{thread_id}-{i}", version=1)
                cache.set(key, f"value-{thread_id}-{i}")
        except Exception as e:
            errors.append(e)

    def reader(thread_id):
        try:
            for i in range(50):
                key = PromptCacheKey.from_parts(f"prompt-{thread_id}-{i}", version=1)
                cache.get(key)
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_set_uses_default_ttl():
    cache = PromptCache.get_instance()
    key = PromptCacheKey.from_parts("test", version=1)
    cache.set(key, "value")
    assert cache.get(key) == "value"

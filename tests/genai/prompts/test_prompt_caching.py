import time
from unittest import mock

import pytest

import mlflow
from mlflow.prompt.registry_utils import PromptCache


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the prompt cache before and after each test."""
    PromptCache._reset_instance()
    yield
    PromptCache._reset_instance()


def test_prompt_cache_hit():
    """Test that loading the same prompt twice uses the cache."""
    mlflow.genai.register_prompt(name="cached_prompt", template="Hello {{name}}!")

    # First load - cache miss (fetch from server)
    prompt1 = mlflow.genai.load_prompt("cached_prompt", version=1)

    # Second load - cache hit (should not call registry client)
    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version",
    ) as mock_load:
        prompt2 = mlflow.genai.load_prompt("cached_prompt", version=1)
        assert mock_load.call_count == 0

    assert prompt1.template == prompt2.template
    assert prompt1.name == prompt2.name


def test_prompt_cache_ttl_expiration():
    """Test that cached prompts expire after TTL."""
    mlflow.genai.register_prompt(name="expiring_prompt", template="Hello {{name}}!")

    # Load with very short TTL
    mlflow.genai.load_prompt("expiring_prompt", version=1, cache_ttl_seconds=1)

    # Immediate second load should hit cache
    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version",
    ) as mock_load:
        mlflow.genai.load_prompt("expiring_prompt", version=1, cache_ttl_seconds=1)
        assert mock_load.call_count == 0

    # Wait for TTL to expire
    time.sleep(1.1)

    # Load after expiration should miss cache - need to actually fetch
    prompt = mlflow.genai.load_prompt("expiring_prompt", version=1, cache_ttl_seconds=1)
    assert prompt is not None
    assert prompt.template == "Hello {{name}}!"


def test_prompt_cache_bypass_with_zero_ttl():
    """Test that setting `cache_ttl_seconds=0` bypasses the cache."""
    mlflow.genai.register_prompt(name="bypass_prompt", template="Hello {{name}}!")

    # First load to populate cache
    mlflow.genai.load_prompt("bypass_prompt", version=1)

    # Load with TTL=0 should bypass cache even though it's cached
    # We verify by checking that the registry is called
    call_count = 0
    original_get = mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version

    def counting_get(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_get(self, *args, **kwargs)

    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version",
        counting_get,
    ):
        mlflow.genai.load_prompt("bypass_prompt", version=1, cache_ttl_seconds=0)
        mlflow.genai.load_prompt("bypass_prompt", version=1, cache_ttl_seconds=0)
        mlflow.genai.load_prompt("bypass_prompt", version=1, cache_ttl_seconds=0)
        assert call_count == 3


def test_prompt_cache_alias_cached():
    """Test that prompts loaded by alias are cached."""
    mlflow.genai.register_prompt(name="alias_prompt", template="Version 1")
    mlflow.genai.set_prompt_alias("alias_prompt", alias="production", version=1)

    # First load by alias - cache miss
    mlflow.genai.load_prompt("prompts:/alias_prompt@production")

    # Second load by alias - cache hit
    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version_by_alias",
    ) as mock_load:
        mlflow.genai.load_prompt("prompts:/alias_prompt@production")
        assert mock_load.call_count == 0


def test_prompt_cache_different_versions():
    """Test that different versions are cached separately."""
    mlflow.genai.register_prompt(name="multi_version", template="Version 1")
    mlflow.genai.register_prompt(name="multi_version", template="Version 2")

    # Load both versions
    prompt_v1 = mlflow.genai.load_prompt("multi_version", version=1)
    prompt_v2 = mlflow.genai.load_prompt("multi_version", version=2)

    assert prompt_v1.template == "Version 1"
    assert prompt_v2.template == "Version 2"

    # Both should be cached now
    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version",
    ) as mock_load:
        mlflow.genai.load_prompt("multi_version", version=1)
        mlflow.genai.load_prompt("multi_version", version=2)
        assert mock_load.call_count == 0


def test_prompt_cache_custom_ttl():
    mlflow.genai.register_prompt(name="custom_ttl_prompt", template="Hello!")

    # Load with custom TTL
    mlflow.genai.load_prompt("custom_ttl_prompt", version=1, cache_ttl_seconds=300)

    # Should be cached
    cache = PromptCache.get_instance()
    key = PromptCache.generate_cache_key("custom_ttl_prompt", version=1)
    cached = cache.get(key)
    assert cached is not None
    assert cached.template == "Hello!"


def test_prompt_cache_invalidation():
    """Test that cache invalidation works correctly."""
    mlflow.genai.register_prompt(name="invalidate_prompt", template="Hello!")

    # Load and cache
    mlflow.genai.load_prompt("invalidate_prompt", version=1)

    # Verify it's cached
    cache = PromptCache.get_instance()
    key = PromptCache.generate_cache_key("invalidate_prompt", version=1)
    assert cache.get(key) is not None

    # Delete specific version from cache
    cache.delete("invalidate_prompt", version=1)

    # Should be gone
    assert cache.get(key) is None

    # Next load should fetch from server
    prompt = mlflow.genai.load_prompt("invalidate_prompt", version=1)
    assert prompt is not None
    assert prompt.template == "Hello!"


def test_prompt_cache_uri_format():
    """Test caching works with URI format."""
    mlflow.genai.register_prompt(name="uri_prompt", template="Hello!")

    # Load using URI format
    prompt1 = mlflow.genai.load_prompt("prompts:/uri_prompt/1")

    # Should be cached
    with mock.patch(
        "mlflow.tracking._model_registry.client.ModelRegistryClient.get_prompt_version",
    ) as mock_load:
        prompt2 = mlflow.genai.load_prompt("prompts:/uri_prompt/1")
        assert mock_load.call_count == 0

    assert prompt1.template == prompt2.template


def test_prompt_cache_clear():
    """Test clearing the entire cache."""
    mlflow.genai.register_prompt(name="clear_test_1", template="Hello 1!")
    mlflow.genai.register_prompt(name="clear_test_2", template="Hello 2!")

    # Load both
    mlflow.genai.load_prompt("clear_test_1", version=1)
    mlflow.genai.load_prompt("clear_test_2", version=1)

    # Clear cache
    cache = PromptCache.get_instance()
    cache.clear()

    # Both should require fetching from server
    prompt1 = mlflow.genai.load_prompt("clear_test_1", version=1)
    prompt2 = mlflow.genai.load_prompt("clear_test_2", version=1)
    assert prompt1.template == "Hello 1!"
    assert prompt2.template == "Hello 2!"


def test_prompt_cache_env_variable(monkeypatch):
    """Test that MLFLOW_PROMPT_CACHE_TTL_SECONDS environment variable is respected."""
    mlflow.genai.register_prompt(name="env_var_prompt", template="Hello!")

    # Set environment variable to 1 second
    monkeypatch.setenv("MLFLOW_PROMPT_CACHE_TTL_SECONDS", "1")

    # Need to reload the environment variable

    # Load prompt (uses env var TTL)
    mlflow.genai.load_prompt("env_var_prompt", version=1)

    # Wait for expiration
    time.sleep(1.1)

    # Should fetch from server again
    prompt = mlflow.genai.load_prompt("env_var_prompt", version=1)
    assert prompt is not None
    assert prompt.template == "Hello!"

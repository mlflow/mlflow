from concurrent.futures import ThreadPoolExecutor
import pytest
import time
from unittest import mock
from mlflow.tracing.utils.cache import TTLCache


@pytest.fixture
def ttl_cache(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_TTL_CHECK_INTERVAL_SECONDS", "2")

    cache = TTLCache(maxsize=2, ttl=1)  # TTL of 2 seconds for testing
    yield cache
    cache._shutdown()


def test_set_and_get_item(ttl_cache):
    ttl_cache["a"] = 1
    assert ttl_cache["a"] == 1
    assert ttl_cache.get("a") == 1

    with pytest.raises(KeyError):
        ttl_cache["b"]
    assert ttl_cache.get("b") == None
    assert ttl_cache.get("b", "default") == "default"


def test_item_expiration(ttl_cache):
    ttl_cache["a"] = 1
    time.sleep(3)
    with pytest.raises(KeyError):
        ttl_cache["a"]

    ttl_cache["b"] = 2
    ttl_cache["c"] = 3
    ttl_cache["d"] = 4
    with pytest.raises(KeyError):
        ttl_cache["b"]


def test_expiration_loop_start_after_first_set(ttl_cache):
    assert not hasattr(ttl_cache, "_expire_checker_thread")

    ttl_cache["a"] = 1
    assert hasattr(ttl_cache, "_expire_checker_thread")

def test_expiration_loop_start_thread_safe(ttl_cache):
    assert not hasattr(ttl_cache, "_expire_checker_thread")

    # Set item in parallel
    with mock.patch("atexit.register") as mock_register:
        with ThreadPoolExecutor(max_workers=10) as executor:
            def set_item(key):
                ttl_cache[str(key)] = key
            executor.map(set_item, range(20))

    # Check that the expire loop is started only once
    assert mock_register.call_count == 1

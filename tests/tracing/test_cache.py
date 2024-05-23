import time

from mlflow.tracing.cache import SizedTTLCache


def test_sized_ttl_cache_limit_size():
    cache = SizedTTLCache(maxsize_bytes=150, ttl=60, serializer=lambda x: x)

    assert cache.getsizeof("a") == 50
    assert cache.getsizeof(["a", "b", "c"]) == 150

    cache["key1"] = "ab"  # This is 51 bytes in Python3
    cache["key2"] = "c"  # This is 50 bytes
    assert len(cache) == 2

    cache["key3"] = "c"  # Breach the limit and push out the oldest key
    assert len(cache) == 2
    assert "key1" not in cache

    cache["key4"] = "d"  # Just fit
    assert len(cache) == 3
    assert "key2" in cache
    assert "key3" in cache
    assert "key4" in cache

    cache["key5"] = ["e", "f"]  # Breach the limit again (100 bytes) and push out two keys
    assert len(cache) == 2
    assert "key2" not in cache
    assert "key3" not in cache
    assert "key4" in cache
    assert "key5" in cache


def test_sized_ttl_cache_drop_too_large_value():
    cache = SizedTTLCache(maxsize_bytes=100, ttl=60, serializer=str)
    cache["key1"] = "a" * 500
    assert len(cache) == 0  # Just drop with warning


def test_sized_ttl_cache_handle_value_update():
    cache = SizedTTLCache(maxsize_bytes=200, ttl=60, serializer=str)
    value1 = ["a", "b"]
    cache["key1"] = value1
    value2 = ["c", "d"]
    cache["key2"] = value2
    assert len(cache) == 2

    # Update the value container in-place
    value2.append("e")
    cache.update_size("key2", 50)
    assert len(cache) == 1
    assert "key2" in cache

    # If the container size exceeds max size, it should push out itself
    value2.extend(["f", "g"])
    cache.update_size("key2", 100)
    assert len(cache) == 0


def test_sized_ttl_cache_expore_after_ttl():
    cache = SizedTTLCache(maxsize_bytes=100, ttl=1, serializer=str)
    cache["key1"] = "a"
    cache["key2"] = "b"
    assert len(cache) == 2

    time.sleep(1.1)
    assert len(cache) == 0

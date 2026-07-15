import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlflow.utils.thread_utils import map_with_context

_TEST_CTX = contextvars.ContextVar("test_ctx", default=None)


def test_map_with_context_propagates_caller_context():
    _TEST_CTX.set("caller-value")

    def worker(_):
        return _TEST_CTX.get()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(map_with_context(executor, worker, range(2)))

    assert results == ["caller-value", "caller-value"]


def test_map_with_context_concurrent_workers():
    _TEST_CTX.set("caller-value")

    def worker(_):
        return _TEST_CTX.get()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, worker, range(16)))

    assert results == ["caller-value"] * 16


def test_map_with_context_preserves_order():
    def worker(x):
        return x * 2

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(map_with_context(executor, worker, range(8)))

    assert results == [x * 2 for x in range(8)]


def test_map_with_context_propagates_exceptions():
    def worker(x):
        if x == 3:
            raise ValueError("boom")
        return x

    with ThreadPoolExecutor(max_workers=4) as executor:
        with pytest.raises(ValueError, match="boom"):
            list(map_with_context(executor, worker, range(6)))


def test_map_with_context_worker_mutations_do_not_leak_to_caller():
    _TEST_CTX.set("original")

    def worker(_):
        _TEST_CTX.set("mutated-in-worker")
        return _TEST_CTX.get()

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(map_with_context(executor, worker, range(2)))

    assert _TEST_CTX.get() == "original"

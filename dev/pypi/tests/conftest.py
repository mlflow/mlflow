from collections.abc import Iterator

import pypi
import pytest


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    pypi.clear_cache()
    # Don't actually sleep between retries inside tests.
    monkeypatch.setattr("pypi._client.time.sleep", lambda _seconds: None)
    yield
    pypi.clear_cache()

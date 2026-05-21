from collections.abc import Iterator

import pypi
import pytest


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    pypi.clear_cache()
    yield
    pypi.clear_cache()

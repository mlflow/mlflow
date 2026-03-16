import pytest

import mlflow.gateway.providers.utils as _utils


@pytest.fixture(autouse=True)
def _reset_aiohttp_session():
    """Reset the shared aiohttp session before each test so mocks take effect."""
    _utils._session = None
    yield
    _utils._session = None

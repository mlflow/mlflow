import pytest

from tests.helper_functions import start_mock_openai_server


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url

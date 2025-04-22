import pytest
from packaging.version import Version

import mlflow

from tests.helper_functions import start_mock_openai_server

is_v1 = Version(mlflow.openai._get_openai_package_version()).major >= 1


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url

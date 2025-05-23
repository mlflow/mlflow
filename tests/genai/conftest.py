from unittest import mock

import pytest

from mlflow.entities import TraceInfoV2


# TODO: Remove this fixture once `databricks-agents` releases a new version that's compatible with
# trace v3
@pytest.fixture(scope="module", autouse=True)
def mock_trace_info():
    # Monkey patch TraceInfo (V3) to use TraceInfoV2
    with mock.patch("mlflow.entities.TraceInfo", wraps=TraceInfoV2):
        yield


@pytest.fixture(autouse=True)
def mock_init_auth():
    def mocked_init_auth(config_instance):
        config_instance.host = "https://databricks.com/"
        config_instance._header_factory = lambda: {}

    with mock.patch("databricks.sdk.config.Config.init_auth", new=mocked_init_auth):
        yield


@pytest.fixture(autouse=True)
def spoof_tracking_uri_check():
    # NB: The mlflow.genai.evaluate() API is only runnable when the tracking URI is set
    # to Databricks. However, we cannot test against real Databricks server in CI, so
    # we spoof the check by patching the is_databricks_uri() function.
    with mock.patch("mlflow.genai.evaluation.base.is_databricks_uri", return_value=True):
        yield

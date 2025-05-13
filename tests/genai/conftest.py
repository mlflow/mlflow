# TODO: Remove this file once `databricks-agents` releases a new version that's compatible with
# trace v3.
from unittest import mock

import pytest

from mlflow.entities import TraceInfoV2


@pytest.fixture(scope="module", autouse=True)
def mock_trace_info():
    # Monkey patch TraceInfo (V3) to use TraceInfoV2
    with mock.patch("mlflow.entities.TraceInfo", wraps=TraceInfoV2):
        yield

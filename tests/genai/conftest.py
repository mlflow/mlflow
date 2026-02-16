import functools
import os
from unittest import mock

import pytest

import mlflow
import mlflow.telemetry.utils
from mlflow.entities.assessment import Expectation
from mlflow.entities.document import Document
from mlflow.entities.span import SpanType
from mlflow.genai.scorers.validation import IS_DBX_AGENTS_INSTALLED

# Import telemetry test fixtures from tests/telemetry/conftest.py
# This allows genai tests to use the same telemetry testing infrastructure
from tests.telemetry.conftest import (  # noqa: F401
    mock_requests,
    mock_requests_get,
    mock_telemetry_client,
    terminate_telemetry_client,
)


@pytest.fixture
def enable_telemetry_in_tests(monkeypatch):
    """
    Enable telemetry for tests that need to verify telemetry tracking.
    Use this fixture explicitly in tests that validate telemetry behavior.
    """
    monkeypatch.setattr(mlflow.telemetry.utils, "_IS_MLFLOW_TESTING_TELEMETRY", True)


@pytest.fixture(autouse=True)
def mock_init_auth():
    def mocked_init_auth(config_instance):
        config_instance.host = "https://databricks.com/"
        config_instance._header_factory = lambda: {}

    with mock.patch("databricks.sdk.config.Config.init_auth", new=mocked_init_auth):
        yield


@pytest.fixture(params=[True, False], ids=["databricks", "oss"])
def is_in_databricks(request):
    if request.param and not IS_DBX_AGENTS_INSTALLED:
        pytest.skip("Skipping Databricks test because `databricks-agents` is not installed.")

    # In CI, we run test twice, once without `databricks-agents` and once with.
    # To be effective, we skip OSS test when running with `databricks-agents`.
    if "GITHUB_ACTIONS" in os.environ:
        if not request.param and IS_DBX_AGENTS_INSTALLED:
            pytest.skip("Skipping OSS test in CI because `databricks-agents` is installed.")

    with (
        mock.patch("mlflow.genai.judges.utils.is_databricks_uri", return_value=request.param),
        mock.patch(
            "mlflow.utils.databricks_utils.is_databricks_default_tracking_uri",
            return_value=request.param,
        ),
    ):
        yield request.param


def databricks_only(func):
    """Decorator that skips test if not in Databricks environment"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not IS_DBX_AGENTS_INSTALLED:
            pytest.skip("Skipping Databricks only test.")

        with mock.patch("mlflow.get_tracking_uri", return_value="databricks"):
            return func(*args, **kwargs)

    return wrapper


@pytest.fixture
def sample_rag_trace():
    @mlflow.trace(name="rag", span_type=SpanType.AGENT)
    def _predict(question):
        # Two retrievers calls
        _retrieve_1(question)
        _retrieve_2(question)
        return "answer"

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _retrieve_1(question):
        return [
            Document(
                page_content="content_1",
                metadata={"doc_uri": "url_1"},
            ),
            Document(
                page_content="content_2",
                metadata={"doc_uri": "url_2"},
            ),
        ]

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def _retrieve_2(question):
        return [Document(page_content="content_3")]

    _predict("query")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Add expectations. Directly append to the trace info because OSS backend doesn't
    # support assessment logging yet.
    trace.info.assessments = [
        Expectation(name="expected_response", value="expected answer"),
        Expectation(name="expected_facts", value=["fact1", "fact2"]),
        Expectation(name="guidelines", value=["write in english"]),
    ]
    return trace

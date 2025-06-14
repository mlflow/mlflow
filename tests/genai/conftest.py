from unittest import mock

import pytest

import mlflow
from mlflow.entities.assessment import Expectation
from mlflow.entities.document import Document
from mlflow.entities.span import SpanType


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

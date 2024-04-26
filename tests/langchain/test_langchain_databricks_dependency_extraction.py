import sys
from unittest.mock import MagicMock

import langchain
import pytest
from packaging.version import Version

from mlflow.langchain.databricks_dependencies import (
    _extract_databricks_dependencies_from_chat_model,
    _extract_databricks_dependencies_from_llm,
    _extract_databricks_dependencies_from_retriever,
)
from mlflow.langchain.utils import IS_PICKLE_SERIALIZATION_RESTRICTED
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex


class MockDatabricksServingEndpointClient:
    def __init__(
        self,
        host: str,
        api_token: str,
        endpoint_name: str,
        databricks_uri: str,
        task: str,
    ):
        self.host = host
        self.api_token = api_token
        self.endpoint_name = endpoint_name
        self.databricks_uri = databricks_uri
        self.task = task


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_llm(monkeypatch: pytest.MonkeyPatch):
    from langchain.llms import Databricks

    monkeypatch.setattr(
        "langchain_community.llms.databricks._DatabricksServingEndpointClient",
        MockDatabricksServingEndpointClient,
    )
    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")

    llm_kwargs = {"endpoint_name": "databricks-mixtral-8x7b-instruct"}
    if IS_PICKLE_SERIALIZATION_RESTRICTED:
        llm_kwargs["allow_dangerous_deserialization"] = True

    llm = Databricks(**llm_kwargs)
    d = []
    _extract_databricks_dependencies_from_llm(llm, d)
    assert d == [DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct")]


class MockVectorSearchIndex:
    def __init__(self, endpoint_name, index_name) -> None:
        self.endpoint_name = endpoint_name
        self.name = index_name

    def describe(self):
        return {
            "primary_key": "id",
        }


class MockVectorSearchClient:
    def get_index(self, endpoint_name, index_name):
        return MockVectorSearchIndex(endpoint_name, index_name)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever(monkeypatch: pytest.MonkeyPatch):
    from langchain.embeddings import DatabricksEmbeddings
    from langchain.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(endpoint_name="dbdemos_vs_endpoint", index_name="mlflow.rag.vs_index")
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content", embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    d = []
    _extract_databricks_dependencies_from_retriever(retriever, d)
    assert d == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="dbdemos_vs_endpoint"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.embeddings import DatabricksEmbeddings
    from langchain_community.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(endpoint_name="dbdemos_vs_endpoint", index_name="mlflow.rag.vs_index")
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content", embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    d = []
    _extract_databricks_dependencies_from_retriever(retriever, d)
    assert d == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="dbdemos_vs_endpoint"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_chat(monkeypatch: pytest.MonkeyPatch):
    from langchain.chat_models import ChatDatabricks

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    d = []
    _extract_databricks_dependencies_from_chat_model(chat_model, d)
    assert d == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_chat(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.chat_models import ChatDatabricks

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    d = []
    _extract_databricks_dependencies_from_chat_model(chat_model, d)
    assert d == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]

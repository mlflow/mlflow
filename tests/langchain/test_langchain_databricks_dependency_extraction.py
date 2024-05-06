import sys
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import MagicMock

import langchain
import pytest
from packaging.version import Version

from mlflow.langchain.databricks_dependencies import (
    _DATABRICKS_CHAT_ENDPOINT_NAME_KEY,
    _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY,
    _DATABRICKS_LLM_ENDPOINT_NAME_KEY,
    _DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY,
    _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY,
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
    d = defaultdict(list)
    resources = []
    _extract_databricks_dependencies_from_llm(llm, d, resources)
    assert d.get(_DATABRICKS_LLM_ENDPOINT_NAME_KEY) == ["databricks-mixtral-8x7b-instruct"]
    assert resources == [
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct")
    ]


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
    d = defaultdict(list)
    resources = []
    _extract_databricks_dependencies_from_retriever(retriever, d, resources)
    assert d.get(_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY) == ["databricks-bge-large-en"]
    assert d.get(_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY) == ["mlflow.rag.vs_index"]
    assert d.get(_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY) == ["dbdemos_vs_endpoint"]
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
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
    d = defaultdict(list)
    resources = []
    _extract_databricks_dependencies_from_retriever(retriever, d, resources)
    assert d.get(_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY) == ["databricks-bge-large-en"]
    assert d.get(_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY) == ["mlflow.rag.vs_index"]
    assert d.get(_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY) == ["dbdemos_vs_endpoint"]
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
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
    d = defaultdict(list)
    resources = []
    _extract_databricks_dependencies_from_chat_model(chat_model, d, resources)
    assert d.get(_DATABRICKS_CHAT_ENDPOINT_NAME_KEY) == ["databricks-llama-2-70b-chat"]
    assert resources == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_chat(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.chat_models import ChatDatabricks

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    d = defaultdict(list)
    resources = []
    _extract_databricks_dependencies_from_chat_model(chat_model, d, resources)
    assert d.get(_DATABRICKS_CHAT_ENDPOINT_NAME_KEY) == ["databricks-llama-2-70b-chat"]
    assert resources == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@contextmanager
def remove_langchain_modules():
    prefixes_to_remove = [
        "langchain",
        "langchain_community",
    ]
    exceptions = ["langchain_core", "langchain_community.llms.databricks"]

    saved_modules = {}
    for mod in list(sys.modules):
        if any(mod.startswith(prefix) for prefix in prefixes_to_remove) and not any(
            mod.startswith(exc) for exc in exceptions
        ):
            saved_modules[mod] = sys.modules.pop(mod)

    try:
        yield
    finally:
        sys.modules.update(saved_modules)  # Restore all removed modules safely


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"), reason="feature not existing"
)
def test_parsing_dependency_correct_loads_langchain_modules():
    with remove_langchain_modules():
        import langchain_community

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'llms'"
        ):
            langchain_community.llms.Databricks
        d = defaultdict(list)
        resources = []
        _extract_databricks_dependencies_from_llm("", d, resources)

        # import works as expected after _extract_databricks_dependencies_from_llm
        langchain_community.llms.Databricks

    with remove_langchain_modules():
        import langchain_community

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'embeddings'"
        ):
            langchain_community.embeddings.DatabricksEmbeddings

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'vectorstores'"
        ):
            langchain_community.vectorstores.DatabricksVectorSearch

        d = defaultdict(list)
        resources = []
        _extract_databricks_dependencies_from_retriever("", d, resources)
        # import works as expected after _extract_databricks_dependencies_from_retriever
        langchain_community.vectorstores.DatabricksVectorSearch
        langchain_community.embeddings.DatabricksEmbeddings

    with remove_langchain_modules():
        import langchain_community

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'chat_models'"
        ):
            langchain_community.chat_models.ChatDatabricks

        d = defaultdict(list)
        resources = []
        _extract_databricks_dependencies_from_chat_model("", d, resources)
        # import works as expected after _extract_databricks_dependencies_from_chat_model
        langchain_community.chat_models.ChatDatabricks


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_module_removal():
    import langchain_community.llms

    langchain_community.llms.Databricks
    with remove_langchain_modules():
        import langchain_community

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'llms'"
        ):
            langchain_community.llms.Databricks

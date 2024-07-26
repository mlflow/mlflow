import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import langchain
import pytest
from packaging.version import Version

from mlflow.langchain.databricks_dependencies import (
    _detect_databricks_dependencies,
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
    resources = list(_extract_databricks_dependencies_from_llm(llm))
    assert resources == [
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct")
    ]


class MockVectorSearchIndex:
    def __init__(self, endpoint_name, index_name, has_embedding_endpoint=False) -> None:
        self.endpoint_name = endpoint_name
        self.name = index_name
        self.has_embedding_endpoint = has_embedding_endpoint

    def describe(self):
        if self.has_embedding_endpoint:
            return {
                "name": self.name,
                "endpoint_name": self.endpoint_name,
                "primary_key": "id",
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": "ml.schema.databricks_documentation",
                    "embedding_source_columns": [
                        {"name": "content", "embedding_model_endpoint_name": "embedding-model"}
                    ],
                    "pipeline_type": "TRIGGERED",
                    "pipeline_id": "79a76fcc-67ad-4ac6-8d8e-20f7d485ffa6",
                },
                "status": {
                    "detailed_state": "OFFLINE_FAILED",
                    "message": "Index creation failed.",
                    "indexed_row_count": 0,
                    "failed_status": {"error_message": ""},
                    "ready": False,
                    "index_url": "e2-dogfood.staging.cloud.databricks.com/rest_of_url",
                },
                "creator": "first.last@databricks.com",
            }
        else:
            return {
                "name": self.name,
                "endpoint_name": self.endpoint_name,
                "primary_key": "id",
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": "ml.schema.databricks_documentation",
                    "embedding_vector_columns": [],
                    "pipeline_type": "TRIGGERED",
                    "pipeline_id": "fbbd5bf1-2b9b-4a7e-8c8d-c0f6cc1030de",
                },
                "status": {
                    "detailed_state": "ONLINE",
                    "message": "Index is currently online",
                    "indexed_row_count": 17183,
                    "ready": True,
                    "index_url": "e2-dogfood.staging.cloud.databricks.com/rest_of_url",
                },
                "creator": "first.last@databricks.com",
            }


class MockVectorSearchClient:
    def get_index(self, endpoint_name, index_name, has_embedding_endpoint=False):
        return MockVectorSearchIndex(endpoint_name, index_name, has_embedding_endpoint)


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever(monkeypatch: pytest.MonkeyPatch):
    from langchain.embeddings import DatabricksEmbeddings
    from langchain.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index_1 = vsc.get_index(endpoint_name="vs_endpoint", index_name="mlflow.rag.vs_index_1")
    vs_index_2 = vsc.get_index(
        endpoint_name="vs_endpoint", index_name="mlflow.rag.vs_index_2", has_embedding_endpoint=True
    )
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    # set embedding model
    vectorstore_1 = DatabricksVectorSearch(
        vs_index_1, text_column="content", embedding=embedding_model
    )
    retriever_1 = vectorstore_1.as_retriever()

    # vs_index_2 has builtin embedding endpoint "embedding-model"
    vectorstore_2 = DatabricksVectorSearch(vs_index_2, text_column="content")
    retriever_2 = vectorstore_2.as_retriever()

    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0)

    assert list(_extract_databricks_dependencies_from_retriever(retriever_1)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    assert list(_extract_databricks_dependencies_from_retriever(retriever_2)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_2"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]

    from langchain.retrievers.multi_query import MultiQueryRetriever

    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever_1, llm=llm)
    assert list(_extract_databricks_dependencies_from_retriever(multi_query_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_1
    )
    assert list(_extract_databricks_dependencies_from_retriever(compression_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    from langchain.retrievers import EnsembleRetriever

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_1, retriever_2], weights=[0.5, 0.5]
    )
    assert list(_extract_databricks_dependencies_from_retriever(ensemble_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_2"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]

    from langchain.retrievers import TimeWeightedVectorStoreRetriever

    time_weighted_retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore_1, decay_rate=0.0000000000000000000000001, k=1
    )
    assert list(_extract_databricks_dependencies_from_retriever(time_weighted_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.embeddings import DatabricksEmbeddings
    from langchain_community.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index_1 = vsc.get_index(endpoint_name="vs_endpoint", index_name="mlflow.rag.vs_index_1")
    vs_index_2 = vsc.get_index(
        endpoint_name="vs_endpoint", index_name="mlflow.rag.vs_index_2", has_embedding_endpoint=True
    )
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    # set embedding model
    vectorstore_1 = DatabricksVectorSearch(
        vs_index_1, text_column="content", embedding=embedding_model
    )
    retriever_1 = vectorstore_1.as_retriever()

    # vs_index_2 has builtin embedding endpoint "embedding-model"
    vectorstore_2 = DatabricksVectorSearch(vs_index_2, text_column="content")
    retriever_2 = vectorstore_2.as_retriever()

    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0)

    assert list(_extract_databricks_dependencies_from_retriever(retriever_1)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    assert list(_extract_databricks_dependencies_from_retriever(retriever_2)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_2"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]

    from langchain.retrievers.multi_query import MultiQueryRetriever

    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever_1, llm=llm)
    assert list(_extract_databricks_dependencies_from_retriever(multi_query_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_1
    )
    assert list(_extract_databricks_dependencies_from_retriever(compression_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]

    from langchain.retrievers import EnsembleRetriever

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_1, retriever_2], weights=[0.5, 0.5]
    )
    assert list(_extract_databricks_dependencies_from_retriever(ensemble_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_2"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]

    from langchain.retrievers import TimeWeightedVectorStoreRetriever

    time_weighted_retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore_1, decay_rate=0.0000000000000000000000001, k=1
    )
    assert list(_extract_databricks_dependencies_from_retriever(time_weighted_retriever)) == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index_1"),
        DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever_with_embedding_endpoint_in_index(
    monkeypatch: pytest.MonkeyPatch,
):
    from langchain.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content")
    retriever = vectorstore.as_retriever()
    resources = list(_extract_databricks_dependencies_from_retriever(retriever))
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_retriever_with_embedding_endpoint_in_index(
    monkeypatch: pytest.MonkeyPatch,
):
    from langchain_community.vectorstores import DatabricksVectorSearch

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )
    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content")
    retriever = vectorstore.as_retriever()
    resources = list(_extract_databricks_dependencies_from_retriever(retriever))
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_chat(monkeypatch: pytest.MonkeyPatch):
    from langchain.chat_models import ChatDatabricks

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    resources = list(_extract_databricks_dependencies_from_chat_model(chat_model))
    assert resources == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks_chat(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.chat_models import ChatDatabricks

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    resources = list(_extract_databricks_dependencies_from_chat_model(chat_model))
    assert resources == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.0.311"), reason="feature not existing"
)
def test_parsing_dependency_from_databricks(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.chat_models import ChatDatabricks
    from langchain_community.vectorstores import DatabricksVectorSearch

    mock_get_deploy_client = MagicMock()

    monkeypatch.setattr("mlflow.deployments.get_deploy_client", mock_get_deploy_client)

    vsc = MockVectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )

    mock_module = MagicMock()
    mock_module.VectorSearchIndex = MockVectorSearchIndex

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", mock_module)

    vectorstore = DatabricksVectorSearch(vs_index, text_column="content")
    retriever = vectorstore.as_retriever()
    llm = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    llm2 = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)

    model = retriever | llm | llm2
    resources = _detect_databricks_dependencies(model)
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat"),
    ]


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
        list(_extract_databricks_dependencies_from_llm(""))

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

        list(_extract_databricks_dependencies_from_retriever(""))
        # import works as expected after _extract_databricks_dependencies_from_retriever
        langchain_community.vectorstores.DatabricksVectorSearch
        langchain_community.embeddings.DatabricksEmbeddings

    with remove_langchain_modules():
        import langchain_community

        with pytest.raises(
            AttributeError, match="module 'langchain_community' has no attribute 'chat_models'"
        ):
            langchain_community.chat_models.ChatDatabricks

        list(_extract_databricks_dependencies_from_chat_model(""))
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

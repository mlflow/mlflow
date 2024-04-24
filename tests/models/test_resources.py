from mlflow.models.resources import (
    DatabricksChatEndpoint,
    DatabricksEmbeddingEndpoint,
    DatabricksLLMEndpoint,
    DatabricksVectorSearchEndpoint,
    DatabricksVectorSearchIndexName,
    _ResourceBuilder,
)


def test_llm_endpoint():
    endpoint = DatabricksLLMEndpoint(endpoint_name="llm_server")
    expected = {"databricks_llm_endpoint_name": ["llm_server"]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == expected

    resources = [
        DatabricksLLMEndpoint(endpoint_name="llm_server"),
        DatabricksLLMEndpoint(endpoint_name="llm_server2"),
    ]
    expected = {"databricks_llm_endpoint_name": ["llm_server", "llm_server2"]}
    assert _ResourceBuilder.from_resources(resources) == expected


def test_chat_endpoint():
    endpoint = DatabricksChatEndpoint(endpoint_name="chat_server")
    expected = {"databricks_chat_endpoint_name": ["chat_server"]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == expected

    resources = [
        DatabricksChatEndpoint(endpoint_name="chat_server"),
        DatabricksChatEndpoint(endpoint_name="chat_server2"),
    ]
    expected = {"databricks_chat_endpoint_name": ["chat_server", "chat_server2"]}
    assert _ResourceBuilder.from_resources(resources) == expected


def test_embedding_endpoint():
    endpoint = DatabricksEmbeddingEndpoint(endpoint_name="embedding_server")
    expected = {"databricks_embeddings_endpoint_name": ["embedding_server"]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == expected

    resources = [
        DatabricksEmbeddingEndpoint(endpoint_name="embedding_server"),
        DatabricksEmbeddingEndpoint(endpoint_name="embedding_server2"),
    ]
    expected = {"databricks_embeddings_endpoint_name": ["embedding_server", "embedding_server2"]}
    assert _ResourceBuilder.from_resources(resources) == expected


def test_vector_search_endpoint():
    endpoint = DatabricksVectorSearchEndpoint(endpoint_name="search_server")
    expected = {"databricks_vector_search_endpoint_name": ["search_server"]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == expected

    resources = [
        DatabricksVectorSearchEndpoint(endpoint_name="search_server"),
        DatabricksVectorSearchEndpoint(endpoint_name="search_server2"),
    ]
    expected = {"databricks_vector_search_endpoint_name": ["search_server", "search_server2"]}
    assert _ResourceBuilder.from_resources(resources) == expected


def test_index_name():
    index = DatabricksVectorSearchIndexName(index_name="index1")
    expected = {"databricks_vector_search_index_name": ["index1"]}
    assert index.to_dict() == expected
    assert _ResourceBuilder.from_resources([index]) == expected

    resources = [
        DatabricksVectorSearchIndexName(index_name="index1"),
        DatabricksVectorSearchIndexName(index_name="index2"),
    ]
    expected = {"databricks_vector_search_index_name": ["index1", "index2"]}
    assert _ResourceBuilder.from_resources(resources) == expected


def test_resources():
    resources = [
        DatabricksVectorSearchIndexName(index_name="rag.studio_bugbash.databricks_docs_index"),
        DatabricksVectorSearchEndpoint(endpoint_name="azure-eastus-model-serving-2_vs_endpoint"),
        DatabricksEmbeddingEndpoint(endpoint_name="databricks-bge-large-en"),
        DatabricksLLMEndpoint(endpoint_name="llm-endpoint"),
        DatabricksChatEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
        DatabricksChatEndpoint(endpoint_name="databricks-llama-8x7b-instruct"),
    ]
    expected = {
        "databricks_vector_search_index_name": ["rag.studio_bugbash.databricks_docs_index"],
        "databricks_vector_search_endpoint_name": ["azure-eastus-model-serving-2_vs_endpoint"],
        "databricks_embeddings_endpoint_name": ["databricks-bge-large-en"],
        "databricks_llm_endpoint_name": ["llm-endpoint"],
        "databricks_chat_endpoint_name": [
            "databricks-mixtral-8x7b-instruct",
            "databricks-llama-8x7b-instruct",
        ],
    }

    assert _ResourceBuilder.from_resources(resources) == expected

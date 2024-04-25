from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    _ResourceBuilder,
)


def test_serving_endpoint():
    endpoint = DatabricksServingEndpoint(endpoint_name="llm_server")
    expected = {"serving_endpoint": [{"name": "llm_server"}]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == {"databricks": expected}


def test_index_name():
    index = DatabricksVectorSearchIndex(index_name="index1")
    expected = {"vector_search_index": [{"name": "index1"}]}
    assert index.to_dict() == expected
    assert _ResourceBuilder.from_resources([index]) == {"databricks": expected}


def test_resources():
    resources = [
        DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-8x7b-instruct"),
    ]
    expected = {
        "databricks": {
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-llama-8x7b-instruct"},
            ],
        }
    }

    assert _ResourceBuilder.from_resources(resources) == expected

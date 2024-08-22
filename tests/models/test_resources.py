import pytest

from mlflow.models.resources import (
    DEFAULT_API_VERSION,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksUCFunction,
    DatabricksVectorSearchIndex,
    _ResourceBuilder,
)


def test_serving_endpoint():
    endpoint = DatabricksServingEndpoint(endpoint_name="llm_server")
    expected = {"serving_endpoint": [{"name": "llm_server"}]}
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


def test_index_name():
    index = DatabricksVectorSearchIndex(index_name="index1")
    expected = {"vector_search_index": [{"name": "index1"}]}
    assert index.to_dict() == expected
    assert _ResourceBuilder.from_resources([index]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


def test_sql_warehouse():
    sql_warehouse = DatabricksSQLWarehouse(warehouse_id="id1")
    expected = {"sql_warehouse": [{"name": "id1"}]}
    assert sql_warehouse.to_dict() == expected
    assert _ResourceBuilder.from_resources([sql_warehouse]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


def test_uc_function():
    uc_function = DatabricksUCFunction(function_name="function")
    expected = {"uc_function": [{"name": "function"}]}
    assert uc_function.to_dict() == expected
    assert _ResourceBuilder.from_resources([uc_function]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


def test_resources():
    resources = [
        DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-8x7b-instruct"),
        DatabricksSQLWarehouse(warehouse_id="id123"),
        DatabricksUCFunction(function_name="rag.studio.test_function_1"),
        DatabricksUCFunction(function_name="rag.studio.test_function_2"),
    ]
    expected = {
        "api_version": DEFAULT_API_VERSION,
        "databricks": {
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-llama-8x7b-instruct"},
            ],
            "sql_warehouse": [{"name": "id123"}],
            "uc_function": [
                {"name": "rag.studio.test_function_1"},
                {"name": "rag.studio.test_function_2"},
            ],
        },
    }

    assert _ResourceBuilder.from_resources(resources) == expected


def test_resources_from_yaml(tmp_path):
    yaml_file = tmp_path.joinpath("resources.yaml")
    with open(yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-llama-8x7b-instruct
                sql_warehouse:
                - name: id123
                uc_function:
                - name: rag.studio.test_function_1
                - name: rag.studio.test_function_2
            """
        )

    assert _ResourceBuilder.from_yaml_file(yaml_file) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": {
            "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-llama-8x7b-instruct"},
            ],
            "sql_warehouse": [{"name": "id123"}],
            "uc_function": [
                {"name": "rag.studio.test_function_1"},
                {"name": "rag.studio.test_function_2"},
            ],
        },
    }

    with pytest.raises(OSError, match="No such file or directory: 'no-file.yaml'"):
        _ResourceBuilder.from_yaml_file("no-file.yaml")

    incorrect_version = tmp_path.joinpath("incorrect_file.yaml")
    with open(incorrect_version, "w") as f:
        f.write(
            """
            api_version: "v1"
            """
        )

    with pytest.raises(ValueError, match="Unsupported API version: v1"):
        _ResourceBuilder.from_yaml_file(incorrect_version)

    incorrect_target_uri = tmp_path.joinpath("incorrect_target_uri.yaml")
    with open(incorrect_target_uri, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks-aa:
                vector_search_index_name:
                - name: rag.studio_bugbash.databricks_docs_index
            """
        )

    with pytest.raises(ValueError, match="Unsupported target URI: databricks-aa"):
        _ResourceBuilder.from_yaml_file(incorrect_target_uri)

    incorrect_resource = tmp_path.joinpath("incorrect_resource.yaml")
    with open(incorrect_resource, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index_name:
                - name: rag.studio_bugbash.databricks_docs_index
            """
        )

    with pytest.raises(ValueError, match="Unsupported resource type: vector_search_index_name"):
        _ResourceBuilder.from_yaml_file(incorrect_resource)

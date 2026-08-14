import pytest

from mlflow.models.resources import (
    DEFAULT_API_VERSION,
    DatabricksApp,
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksLakebase,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
    _ResourceBuilder,
)


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_serving_endpoint(on_behalf_of_user):
    endpoint = DatabricksServingEndpoint(
        endpoint_name="llm_server", on_behalf_of_user=on_behalf_of_user
    )
    expected = (
        {"serving_endpoint": [{"name": "llm_server"}]}
        if on_behalf_of_user is None
        else {"serving_endpoint": [{"name": "llm_server", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert endpoint.to_dict() == expected
    assert _ResourceBuilder.from_resources([endpoint]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_index_name(on_behalf_of_user):
    index = DatabricksVectorSearchIndex(index_name="index1", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"vector_search_index": [{"name": "index1"}]}
        if on_behalf_of_user is None
        else {"vector_search_index": [{"name": "index1", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert index.to_dict() == expected
    assert _ResourceBuilder.from_resources([index]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_sql_warehouse(on_behalf_of_user):
    sql_warehouse = DatabricksSQLWarehouse(warehouse_id="id1", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"sql_warehouse": [{"name": "id1"}]}
        if on_behalf_of_user is None
        else {"sql_warehouse": [{"name": "id1", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert sql_warehouse.to_dict() == expected
    assert _ResourceBuilder.from_resources([sql_warehouse]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_uc_function(on_behalf_of_user):
    uc_function = DatabricksFunction(function_name="function", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"function": [{"name": "function"}]}
        if on_behalf_of_user is None
        else {"function": [{"name": "function", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert uc_function.to_dict() == expected
    assert _ResourceBuilder.from_resources([uc_function]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_genie_space(on_behalf_of_user):
    genie_space = DatabricksGenieSpace(genie_space_id="id1", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"genie_space": [{"name": "id1"}]}
        if on_behalf_of_user is None
        else {"genie_space": [{"name": "id1", "on_behalf_of_user": on_behalf_of_user}]}
    )

    assert genie_space.to_dict() == expected
    assert _ResourceBuilder.from_resources([genie_space]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_uc_connection(on_behalf_of_user):
    uc_function = DatabricksUCConnection(
        connection_name="slack_connection", on_behalf_of_user=on_behalf_of_user
    )
    expected = (
        {"uc_connection": [{"name": "slack_connection"}]}
        if on_behalf_of_user is None
        else {
            "uc_connection": [{"name": "slack_connection", "on_behalf_of_user": on_behalf_of_user}]
        }
    )
    assert uc_function.to_dict() == expected
    assert _ResourceBuilder.from_resources([uc_function]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_table(on_behalf_of_user):
    table = DatabricksTable(table_name="tableName", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"table": [{"name": "tableName"}]}
        if on_behalf_of_user is None
        else {"table": [{"name": "tableName", "on_behalf_of_user": on_behalf_of_user}]}
    )

    assert table.to_dict() == expected
    assert _ResourceBuilder.from_resources([table]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_app(on_behalf_of_user):
    app = DatabricksApp(app_name="id1", on_behalf_of_user=on_behalf_of_user)
    expected = (
        {"app": [{"name": "id1"}]}
        if on_behalf_of_user is None
        else {"app": [{"name": "id1", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert app.to_dict() == expected
    assert _ResourceBuilder.from_resources([app]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


@pytest.mark.parametrize("on_behalf_of_user", [True, False, None])
def test_lakebase(on_behalf_of_user):
    lakebase = DatabricksLakebase(
        database_instance_name="lakebase_name", on_behalf_of_user=on_behalf_of_user
    )
    expected = (
        {"lakebase": [{"name": "lakebase_name"}]}
        if on_behalf_of_user is None
        else {"lakebase": [{"name": "lakebase_name", "on_behalf_of_user": on_behalf_of_user}]}
    )
    assert lakebase.to_dict() == expected
    assert _ResourceBuilder.from_resources([lakebase]) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": expected,
    }


def test_resources():
    resources = [
        DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-8x7b-instruct"),
        DatabricksSQLWarehouse(warehouse_id="id123"),
        DatabricksFunction(function_name="rag.studio.test_function_1"),
        DatabricksFunction(function_name="rag.studio.test_function_2"),
        DatabricksUCConnection(connection_name="slack_connection"),
        DatabricksApp(app_name="test_databricks_app"),
        DatabricksLakebase(database_instance_name="test_databricks_lakebase"),
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
            "function": [
                {"name": "rag.studio.test_function_1"},
                {"name": "rag.studio.test_function_2"},
            ],
            "uc_connection": [{"name": "slack_connection"}],
            "app": [{"name": "test_databricks_app"}],
            "lakebase": [{"name": "test_databricks_lakebase"}],
        },
    }

    assert _ResourceBuilder.from_resources(resources) == expected


def test_invoker_resources():
    resources = [
        DatabricksVectorSearchIndex(
            index_name="rag.studio_bugbash.databricks_docs_index", on_behalf_of_user=True
        ),
        DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
        DatabricksServingEndpoint(
            endpoint_name="databricks-llama-8x7b-instruct", on_behalf_of_user=True
        ),
        DatabricksSQLWarehouse(warehouse_id="id123"),
        DatabricksFunction(function_name="rag.studio.test_function_1"),
        DatabricksFunction(function_name="rag.studio.test_function_2", on_behalf_of_user=True),
        DatabricksUCConnection(connection_name="slack_connection"),
    ]
    expected = {
        "api_version": DEFAULT_API_VERSION,
        "databricks": {
            "vector_search_index": [
                {"name": "rag.studio_bugbash.databricks_docs_index", "on_behalf_of_user": True}
            ],
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-llama-8x7b-instruct", "on_behalf_of_user": True},
            ],
            "sql_warehouse": [{"name": "id123"}],
            "function": [
                {"name": "rag.studio.test_function_1"},
                {"name": "rag.studio.test_function_2", "on_behalf_of_user": True},
            ],
            "uc_connection": [{"name": "slack_connection"}],
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
                function:
                - name: rag.studio.test_function_1
                - name: rag.studio.test_function_2
                lakebase:
                - name: test_databricks_lakebase
                uc_connection:
                - name: slack_connection
                app:
                - name: test_databricks_app
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
            "function": [
                {"name": "rag.studio.test_function_1"},
                {"name": "rag.studio.test_function_2"},
            ],
            "uc_connection": [{"name": "slack_connection"}],
            "app": [{"name": "test_databricks_app"}],
            "lakebase": [{"name": "test_databricks_lakebase"}],
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

    invokers_yaml_file = tmp_path.joinpath("invokers_resources.yaml")
    with open(invokers_yaml_file, "w") as f:
        f.write(
            """
            api_version: "1"
            databricks:
                vector_search_index:
                - name: rag.studio_bugbash.databricks_docs_index
                  on_behalf_of_user: true
                serving_endpoint:
                - name: databricks-mixtral-8x7b-instruct
                - name: databricks-llama-8x7b-instruct
                  on_behalf_of_user: true
                sql_warehouse:
                - name: id123
                function:
                - name: rag.studio.test_function_1
                  on_behalf_of_user: true
                - name: rag.studio.test_function_2
                uc_connection:
                - name: slack_connection
                  on_behalf_of_user: true
            """
        )

    assert _ResourceBuilder.from_yaml_file(invokers_yaml_file) == {
        "api_version": DEFAULT_API_VERSION,
        "databricks": {
            "vector_search_index": [
                {"name": "rag.studio_bugbash.databricks_docs_index", "on_behalf_of_user": True}
            ],
            "serving_endpoint": [
                {"name": "databricks-mixtral-8x7b-instruct"},
                {"name": "databricks-llama-8x7b-instruct", "on_behalf_of_user": True},
            ],
            "sql_warehouse": [{"name": "id123"}],
            "function": [
                {"name": "rag.studio.test_function_1", "on_behalf_of_user": True},
                {"name": "rag.studio.test_function_2"},
            ],
            "uc_connection": [{"name": "slack_connection", "on_behalf_of_user": True}],
        },
    }

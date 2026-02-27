from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
)


def test_complete_auth_policy():
    system_auth_policy = SystemAuthPolicy(
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
            DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
            DatabricksFunction(function_name="rag.studio.test_function_a"),
            DatabricksUCConnection(connection_name="test_connection_1"),
        ]
    )

    user_auth_policy = UserAuthPolicy(
        api_scopes=[
            "catalog.catalogs",
            "vectorsearch.vector-search-indexes",
            "workspace.workspace",
        ]
    )

    auth_policy = AuthPolicy(
        user_auth_policy=user_auth_policy, system_auth_policy=system_auth_policy
    )

    serialized_auth_policy = auth_policy.to_dict()

    expected_serialized_auth_policy = {
        "user_auth_policy": {
            "api_scopes": [
                "catalog.catalogs",
                "vectorsearch.vector-search-indexes",
                "workspace.workspace",
            ]
        },
        "system_auth_policy": {
            "resources": {
                "databricks": {
                    "serving_endpoint": [{"name": "databricks-mixtral-8x7b-instruct"}],
                    "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
                    "function": [{"name": "rag.studio.test_function_a"}],
                    "uc_connection": [{"name": "test_connection_1"}],
                },
                "api_version": "1",
            }
        },
    }
    assert serialized_auth_policy == expected_serialized_auth_policy


def test_user_auth_policy():
    user_auth_policy = UserAuthPolicy(
        api_scopes=[
            "catalog.catalogs",
            "vectorsearch.vector-search-indexes",
            "workspace.workspace",
        ]
    )

    auth_policy = AuthPolicy(user_auth_policy=user_auth_policy)

    serialized_auth_policy = auth_policy.to_dict()

    expected_serialized_auth_policy = {
        "system_auth_policy": {},
        "user_auth_policy": {
            "api_scopes": [
                "catalog.catalogs",
                "vectorsearch.vector-search-indexes",
                "workspace.workspace",
            ]
        },
    }
    assert serialized_auth_policy == expected_serialized_auth_policy


def test_system_auth_policy():
    system_auth_policy = SystemAuthPolicy(
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-mixtral-8x7b-instruct"),
            DatabricksVectorSearchIndex(index_name="rag.studio_bugbash.databricks_docs_index"),
            DatabricksFunction(function_name="rag.studio.test_function_a"),
            DatabricksUCConnection(connection_name="test_connection_1"),
        ]
    )

    auth_policy = AuthPolicy(system_auth_policy=system_auth_policy)

    serialized_auth_policy = auth_policy.to_dict()

    expected_serialized_auth_policy = {
        "system_auth_policy": {
            "resources": {
                "databricks": {
                    "serving_endpoint": [{"name": "databricks-mixtral-8x7b-instruct"}],
                    "vector_search_index": [{"name": "rag.studio_bugbash.databricks_docs_index"}],
                    "function": [{"name": "rag.studio.test_function_a"}],
                    "uc_connection": [{"name": "test_connection_1"}],
                },
                "api_version": "1",
            }
        },
        "user_auth_policy": {},
    }
    assert serialized_auth_policy == expected_serialized_auth_policy


def test_empty_auth_policy():
    auth_policy = AuthPolicy()

    serialized_auth_policy = auth_policy.to_dict()

    expected_serialized_auth_policy = {"system_auth_policy": {}, "user_auth_policy": {}}
    assert serialized_auth_policy == expected_serialized_auth_policy

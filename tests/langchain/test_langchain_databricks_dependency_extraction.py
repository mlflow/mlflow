from collections import Counter, defaultdict
from unittest import mock

import langchain
import pytest
from databricks.vector_search.client import VectorSearchIndex
from packaging.version import Version

from mlflow.langchain.databricks_dependencies import (
    _detect_databricks_dependencies,
    _extract_databricks_dependencies_from_chat_model,
    _extract_databricks_dependencies_from_llm,
    _extract_databricks_dependencies_from_retriever,
    _extract_dependency_list_from_lc_model,
)
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)


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


def _is_partner_package_installed():
    try:
        import langchain_databricks  # noqa: F401

        return True
    except ImportError:
        return False


def remove_langchain_community(monkeypatch):
    # Simulate the environment where langchain_community is not installed
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("langchain_community"):
            raise ImportError("No module named 'langchain_community'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)


def test_parsing_dependency_from_databricks_llm(monkeypatch: pytest.MonkeyPatch):
    from langchain_community.llms import Databricks

    from mlflow.langchain.utils import IS_PICKLE_SERIALIZATION_RESTRICTED

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


class MockVectorSearchIndex(VectorSearchIndex):
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


def get_vector_search(
    use_partner_package: bool,
    endpoint_name: str,
    index_name: str,
    has_embedding_endpoint=False,
    **kwargs,
):
    index = MockVectorSearchIndex(endpoint_name, index_name, has_embedding_endpoint)

    if use_partner_package:
        from langchain_databricks import DatabricksVectorSearch

        with mock.patch("databricks.vector_search.client.VectorSearchClient") as mock_client:
            mock_client().get_index.return_value = index
            vectorstore = DatabricksVectorSearch(
                endpoint=endpoint_name,
                index_name=index_name,
                **kwargs,
            )
    else:
        from langchain_community.vectorstores import DatabricksVectorSearch

        vectorstore = DatabricksVectorSearch(index, **kwargs)

    return vectorstore


@pytest.mark.parametrize("use_partner_package", [True, False])
def test_parsing_dependency_from_databricks_retriever(monkeypatch, use_partner_package):
    if use_partner_package and not _is_partner_package_installed():
        pytest.skip("`langchain-databricks` is not installed")

    if use_partner_package:
        from langchain_databricks import DatabricksEmbeddings
        from langchain_openai import ChatOpenAI

        remove_langchain_community(monkeypatch)
        with pytest.raises(ImportError, match="No module named 'langchain_community"):
            from langchain_community.embeddings import DatabricksEmbeddings
    else:
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.embeddings import DatabricksEmbeddings

    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

    # vs_index_1 is a direct access index
    vectorstore_1 = get_vector_search(
        use_partner_package=use_partner_package,
        endpoint_name="vs_endpoint",
        index_name="mlflow.rag.vs_index_1",
        text_column="content",
        embedding=embedding_model,
    )
    retriever_1 = vectorstore_1.as_retriever()

    # vs_index_2 has builtin embedding endpoint "embedding-model"
    vectorstore_2 = get_vector_search(
        use_partner_package=use_partner_package,
        endpoint_name="vs_endpoint",
        index_name="mlflow.rag.vs_index_2",
        has_embedding_endpoint=True,
    )
    retriever_2 = vectorstore_2.as_retriever()

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


@pytest.mark.parametrize("use_partner_package", [True, False])
def test_parsing_dependency_from_retriever_with_embedding_endpoint_in_index(use_partner_package):
    if use_partner_package and not _is_partner_package_installed():
        pytest.skip("`langchain-databricks` is not installed")

    vectorstore = get_vector_search(
        use_partner_package=use_partner_package,
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )
    retriever = vectorstore.as_retriever()
    resources = list(_extract_databricks_dependencies_from_retriever(retriever))
    assert resources == [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
    ]


def test_parsing_dependency_from_agent(monkeypatch: pytest.MonkeyPatch):
    from databricks.sdk.service.catalog import FunctionInfo
    from langchain.agents import initialize_agent
    from langchain_openai.llms import OpenAI

    try:
        from langchain_community.tools.databricks import UCFunctionToolkit
    except Exception:
        return

    # When get is called return a function
    def mock_function_get(self, function_name):
        components = function_name.split(".")
        # Initialize agent used below requires functions to take in exactly one parameter
        param_dict = {
            "parameters": [
                {
                    "name": "param",
                    "parameter_type": "PARAM",
                    "position": 0,
                    "type_json": '{"name":"param","type":"string","nullable":true,"metadata":{}}',
                    "type_name": "STRING",
                    "type_precision": 0,
                    "type_scale": 0,
                    "type_text": "string",
                }
            ]
        }
        # Add the catalog, schema and name to the function Info followed by the parameter
        return FunctionInfo.from_dict(
            {
                "catalog_name": components[0],
                "schema_name": components[1],
                "name": components[2],
                "input_params": param_dict,
            }
        )

    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.get", mock_function_get)

    toolkit = UCFunctionToolkit(warehouse_id="testId1").include("rag.test.test_function")
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        toolkit.get_tools(),
        llm,
        verbose=True,
    )

    resources = list(_extract_dependency_list_from_lc_model(agent))
    assert resources == [
        DatabricksFunction(function_name="rag.test.test_function"),
        DatabricksSQLWarehouse(warehouse_id="testId1"),
    ]


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.1.0"),
    reason="Tools are not supported the way we want in earlier versions",
)
@pytest.mark.parametrize("use_partner_package", [True, False])
def test_parsing_multiple_dependency_from_agent(monkeypatch, use_partner_package):
    if use_partner_package and not _is_partner_package_installed():
        pytest.skip("`langchain-databricks` is not installed")

    from databricks.sdk.service.catalog import FunctionInfo
    from langchain.agents import initialize_agent
    from langchain.tools.retriever import create_retriever_tool

    if use_partner_package:
        from langchain_databricks import ChatDatabricks

        remove_langchain_community(monkeypatch)
        with pytest.raises(ImportError, match="No module named 'langchain_community"):
            from langchain_community.chat_models import ChatDatabricks
    else:
        from langchain_community.chat_models import ChatDatabricks

    def mock_function_get(self, function_name):
        components = function_name.split(".")
        param_dict = {
            "parameters": [
                {
                    "name": "param",
                    "parameter_type": "PARAM",
                    "position": 0,
                    "type_json": '{"name":"param","type":"string","nullable":true,"metadata":{}}',
                    "type_name": "STRING",
                    "type_precision": 0,
                    "type_scale": 0,
                    "type_text": "string",
                }
            ]
        }
        return FunctionInfo.from_dict(
            {
                "catalog_name": components[0],
                "schema_name": components[1],
                "name": components[2],
                "input_params": param_dict,
            }
        )

    # In addition to above now handle the case where a '*' is passed in and list all the functions
    def mock_function_list(self, catalog_name, schema_name):
        assert catalog_name == "rag"
        assert schema_name == "test"
        return [
            FunctionInfo(full_name="rag.test.test_function"),
            FunctionInfo(full_name="rag.test.test_function_2"),
            FunctionInfo(full_name="rag.test.test_function_3"),
        ]

    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.get", mock_function_get)
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.list", mock_function_list)

    include_uc_function_tools = False
    try:
        from langchain_community.tools.databricks import UCFunctionToolkit

        include_uc_function_tools = True
    except Exception:
        include_uc_function_tools = False

    uc_function_tools = (
        (UCFunctionToolkit(warehouse_id="testId1").include("rag.test.*").get_tools())
        if include_uc_function_tools
        else []
    )
    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)

    vectorstore = get_vector_search(
        use_partner_package=use_partner_package,
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )
    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(retriever, "vs_index_name", "vs_index_desc")

    agent = initialize_agent(
        uc_function_tools + [retriever_tool],
        chat_model,
        verbose=True,
    )
    resources = list(_extract_dependency_list_from_lc_model(agent))
    # Ensure all resources are added in
    expected = [
        DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
        DatabricksServingEndpoint(endpoint_name="embedding-model"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat"),
    ]
    if include_uc_function_tools:
        expected = [
            DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat"),
            DatabricksFunction(function_name="rag.test.test_function"),
            DatabricksFunction(function_name="rag.test.test_function_2"),
            DatabricksFunction(function_name="rag.test.test_function_3"),
            DatabricksVectorSearchIndex(index_name="mlflow.rag.vs_index"),
            DatabricksServingEndpoint(endpoint_name="embedding-model"),
            DatabricksSQLWarehouse(warehouse_id="testId1"),
        ]

    def build_resource_map(resources):
        resource_map = defaultdict(list)

        for resource in resources:
            resource_type = resource.type.value
            resource_name = resource.to_dict()[resource_type][0]["name"]
            resource_map[resource_type].append(resource_name)

        return dict(resource_map)

    # Build maps for resources and expected resources
    resource_maps = build_resource_map(resources)
    expected_maps = build_resource_map(expected)

    assert len(resource_maps) == len(expected_maps)

    for resource_type in resource_maps:
        assert Counter(resource_maps[resource_type]) == Counter(
            expected_maps.get(resource_type, [])
        )


@pytest.mark.parametrize("use_partner_package", [True, False])
def test_parsing_dependency_from_databricks_chat(monkeypatch, use_partner_package):
    if use_partner_package and not _is_partner_package_installed():
        pytest.skip("`langchain-databricks` is not installed")

    if use_partner_package:
        from langchain_databricks import ChatDatabricks

        remove_langchain_community(monkeypatch)
        with pytest.raises(ImportError, match="No module named 'langchain_community"):
            from langchain_community.chat_models import ChatDatabricks
    else:
        from langchain_community.chat_models import ChatDatabricks

    chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=500)
    resources = list(_extract_databricks_dependencies_from_chat_model(chat_model))
    assert resources == [DatabricksServingEndpoint(endpoint_name="databricks-llama-2-70b-chat")]


@pytest.mark.parametrize("use_partner_package", [True, False])
def test_parsing_dependency_from_databricks(monkeypatch, use_partner_package):
    if use_partner_package and not _is_partner_package_installed():
        pytest.skip("`langchain-databricks` is not installed")

    if use_partner_package:
        from langchain_databricks import ChatDatabricks

        remove_langchain_community(monkeypatch)
        with pytest.raises(ImportError, match="No module named 'langchain_community"):
            from langchain_community.chat_models import ChatDatabricks
    else:
        from langchain_community.chat_models import ChatDatabricks

    vectorstore = get_vector_search(
        use_partner_package=use_partner_package,
        endpoint_name="dbdemos_vs_endpoint",
        index_name="mlflow.rag.vs_index",
        has_embedding_endpoint=True,
    )
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


@pytest.mark.skipif(
    Version(langchain.__version__) < Version("0.2.0"),
    reason="unitycatalog-langchain depends on langchain>=0.2.0",
)
def test_parsing_unitycatalog_tool_as_dependency(monkeypatch: pytest.MonkeyPatch):
    from databricks.sdk.service.catalog import FunctionInfo
    from langchain.agents import initialize_agent
    from langchain_openai.llms import OpenAI
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit

    # When get is called return a function
    def mock_function_get(self, function_name):
        components = function_name.split(".")
        # Initialize agent used below requires functions to take in exactly one parameter
        param_dict = {
            "parameters": [
                {
                    "name": "param",
                    "parameter_type": "PARAM",
                    "position": 0,
                    "type_json": '{"name":"param","type":"string","nullable":true,"metadata":{}}',
                    "type_name": "STRING",
                    "type_precision": 0,
                    "type_scale": 0,
                    "type_text": "string",
                }
            ]
        }
        # Add the catalog, schema and name to the function Info followed by the parameter
        return FunctionInfo.from_dict(
            {
                "catalog_name": components[0],
                "schema_name": components[1],
                "name": components[2],
                "input_params": param_dict,
            }
        )

    monkeypatch.setenv("DATABRICKS_HOST", "my-default-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-default-token")
    monkeypatch.setattr("databricks.sdk.service.catalog.FunctionsAPI.get", mock_function_get)

    with mock.patch(
        "unitycatalog.ai.core.databricks.DatabricksFunctionClient._validate_warehouse_type",
        return_value=None,
    ):
        client = DatabricksFunctionClient(warehouse_id="testId1")
    toolkit = UCFunctionToolkit(function_names=["rag.test.test_function"], client=client)
    llm = OpenAI(temperature=0)
    agent = initialize_agent(
        toolkit.tools,
        llm,
        verbose=True,
    )

    resources = list(_extract_dependency_list_from_lc_model(agent))
    assert resources == [
        DatabricksFunction(function_name="rag.test.test_function"),
        DatabricksSQLWarehouse(warehouse_id="testId1"),
    ]

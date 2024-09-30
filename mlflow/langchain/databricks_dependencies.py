import importlib
import inspect
import logging
import warnings
from typing import Any, Generator, List, Optional, Set

from packaging import version

from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
    Resource,
)

_logger = logging.getLogger(__name__)


def _get_embedding_model_endpoint_names(index):
    embedding_model_endpoint_names = []
    desc = index.describe()
    delta_sync_index_spec = desc.get("delta_sync_index_spec", {})
    embedding_source_columns = delta_sync_index_spec.get("embedding_source_columns", [])
    for column in embedding_source_columns:
        embedding_model_endpoint_name = column.get("embedding_model_endpoint_name", None)
        if embedding_model_endpoint_name:
            embedding_model_endpoint_names.append(embedding_model_endpoint_name)
    return embedding_model_endpoint_names


def _get_vectorstore_from_retriever(retriever) -> Generator[Resource, None, None]:
    vectorstore = getattr(retriever, "vectorstore", None)
    if _isinstance_with_multiple_modules(
        vectorstore,
        "DatabricksVectorSearch",
        ["langchain_databricks", "langchain_community.vectorstores", "langchain.vectorstores"],
    ):
        index = vectorstore.index
        yield DatabricksVectorSearchIndex(index_name=index.name)
        for embedding_endpoint in _get_embedding_model_endpoint_names(index):
            yield DatabricksServingEndpoint(endpoint_name=embedding_endpoint)

    embeddings = getattr(vectorstore, "embeddings", None)
    if _isinstance_with_multiple_modules(
        embeddings,
        "DatabricksEmbeddings",
        ["langchain_databricks", "langchain_community.embeddings", "langchain.embeddings"],
    ):
        yield DatabricksServingEndpoint(endpoint_name=embeddings.endpoint)


def _extract_databricks_dependencies_from_tools(tools) -> Generator[Resource, None, None]:
    if isinstance(tools, list):
        warehouse_ids = set()
        for tool in tools:
            if _isinstance_with_multiple_modules(
                tool, "BaseTool", ["langchain_core.tools", "langchain_community.tools"]
            ):
                # Handle Retriever tools
                if hasattr(tool.func, "keywords") and "retriever" in tool.func.keywords:
                    retriever = tool.func.keywords.get("retriever")
                    yield from _get_vectorstore_from_retriever(retriever)
                else:
                    try:
                        from langchain_community.tools.databricks import UCFunctionToolkit
                    except Exception:
                        continue
                    # Tools here are a part of the BaseTool and have no attribute of a
                    # WarehouseID Extract the global variables of the function defined
                    # in the tool to get the UCFunctionToolkit Constants
                    nonlocal_vars = inspect.getclosurevars(tool.func).nonlocals
                    if "self" in nonlocal_vars and isinstance(
                        nonlocal_vars.get("self"), UCFunctionToolkit
                    ):
                        uc_function_toolkit = nonlocal_vars.get("self")
                        # As we are iterating through each tool, adding a warehouse id everytime
                        # is a duplicative resouce. Use a set to dedup warehouse ids and add
                        # them in the end
                        warehouse_ids.add(uc_function_toolkit.warehouse_id)

                        # In langchain the names of the tools are modified to have underscores:
                        # main.catalog.test_func -> main_catalog_test_func
                        # The original name of the tool is stored as the key in the tools
                        # dictionary. This code finds the correct tool and extract the key
                        langchain_tool_name = tool.name
                        filtered_tool_names = [
                            tool_name
                            for tool_name, uc_tool in uc_function_toolkit.tools.items()
                            if uc_tool.name == langchain_tool_name
                        ]
                        # This should always have the length 1
                        for tool_name in filtered_tool_names:
                            yield DatabricksFunction(function_name=tool_name)
        # Add the deduped warehouse ids
        for warehouse_id in warehouse_ids:
            yield DatabricksSQLWarehouse(warehouse_id=warehouse_id)


def _extract_databricks_dependencies_from_retriever(retriever) -> Generator[Resource, None, None]:
    # ContextualCompressionRetriever uses attribute "base_retriever"
    if hasattr(retriever, "base_retriever"):
        retriever = getattr(retriever, "base_retriever", None)

    # Most other retrievers use attribute "retriever"
    if hasattr(retriever, "retriever"):
        retriever = getattr(retriever, "retriever", None)

    # EnsembleRetriever uses attribute "retrievers" for multiple retrievers
    if hasattr(retriever, "retrievers"):
        retriever = getattr(retriever, "retrievers", None)

    # If there are multiple retrievers, we iterate over them to get dependencies from each of them
    if isinstance(retriever, list):
        for single_retriever in retriever:
            yield from _get_vectorstore_from_retriever(single_retriever)
    else:
        yield from _get_vectorstore_from_retriever(retriever)


def _extract_databricks_dependencies_from_llm(llm) -> Generator[Resource, None, None]:
    if _isinstance_with_multiple_modules(
        llm, "Databricks", ["langchain.llms", "langchain_community.llms"]
    ):
        yield DatabricksServingEndpoint(endpoint_name=llm.endpoint_name)


def _extract_databricks_dependencies_from_chat_model(chat_model) -> Generator[Resource, None, None]:
    if _isinstance_with_multiple_modules(
        chat_model,
        "ChatDatabricks",
        ["langchain_databricks", "langchain.chat_models", "langchain_community.chat_models"],
    ):
        yield DatabricksServingEndpoint(endpoint_name=chat_model.endpoint)


def _extract_databricks_dependencies_from_tool_nodes(tool_node) -> Generator[Resource, None, None]:
    try:
        from langgraph.prebuilt.tool_node import ToolNode

        if isinstance(tool_node, ToolNode):
            yield from _extract_databricks_dependencies_from_tools(
                list(tool_node.tools_by_name.values())
            )
    except ImportError:
        pass


def _isinstance_with_multiple_modules(
    object: Any, class_name: str, from_modules: List[str]
) -> bool:
    """
    Databricks components are defined in different modules in LangChain e.g.
    langchain, langchain_community, langchain_databricks due to historical migrations.
    To keep backward compatibility, we need to check if the object is an instance of the
    class defined in any of those different modules.

    Args:
        object: The object to check
        class_name: The name of the class to check
        from_modules: The list of modules to import the class from.
    """
    # Suppress LangChainDeprecationWarning for old imports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        for module_path in from_modules:
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                if cls is not None and isinstance(object, cls):
                    return True
            except (ImportError, AttributeError):
                pass

    return False


_LEGACY_MODEL_ATTR_SET = {
    "llm",  # LLMChain
    "retriever",  # RetrievalQA
    "llm_chain",  # StuffDocumentsChain, MapRerankDocumentsChain, MapReduceDocumentsChain
    "question_generator",  # BaseConversationalRetrievalChain
    "initial_llm_chain",  # RefineDocumentsChain
    "refine_llm_chain",  # RefineDocumentsChain
    "combine_documents_chain",  # RetrievalQA, ReduceDocumentsChain
    "combine_docs_chain",  # BaseConversationalRetrievalChain
    "collapse_documents_chain",  # ReduceDocumentsChain,
    "agent",  # Agent,
    "tools",  # Tools
}


def _extract_dependency_list_from_lc_model(lc_model) -> Generator[Resource, None, None]:
    """
    This function contains the logic to examine a non-Runnable component of a langchain model.
    The logic here does not cover all legacy chains. If you need to support a custom chain,
    you need to monkey patch this function.
    """
    if lc_model is None:
        return

    # leaf node
    yield from _extract_databricks_dependencies_from_chat_model(lc_model)
    yield from _extract_databricks_dependencies_from_retriever(lc_model)
    yield from _extract_databricks_dependencies_from_llm(lc_model)
    yield from _extract_databricks_dependencies_from_tools(lc_model)
    yield from _extract_databricks_dependencies_from_tool_nodes(lc_model)

    # recursively inspect legacy chain
    for attr_name in _LEGACY_MODEL_ATTR_SET:
        yield from _extract_dependency_list_from_lc_model(getattr(lc_model, attr_name, None))


def _traverse_runnable(
    lc_model,
    visited: Optional[Set[int]] = None,
) -> Generator[Resource, None, None]:
    """
    This function contains the logic to traverse a langchain_core.runnables.RunnableSerializable
    object. It first inspects the current object using _extract_dependency_list_from_lc_model
    and then, if the current object is a Runnable, it recursively inspects its children returned
    by lc_model.get_graph().nodes.values().
    This function supports arbitrary LCEL chain.
    """
    import pydantic
    from langchain_core.runnables import Runnable, RunnableLambda

    visited = visited or set()
    current_object_id = id(lc_model)
    if current_object_id in visited:
        return

    # Visit the current object
    visited.add(current_object_id)
    yield from _extract_dependency_list_from_lc_model(lc_model)

    if isinstance(lc_model, Runnable):
        # Visit the returned graph
        if isinstance(lc_model, RunnableLambda) and version.parse(
            pydantic.version.VERSION
        ) >= version.parse("2.0"):
            nodes = _get_nodes_from_runnable_lambda(lc_model)
        else:
            nodes = lc_model.get_graph().nodes.values()

        for node in nodes:
            yield from _traverse_runnable(node.data, visited)
    else:
        # No-op for non-runnable, if any
        pass


def _get_nodes_from_runnable_lambda(lc_model):
    """
    This is a workaround for the LangGraph issue: https://github.com/langchain-ai/langgraph/issues/1856

    For RunnableLambda, we calling lc_model.get_graph() to get the nodes, which inspect
    the input and output schema using wrapped function's type annotation. However, the
    prebuilt graph (e.g. create_react_agent) from LangGraph uses typing.TypeDict annotation,
    which is not supported by Pydantic V2 on Python < 3.12. If we try to inspect such
    function, it will raise the following error:

        pydantic.errors.PydanticUserError: Please use `typing_extensions.TypedDict`
        instead of`typing.TypedDict` on Python < 3.12. For further information visit
        https://errors.pydantic.dev/2.9/u/typed-dict-version

    Therefore, we cannot use get_graph() for RunnableLambda until LangGraph fixes this issue.
    Luckily, we are not interested in the input/output nodes for extracting databricks
    dependencies. We only care about lc_models.deps, which contains the components that
    the RunnableLambda depends on. Therefore, this function extracts the necessary parts
    from the original get_graph() function, dropping the input/output related logic.
    https://github.com/langchain-ai/langchain/blob/2ea5f60cc5747a334550273a5dba1b70b11414c1/libs/core/langchain_core/runnables/base.py#L4493C1-L4512C46
    """
    if deps := lc_model.deps:
        nodes = []
        for dep in deps:
            dep_graph = dep.get_graph()
            dep_graph.trim_first_node()
            dep_graph.trim_last_node()
            nodes.extend(dep_graph.nodes.values())
    else:
        nodes = lc_model.get_graph().nodes.values()
    return nodes


def _detect_databricks_dependencies(lc_model, log_errors_as_warnings=True) -> List[Resource]:
    """
    Detects the databricks dependencies of a langchain model and returns a list of
    detected endpoint names and index names.

    lc_model can be an arbitrary [chain that is built with LCEL](https://python.langchain.com
    /docs/modules/chains#lcel-chains), which is a langchain_core.runnables.RunnableSerializable.
    [Legacy chains](https://python.langchain.com/docs/modules/chains#legacy-chains) have limited
    support. Only RetrievalQA, StuffDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain,
    MapRerankDocumentsChain, MapReduceDocumentsChain, BaseConversationalRetrievalChain are
    supported. If you need to support a custom chain, you need to monkey patch
    the function mlflow.langchain.databricks_dependencies._extract_dependency_list_from_lc_model().

    For an LCEL chain, all the langchain_core.runnables.RunnableSerializable nodes will be
    traversed.

    If a retriever is found, it will be used to extract the databricks vector search and embeddings
    dependencies.
    If an llm is found, it will be used to extract the databricks llm dependencies.
    If a chat_model is found, it will be used to extract the databricks chat dependencies.
    """
    try:
        dependency_list = list(_traverse_runnable(lc_model))
        # Filter out duplicate dependencies so same dependencies are not added multiple times
        # We can't use set here as the object is not hashable so we need to filter it out manually.
        unique_dependencies = []
        for dependency in dependency_list:
            if dependency not in unique_dependencies:
                unique_dependencies.append(dependency)
        return unique_dependencies
    except Exception:
        if log_errors_as_warnings:
            _logger.warning(
                "Unable to detect Databricks dependencies. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)
            return []
        raise

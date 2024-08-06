import logging
from typing import Generator, List, Optional, Set

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex, Resource

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
    try:
        from langchain.embeddings import DatabricksEmbeddings as LegacyDatabricksEmbeddings
        from langchain.vectorstores import (
            DatabricksVectorSearch as LegacyDatabricksVectorSearch,
        )
    except ImportError:
        from langchain_community.embeddings import (
            DatabricksEmbeddings as LegacyDatabricksEmbeddings,
        )
        from langchain_community.vectorstores import (
            DatabricksVectorSearch as LegacyDatabricksVectorSearch,
        )

    from langchain_community.embeddings import DatabricksEmbeddings
    from langchain_community.vectorstores import DatabricksVectorSearch

    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore:
        if isinstance(vectorstore, (DatabricksVectorSearch, LegacyDatabricksVectorSearch)):
            index = vectorstore.index
            yield DatabricksVectorSearchIndex(index_name=index.name)
            for embedding_endpoint in _get_embedding_model_endpoint_names(index):
                yield DatabricksServingEndpoint(endpoint_name=embedding_endpoint)

        embeddings = getattr(vectorstore, "embeddings", None)
        if isinstance(embeddings, (DatabricksEmbeddings, LegacyDatabricksEmbeddings)):
            yield DatabricksServingEndpoint(endpoint_name=embeddings.endpoint)


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
    try:
        from langchain.llms import Databricks as LegacyDatabricks
    except ImportError:
        from langchain_community.llms import Databricks as LegacyDatabricks

    from langchain_community.llms import Databricks

    if isinstance(llm, (LegacyDatabricks, Databricks)):
        yield DatabricksServingEndpoint(endpoint_name=llm.endpoint_name)


def _extract_databricks_dependencies_from_chat_model(chat_model) -> Generator[Resource, None, None]:
    try:
        from langchain.chat_models import ChatDatabricks as LegacyChatDatabricks
    except ImportError:
        from langchain_community.chat_models import (
            ChatDatabricks as LegacyChatDatabricks,
        )

    from langchain_community.chat_models import ChatDatabricks

    if isinstance(chat_model, (LegacyChatDatabricks, ChatDatabricks)):
        yield DatabricksServingEndpoint(endpoint_name=chat_model.endpoint)


_LEGACY_MODEL_ATTR_SET = {
    "llm",  # LLMChain
    "retriever",  # RetrievalQA
    "llm_chain",  # StuffDocumentsChain, MapRerankDocumentsChain, MapReduceDocumentsChain
    "question_generator",  # BaseConversationalRetrievalChain
    "initial_llm_chain",  # RefineDocumentsChain
    "refine_llm_chain",  # RefineDocumentsChain
    "combine_documents_chain",  # RetrievalQA, ReduceDocumentsChain
    "combine_docs_chain",  # BaseConversationalRetrievalChain
    "collapse_documents_chain",  # ReduceDocumentsChain
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
    from langchain_core.runnables import Runnable

    visited = visited or set()
    current_object_id = id(lc_model)
    if current_object_id in visited:
        return

    # Visit the current object
    visited.add(current_object_id)
    yield from _extract_dependency_list_from_lc_model(lc_model)

    if isinstance(lc_model, Runnable):
        # Visit the returned graph
        for node in lc_model.get_graph().nodes.values():
            yield from _traverse_runnable(node.data, visited)
    else:
        # No-op for non-runnable, if any
        pass


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

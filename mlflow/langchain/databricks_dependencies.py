import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

_DATABRICKS_DEPENDENCY_KEY = "databricks_dependency"
_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY = "databricks_vector_search_index_name"
_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY = "databricks_vector_search_endpoint_name"
_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY = "databricks_embeddings_endpoint_name"
_DATABRICKS_LLM_ENDPOINT_NAME_KEY = "databricks_llm_endpoint_name"
_DATABRICKS_CHAT_ENDPOINT_NAME_KEY = "databricks_chat_endpoint_name"


_logger = logging.getLogger(__name__)


def _extract_databricks_dependencies_from_retriever(
    retriever, dependency_dict: DefaultDict[str, List[Any]]
):
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
            dependency_dict[_DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY].append(index.name)
            dependency_dict[_DATABRICKS_VECTOR_SEARCH_ENDPOINT_NAME_KEY].append(index.endpoint_name)

        embeddings = getattr(vectorstore, "embeddings", None)
        if isinstance(embeddings, (DatabricksEmbeddings, LegacyDatabricksEmbeddings)):
            dependency_dict[_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY].append(embeddings.endpoint)
        elif (
            callable(getattr(vectorstore, "_is_databricks_managed_embeddings", None))
            and vectorstore._is_databricks_managed_embeddings()
        ):
            dependency_dict[_DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY].append(
                "_is_databricks_managed_embeddings"
            )


def _extract_databricks_dependencies_from_llm(llm, dependency_dict: DefaultDict[str, List[Any]]):
    try:
        from langchain.llms import Databricks as LegacyDatabricks
    except ImportError:
        from langchain_community.llms import Databricks as LegacyDatabricks

    from langchain_community.llms import Databricks

    if isinstance(llm, (LegacyDatabricks, Databricks)):
        dependency_dict[_DATABRICKS_LLM_ENDPOINT_NAME_KEY].append(llm.endpoint_name)


def _extract_databricks_dependencies_from_chat_model(
    chat_model, dependency_dict: DefaultDict[str, List[Any]]
):
    try:
        from langchain.chat_models import ChatDatabricks as LegacyChatDatabricks
    except ImportError:
        from langchain_community.chat_models import (
            ChatDatabricks as LegacyChatDatabricks,
        )

    from langchain_community.chat_models import ChatDatabricks

    if isinstance(chat_model, (LegacyChatDatabricks, ChatDatabricks)):
        dependency_dict[_DATABRICKS_CHAT_ENDPOINT_NAME_KEY].append(chat_model.endpoint)


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


def _extract_dependency_dict_from_lc_model(lc_model, dependency_dict: DefaultDict[str, List[Any]]):
    """
    This function contains the logic to examine a non-Runnable component of a langchain model.
    The logic here does not cover all legacy chains. If you need to support a custom chain,
    you need to monkey patch this function.
    """
    if lc_model is None:
        return

    # leaf node
    _extract_databricks_dependencies_from_chat_model(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_retriever(lc_model, dependency_dict)
    _extract_databricks_dependencies_from_llm(lc_model, dependency_dict)

    # recursively inspect legacy chain
    for attr_name in _LEGACY_MODEL_ATTR_SET:
        _extract_dependency_dict_from_lc_model(getattr(lc_model, attr_name, None), dependency_dict)


def _traverse_runnable(lc_model, dependency_dict: DefaultDict[str, List[Any]], visited: Set[str]):
    """
    This function contains the logic to traverse a langchain_core.runnables.RunnableSerializable
    object. It first inspects the current object using _extract_dependency_dict_from_lc_model
    and then, if the current object is a Runnable, it recursively inspects its children returned
    by lc_model.get_graph().nodes.values().
    This function supports arbitrary LCEL chain.
    """
    from langchain_core.runnables import Runnable

    current_object_id = id(lc_model)
    if current_object_id in visited:
        return

    # Visit the current object
    visited.add(current_object_id)
    _extract_dependency_dict_from_lc_model(lc_model, dependency_dict)

    if isinstance(lc_model, Runnable):
        # Visit the returned graph
        for node in lc_model.get_graph().nodes.values():
            _traverse_runnable(node.data, dependency_dict, visited)
    else:
        # No-op for non-runnable, if any
        pass
    return


def _detect_databricks_dependencies(lc_model, log_errors_as_warnings=True) -> Dict[str, List[Any]]:
    """
    Detects the databricks dependencies of a langchain model and returns a dictionary of
    detected endpoint names and index names.

    lc_model can be an arbitrary [chain that is built with LCEL](https://python.langchain.com
    /docs/modules/chains#lcel-chains), which is a langchain_core.runnables.RunnableSerializable.
    [Legacy chains](https://python.langchain.com/docs/modules/chains#legacy-chains) have limited
    support. Only RetrievalQA, StuffDocumentsChain, ReduceDocumentsChain, RefineDocumentsChain,
    MapRerankDocumentsChain, MapReduceDocumentsChain, BaseConversationalRetrievalChain are
    supported. If you need to support a custom chain, you need to monkey patch
    the function mlflow.langchain.databricks_dependencies._extract_dependency_dict_from_lc_model().

    For an LCEL chain, all the langchain_core.runnables.RunnableSerializable nodes will be
    traversed.

    If a retriever is found, it will be used to extract the databricks vector search and embeddings
    dependencies.
    If an llm is found, it will be used to extract the databricks llm dependencies.
    If a chat_model is found, it will be used to extract the databricks chat dependencies.
    """
    try:
        dependency_dict = defaultdict(list)
        _traverse_runnable(lc_model, dependency_dict, set())
        return dict(dependency_dict)
    except Exception:
        if log_errors_as_warnings:
            _logger.warning(
                "Unable to detect Databricks dependencies. "
                "Set logging level to DEBUG to see the full traceback."
            )
            _logger.debug("", exc_info=True)
            return {}
        raise

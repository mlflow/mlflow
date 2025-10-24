def import_base_retriever():
    """Import BaseRetriever from the correct location."""
    try:
        from langchain.schema import BaseRetriever

        return BaseRetriever
    except (ImportError, ModuleNotFoundError):
        from langchain_core.retrievers import BaseRetriever

        return BaseRetriever


def import_document():
    """Import Document from the correct location."""
    try:
        from langchain.schema import Document

        return Document
    except (ImportError, ModuleNotFoundError):
        from langchain_core.documents import Document

        return Document


def import_runnable():
    """Import Runnable from the correct location."""
    try:
        from langchain.schema.runnable import Runnable

        return Runnable
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import Runnable

        return Runnable


def import_runnable_parallel():
    """Import RunnableParallel from the correct location."""
    try:
        from langchain.schema.runnable import RunnableParallel

        return RunnableParallel
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableParallel

        return RunnableParallel


def import_runnable_sequence():
    """Import RunnableSequence from the correct location."""
    try:
        from langchain.schema.runnable import RunnableSequence

        return RunnableSequence
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence


def import_runnable_branch():
    """Import RunnableBranch from the correct location."""
    try:
        from langchain.schema.runnable import RunnableBranch

        return RunnableBranch
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableBranch

        return RunnableBranch


def import_runnable_binding():
    """Import RunnableBinding from the correct location."""
    try:
        from langchain.schema.runnable import RunnableBinding

        return RunnableBinding
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableBinding

        return RunnableBinding


def import_runnable_lambda():
    """Import RunnableLambda from the correct location."""
    try:
        from langchain.schema.runnable import RunnableLambda

        return RunnableLambda
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda


def import_runnable_passthrough():
    """Import RunnablePassthrough from the correct location."""
    try:
        from langchain.schema.runnable import RunnablePassthrough

        return RunnablePassthrough
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnablePassthrough

        return RunnablePassthrough


def import_runnable_assign():
    """Import RunnableAssign from the correct location."""
    try:
        from langchain.schema.runnable.passthrough import RunnableAssign

        return RunnableAssign
    except (ImportError, ModuleNotFoundError):
        from langchain_core.runnables import RunnableAssign

        return RunnableAssign


def import_str_output_parser():
    """Import StrOutputParser from the correct location."""
    try:
        from langchain.schema.output_parser import StrOutputParser

        return StrOutputParser
    except (ImportError, ModuleNotFoundError):
        from langchain_core.output_parsers import StrOutputParser

        return StrOutputParser


def try_import_agent_executor():
    """
    Import AgentExecutor if available (langchain < 1.0.0).

    AgentExecutor was removed in langchain 1.0.0 and replaced by LangGraph.
    Returns None if not available.
    """
    try:
        from langchain.agents.agent import AgentExecutor

        return AgentExecutor
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def try_import_chain():
    """
    Import Chain base class if available.

    Returns None if not available.
    """
    try:
        from langchain.chains.base import Chain

        return Chain
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def try_import_simple_chat_model():
    """
    Import SimpleChatModel if available (moved in langchain 1.0.0).

    SimpleChatModel moved from langchain.chat_models.base to langchain_core.language_models.
    Returns None if not available.
    """
    try:
        from langchain.chat_models.base import SimpleChatModel

        return SimpleChatModel
    except (ImportError, ModuleNotFoundError, AttributeError):
        try:
            from langchain_core.language_models import SimpleChatModel

            return SimpleChatModel
        except (ImportError, ModuleNotFoundError, AttributeError):
            return None


def import_chat_prompt_template():
    """Import ChatPromptTemplate from the correct location."""
    try:
        from langchain.prompts import ChatPromptTemplate

        return ChatPromptTemplate
    except (ImportError, ModuleNotFoundError):
        from langchain_core.prompts import ChatPromptTemplate

        return ChatPromptTemplate


def import_base_callback_handler():
    """Import BaseCallbackHandler from the correct location."""
    try:
        from langchain.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler
    except (ImportError, ModuleNotFoundError):
        from langchain_core.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler


def import_callback_manager_for_chain_run():
    """Import CallbackManagerForChainRun from the correct location."""
    try:
        from langchain.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun
    except (ImportError, ModuleNotFoundError):
        from langchain_core.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun


def import_async_callback_manager_for_chain_run():
    """Import AsyncCallbackManagerForChainRun from the correct location."""
    try:
        from langchain.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun
    except (ImportError, ModuleNotFoundError):
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun


def try_import_llm_chain():
    """
    Import LLMChain if available (removed in langchain 1.0.0).

    Returns None if not available.
    """
    try:
        from langchain.chains.llm import LLMChain

        return LLMChain
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


def try_import_base_chat_model():
    """Import BaseChatModel from the correct location."""
    try:
        from langchain.chat_models.base import BaseChatModel

        return BaseChatModel
    except (ImportError, ModuleNotFoundError):
        try:
            from langchain_core.language_models.chat_models import BaseChatModel

            return BaseChatModel
        except (ImportError, ModuleNotFoundError):
            return None

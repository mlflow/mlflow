def import_base_retriever():
    try:
        from langchain.schema import BaseRetriever

        return BaseRetriever
    except ImportError:
        from langchain_core.retrievers import BaseRetriever

        return BaseRetriever


def import_document():
    try:
        from langchain.schema import Document

        return Document
    except ImportError:
        from langchain_core.documents import Document

        return Document


def import_runnable():
    try:
        from langchain.schema.runnable import Runnable

        return Runnable
    except ImportError:
        from langchain_core.runnables import Runnable

        return Runnable


def import_runnable_parallel():
    try:
        from langchain.schema.runnable import RunnableParallel

        return RunnableParallel
    except ImportError:
        from langchain_core.runnables import RunnableParallel

        return RunnableParallel


def import_runnable_sequence():
    try:
        from langchain.schema.runnable import RunnableSequence

        return RunnableSequence
    except ImportError:
        from langchain_core.runnables import RunnableSequence

        return RunnableSequence


def import_runnable_branch():
    try:
        from langchain.schema.runnable import RunnableBranch

        return RunnableBranch
    except ImportError:
        from langchain_core.runnables import RunnableBranch

        return RunnableBranch


def import_runnable_binding():
    try:
        from langchain.schema.runnable import RunnableBinding

        return RunnableBinding
    except ImportError:
        from langchain_core.runnables import RunnableBinding

        return RunnableBinding


def import_runnable_lambda():
    try:
        from langchain.schema.runnable import RunnableLambda

        return RunnableLambda
    except ImportError:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda


def import_runnable_passthrough():
    try:
        from langchain.schema.runnable import RunnablePassthrough

        return RunnablePassthrough
    except ImportError:
        from langchain_core.runnables import RunnablePassthrough

        return RunnablePassthrough


def import_runnable_assign():
    try:
        from langchain.schema.runnable.passthrough import RunnableAssign

        return RunnableAssign
    except ImportError:
        from langchain_core.runnables import RunnableAssign

        return RunnableAssign


def import_str_output_parser():
    try:
        from langchain.schema.output_parser import StrOutputParser

        return StrOutputParser
    except ImportError:
        from langchain_core.output_parsers import StrOutputParser

        return StrOutputParser


def try_import_agent_executor():
    try:
        from langchain.agents.agent import AgentExecutor

        return AgentExecutor
    except ImportError:
        return None


def try_import_chain():
    try:
        from langchain.chains.base import Chain

        return Chain
    except ImportError:
        return None


def try_import_simple_chat_model():
    try:
        from langchain.chat_models.base import SimpleChatModel

        return SimpleChatModel
    except ImportError:
        pass

    try:
        from langchain_core.language_models import SimpleChatModel

        return SimpleChatModel
    except ImportError:
        return None


def import_chat_prompt_template():
    try:
        from langchain.prompts import ChatPromptTemplate

        return ChatPromptTemplate
    except ImportError:
        from langchain_core.prompts import ChatPromptTemplate

        return ChatPromptTemplate


def import_base_callback_handler():
    try:
        from langchain.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler
    except ImportError:
        from langchain_core.callbacks.base import BaseCallbackHandler

        return BaseCallbackHandler


def import_callback_manager_for_chain_run():
    try:
        from langchain.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun
    except ImportError:
        from langchain_core.callbacks.manager import CallbackManagerForChainRun

        return CallbackManagerForChainRun


def import_async_callback_manager_for_chain_run():
    try:
        from langchain.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun
    except ImportError:
        from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun

        return AsyncCallbackManagerForChainRun


def try_import_llm_chain():
    try:
        from langchain.chains.llm import LLMChain

        return LLMChain
    except ImportError:
        return None


def try_import_base_chat_model():
    try:
        from langchain.chat_models.base import BaseChatModel

        return BaseChatModel
    except ImportError:
        pass

    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        return BaseChatModel
    except ImportError:
        return None

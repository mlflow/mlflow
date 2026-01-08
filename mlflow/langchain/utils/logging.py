"""Utility functions for mlflow.langchain."""

import functools
import importlib
import json
import logging
import os
import shutil
import types
from functools import lru_cache
from importlib.util import find_spec
from typing import Any, Callable, NamedTuple

import cloudpickle
import yaml
from packaging.version import Version

import mlflow
from mlflow.models.utils import _validate_and_get_model_code_path
from mlflow.utils.class_utils import _get_class_from_string

_AGENT_PRIMITIVES_FILE_NAME = "agent_primitive_args.json"
_AGENT_PRIMITIVES_DATA_KEY = "agent_primitive_data"
_AGENT_DATA_FILE_NAME = "agent.yaml"
_AGENT_DATA_KEY = "agent_data"
_TOOLS_DATA_FILE_NAME = "tools.pkl"
_TOOLS_DATA_KEY = "tools_data"
_LOADER_FN_FILE_NAME = "loader_fn.pkl"
_LOADER_FN_KEY = "loader_fn"
_LOADER_ARG_KEY = "loader_arg"
_PERSIST_DIR_NAME = "persist_dir_data"
_PERSIST_DIR_KEY = "persist_dir"
_MODEL_DATA_YAML_FILE_NAME = "model.yaml"
_MODEL_DATA_PKL_FILE_NAME = "model.pkl"
_MODEL_DATA_FOLDER_NAME = "model"
_MODEL_DATA_KEY = "model_data"
_MODEL_TYPE_KEY = "model_type"
_RUNNABLE_LOAD_KEY = "runnable_load"
_BASE_LOAD_KEY = "base_load"
_CONFIG_LOAD_KEY = "config_load"
_PICKLE_LOAD_KEY = "pickle_load"
_MODEL_LOAD_KEY = "model_load"
_UNSUPPORTED_MODEL_WARNING_MESSAGE = (
    "MLflow does not guarantee support for Chains outside of the subclasses of LLMChain, found %s"
)
_UNSUPPORTED_LLM_WARNING_MESSAGE = (
    "MLflow does not guarantee support for LLMs outside of HuggingFacePipeline and OpenAI, found %s"
)


try:
    import langchain_community

    # Since langchain-community 0.0.27, saving or loading a module that relies on the pickle
    # deserialization requires passing `allow_dangerous_deserialization=True`.
    IS_PICKLE_SERIALIZATION_RESTRICTED = Version(langchain_community.__version__) >= Version(
        "0.0.27"
    )
except ImportError:
    IS_PICKLE_SERIALIZATION_RESTRICTED = False

logger = logging.getLogger(__name__)


@lru_cache
def base_lc_types():
    """
    Get base LangChain types (Chain, AgentExecutor, BaseRetriever).

    Note: AgentExecutor was removed in langchain 1.0.0. Use LangGraph instead.
    """
    from mlflow.langchain._compat import (
        import_base_retriever,
        try_import_agent_executor,
        try_import_chain,
    )

    types = []

    if chain_cls := try_import_chain():
        types.append(chain_cls)

    if agent_executor_cls := try_import_agent_executor():
        types.append(agent_executor_cls)

    types.append(import_base_retriever())

    return tuple(types)


@lru_cache
def picklable_runnable_types():
    """
    Runnable types that can be pickled and unpickled by cloudpickle.
    """
    from mlflow.langchain._compat import (
        import_chat_prompt_template,
        import_runnable_lambda,
        import_runnable_passthrough,
        try_import_simple_chat_model,
    )

    types = [
        import_chat_prompt_template(),
        import_runnable_passthrough(),
        import_runnable_lambda(),
    ]

    if simple_chat_model := try_import_simple_chat_model():
        types.insert(0, simple_chat_model)

    return tuple(types)


@lru_cache
def lc_runnable_with_steps_types():
    from mlflow.langchain._compat import import_runnable_parallel, import_runnable_sequence

    return (import_runnable_parallel(), import_runnable_sequence())


def lc_runnable_assign_types():
    from mlflow.langchain._compat import import_runnable_assign

    return (import_runnable_assign(),)


def lc_runnable_branch_types():
    from mlflow.langchain._compat import import_runnable_branch

    return (import_runnable_branch(),)


def lc_runnable_binding_types():
    from mlflow.langchain._compat import import_runnable_binding

    return (import_runnable_binding(),)


def lc_runnables_types():
    return (
        picklable_runnable_types()
        + lc_runnable_with_steps_types()
        + lc_runnable_branch_types()
        + lc_runnable_assign_types()
        + lc_runnable_binding_types()
    )


def langgraph_types():
    try:
        from langgraph.graph.state import CompiledStateGraph

        return (CompiledStateGraph,)
    except ImportError:
        return ()


def supported_lc_types():
    return base_lc_types() + lc_runnables_types() + langgraph_types()


# Wrapping as a function to avoid calling supported_lc_types() at import time
def get_unsupported_model_message(model_type):
    return (
        "MLflow langchain flavor only supports subclasses of "
        f"{supported_lc_types()}, found {model_type}."
    )


@lru_cache
def custom_type_to_loader_dict():
    # helper function to load output_parsers from config
    def _load_output_parser(config: dict[str, Any]) -> Any:
        """Load output parser."""
        from mlflow.langchain._compat import import_str_output_parser

        output_parser_type = config.pop("_type", None)
        if output_parser_type == "default":
            return import_str_output_parser()(**config)
        else:
            raise ValueError(f"Unsupported output parser {output_parser_type}")

    return {"default": _load_output_parser}


class _SpecialChainInfo(NamedTuple):
    loader_arg: str


def _get_special_chain_info_or_none(chain):
    for (
        special_chain_class,
        loader_arg,
    ) in _get_map_of_special_chain_class_to_loader_arg().items():
        if isinstance(chain, special_chain_class):
            return _SpecialChainInfo(loader_arg=loader_arg)


@lru_cache
def _get_map_of_special_chain_class_to_loader_arg():
    class_name_to_loader_arg = {
        "langchain.chains.RetrievalQA": "retriever",
        "langchain.chains.APIChain": "requests_wrapper",
        "langchain.chains.HypotheticalDocumentEmbedder": "embeddings",
    }
    # SQLDatabaseChain is in langchain_experimental (since version 0.0.247+)
    if find_spec("langchain_experimental"):
        # Add this entry only if langchain_experimental is installed
        class_name_to_loader_arg["langchain_experimental.sql.SQLDatabaseChain"] = "database"

    class_to_loader_arg = {}
    try:
        from mlflow.langchain.retriever_chain import _RetrieverChain

        class_to_loader_arg[_RetrieverChain] = "retriever"
    except ImportError:
        pass

    for class_name, loader_arg in class_name_to_loader_arg.items():
        try:
            cls = _get_class_from_string(class_name)
            class_to_loader_arg[cls] = loader_arg
        except Exception:
            logger.warning(
                "Unexpected import failure for class '%s'. Please file an issue at"
                " https://github.com/mlflow/mlflow/issues/.",
                class_name,
                exc_info=True,
            )

    return class_to_loader_arg


@lru_cache
def _get_supported_llms():
    supported_llms = set()

    def try_adding_llm(module, class_name):
        if cls := getattr(module, class_name, None):
            supported_llms.add(cls)

    def safe_import_and_add(module_name, class_name):
        """Add conditional support for `partner` and `community` APIs in langchain"""
        try:
            module = importlib.import_module(module_name)
            try_adding_llm(module, class_name)
        except ImportError:
            pass

    safe_import_and_add("langchain.llms.openai", "OpenAI")
    # HuggingFacePipeline is moved to langchain_huggingface since langchain 0.2.0
    safe_import_and_add("langchain.llms", "HuggingFacePipeline")
    safe_import_and_add("langchain.langchain_huggingface", "HuggingFacePipeline")
    safe_import_and_add("langchain_openai", "OpenAI")
    safe_import_and_add("langchain_databricks", "ChatDatabricks")
    safe_import_and_add("databricks_langchain", "ChatDatabricks")

    for llm_name in ["Databricks", "Mlflow"]:
        safe_import_and_add("langchain.llms", llm_name)

    for chat_model_name in [
        "ChatDatabricks",
        "ChatMlflow",
        "ChatOpenAI",
        "AzureChatOpenAI",
    ]:
        safe_import_and_add("langchain.chat_models", chat_model_name)

    return supported_llms


def _agent_executor_contains_unsupported_llm(lc_model, _SUPPORTED_LLMS):
    from mlflow.langchain._compat import try_import_agent_executor

    agent_executor_cls = try_import_agent_executor()
    if agent_executor_cls is None:
        return False

    return (
        isinstance(lc_model, agent_executor_cls)
        # 'RunnableMultiActionAgent' object has no attribute 'llm_chain'
        and hasattr(lc_model.agent, "llm_chain")
        and not any(
            isinstance(lc_model.agent.llm_chain.llm, supported_llm)
            for supported_llm in _SUPPORTED_LLMS
        )
    )


# temp_dir is only required when lc_model could be a file path
def _validate_and_prepare_lc_model_or_path(lc_model, loader_fn, temp_dir=None):
    if isinstance(lc_model, str):
        return _validate_and_get_model_code_path(lc_model, temp_dir)

    if not isinstance(lc_model, supported_lc_types()):
        raise mlflow.MlflowException.invalid_parameter_value(
            get_unsupported_model_message(type(lc_model).__name__)
        )

    _SUPPORTED_LLMS = _get_supported_llms()

    from mlflow.langchain._compat import try_import_llm_chain

    llm_chain_cls = try_import_llm_chain()
    if (
        llm_chain_cls
        and isinstance(lc_model, llm_chain_cls)
        and not any(isinstance(lc_model.llm, supported_llm) for supported_llm in _SUPPORTED_LLMS)
    ):
        logger.warning(
            _UNSUPPORTED_LLM_WARNING_MESSAGE,
            type(lc_model.llm).__name__,
        )

    if _agent_executor_contains_unsupported_llm(lc_model, _SUPPORTED_LLMS):
        logger.warning(
            _UNSUPPORTED_LLM_WARNING_MESSAGE,
            type(lc_model.agent.llm_chain.llm).__name__,
        )

    if special_chain_info := _get_special_chain_info_or_none(lc_model):
        if loader_fn is None:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"For {type(lc_model).__name__} models, a `loader_fn` must be provided."
            )
        if not isinstance(loader_fn, types.FunctionType):
            raise mlflow.MlflowException.invalid_parameter_value(
                "The `loader_fn` must be a function that returns a {loader_arg}.".format(
                    loader_arg=special_chain_info.loader_arg
                )
            )

    # If lc_model is a retriever, wrap it in a _RetrieverChain
    from mlflow.langchain._compat import import_base_retriever

    BaseRetriever = import_base_retriever()
    if isinstance(lc_model, BaseRetriever):
        try:
            from mlflow.langchain.retriever_chain import _RetrieverChain
        except ImportError:
            raise mlflow.MlflowException.invalid_parameter_value(
                "_RetrieverChain is not available. It requires langchain<1.0.0. "
                "For langchain>=1.0.0, please use LangGraph instead."
            )

        if loader_fn is None:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"For {type(lc_model).__name__} models, a `loader_fn` must be provided."
            )
        if not isinstance(loader_fn, types.FunctionType):
            raise mlflow.MlflowException.invalid_parameter_value(
                "The `loader_fn` must be a function that returns a retriever."
            )
        lc_model = _RetrieverChain(retriever=lc_model)

    return lc_model


def _save_base_lcs(model, path, loader_fn=None, persist_dir=None):
    from mlflow.langchain._compat import (
        try_import_agent_executor,
        try_import_base_chat_model,
        try_import_chain,
        try_import_llm_chain,
    )

    AgentExecutor = try_import_agent_executor()
    Chain = try_import_chain()
    LLMChain = try_import_llm_chain()
    BaseChatModel = try_import_base_chat_model()

    model_data_path = os.path.join(path, _MODEL_DATA_YAML_FILE_NAME)
    model_data_kwargs = {
        _MODEL_DATA_KEY: _MODEL_DATA_YAML_FILE_NAME,
        _MODEL_LOAD_KEY: _BASE_LOAD_KEY,
    }

    is_llm_chain = LLMChain and isinstance(model, LLMChain)
    is_base_chat_model = BaseChatModel and isinstance(model, BaseChatModel)

    if is_llm_chain or is_base_chat_model:
        model.save(model_data_path)
    elif AgentExecutor and isinstance(model, AgentExecutor):
        if model.agent and getattr(model.agent, "llm_chain", None):
            model.agent.llm_chain.save(model_data_path)

        if model.agent:
            agent_data_path = os.path.join(path, _AGENT_DATA_FILE_NAME)
            model.save_agent(agent_data_path)
            model_data_kwargs[_AGENT_DATA_KEY] = _AGENT_DATA_FILE_NAME

        if model.tools:
            tools_data_path = os.path.join(path, _TOOLS_DATA_FILE_NAME)
            try:
                with open(tools_data_path, "wb") as f:
                    cloudpickle.dump(model.tools, f)
            except Exception as e:
                raise mlflow.MlflowException(
                    "Error when attempting to pickle the AgentExecutor tools. "
                    "This model likely does not support serialization."
                ) from e
            model_data_kwargs[_TOOLS_DATA_KEY] = _TOOLS_DATA_FILE_NAME
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                "For initializing the AgentExecutor, tools must be provided."
            )

        key_to_ignore = ["llm_chain", "agent", "tools", "callback_manager"]
        temp_dict = {k: v for k, v in model.__dict__.items() if k not in key_to_ignore}

        agent_primitive_path = os.path.join(path, _AGENT_PRIMITIVES_FILE_NAME)
        with open(agent_primitive_path, "w") as config_file:
            json.dump(temp_dict, config_file, indent=4)

        model_data_kwargs[_AGENT_PRIMITIVES_DATA_KEY] = _AGENT_PRIMITIVES_FILE_NAME

    elif special_chain_info := _get_special_chain_info_or_none(model):
        # Save loader_fn by pickling
        loader_fn_path = os.path.join(path, _LOADER_FN_FILE_NAME)
        with open(loader_fn_path, "wb") as f:
            cloudpickle.dump(loader_fn, f)
        model_data_kwargs[_LOADER_FN_KEY] = _LOADER_FN_FILE_NAME
        model_data_kwargs[_LOADER_ARG_KEY] = special_chain_info.loader_arg

        if persist_dir is not None:
            if os.path.exists(persist_dir):
                # Save persist_dir by copying into subdir _PERSIST_DIR_NAME
                persist_dir_data_path = os.path.join(path, _PERSIST_DIR_NAME)
                shutil.copytree(persist_dir, persist_dir_data_path)
                model_data_kwargs[_PERSIST_DIR_KEY] = _PERSIST_DIR_NAME
            else:
                raise mlflow.MlflowException.invalid_parameter_value(
                    "The directory provided for persist_dir does not exist."
                )

        # Save model
        model.save(model_data_path)
    elif Chain and isinstance(model, Chain):
        logger.warning(get_unsupported_model_message(type(model).__name__))
        model.save(model_data_path)
    else:
        raise mlflow.MlflowException.invalid_parameter_value(
            get_unsupported_model_message(type(model).__name__)
        )

    return model_data_kwargs


def _load_from_pickle(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def _load_from_json(path):
    with open(path) as f:
        return json.load(f)


def _load_from_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _get_path_by_key(root_path, key, conf):
    key_path = conf.get(key)
    return os.path.join(root_path, key_path) if key_path else None


def _patch_loader(loader_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Patch LangChain loader function like load_chain() to handle pickle deserialization.

    Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization
    requires the `allow_dangerous_deserialization` flag to be set to True, for security reasons.

    Args:
        loader_func: The LangChain loader function to be patched e.g. load_chain().

    Returns:
        The patched loader function.
    """
    if not IS_PICKLE_SERIALIZATION_RESTRICTED:
        return loader_func

    # For LangChain >= 0.3.0, we can pass `allow_dangerous_deserialization` flag
    # via the loader APIs. Since the model is serialized by the user (or someone who has
    # access to the tracking server), it is safe to set this flag to True.
    def patched_loader(*args, **kwargs):
        return loader_func(*args, **kwargs, allow_dangerous_deserialization=True)

    return patched_loader


def _load_base_lcs(
    local_model_path,
    conf,
):
    lc_model_path = os.path.join(
        local_model_path, conf.get(_MODEL_DATA_KEY, _MODEL_DATA_YAML_FILE_NAME)
    )

    agent_path = _get_path_by_key(local_model_path, _AGENT_DATA_KEY, conf)
    tools_path = _get_path_by_key(local_model_path, _TOOLS_DATA_KEY, conf)
    agent_primitive_path = _get_path_by_key(local_model_path, _AGENT_PRIMITIVES_DATA_KEY, conf)
    loader_fn_path = _get_path_by_key(local_model_path, _LOADER_FN_KEY, conf)
    persist_dir = _get_path_by_key(local_model_path, _PERSIST_DIR_KEY, conf)

    model_type = conf.get(_MODEL_TYPE_KEY)
    loader_arg = conf.get(_LOADER_ARG_KEY)

    load_chain = None
    try:
        from langchain.chains.loading import load_chain
    except ImportError:
        pass

    _RetrieverChain = None
    try:
        from mlflow.langchain.retriever_chain import _RetrieverChain
    except ImportError:
        pass

    if loader_arg is not None:
        if loader_fn_path is None:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Missing file for loader_fn which is required to build the model."
            )
        loader_fn = _load_from_pickle(loader_fn_path)
        kwargs = {loader_arg: loader_fn(persist_dir)}
        if _RetrieverChain and model_type == _RetrieverChain.__name__:
            model = _RetrieverChain.load(lc_model_path, **kwargs).retriever
        else:
            if load_chain is None:
                raise mlflow.MlflowException(
                    "Cannot load model: langchain.chains.loading.load_chain is not available. "
                    "This may be because you're using langchain>=1.0.0. "
                    "Please use a model saved with langchain>=1.0.0."
                )
            model = _patch_loader(load_chain)(lc_model_path, **kwargs)
    elif agent_path is None and tools_path is None:
        if load_chain is None:
            raise mlflow.MlflowException(
                "Cannot load model: langchain.chains.loading.load_chain is not available. "
                "This may be because you're using langchain>=1.0.0. "
                "Please use a model saved with langchain>=1.0.0."
            )
        model = _patch_loader(load_chain)(lc_model_path)
    else:
        try:
            from langchain.agents import initialize_agent
        except ImportError:
            raise mlflow.MlflowException(
                "Cannot load AgentExecutor: langchain.agents.initialize_agent is not available. "
                "AgentExecutor was removed in langchain 1.0.0. Please use LangGraph instead."
            )

        if load_chain is None:
            raise mlflow.MlflowException(
                "Cannot load model: langchain.chains.loading.load_chain is not available. "
                "This may be because you're using langchain>=1.0.0. "
                "Please use a model saved with langchain>=1.0.0."
            )

        llm = _patch_loader(load_chain)(lc_model_path)
        tools = []
        kwargs = {}

        if os.path.exists(tools_path):
            tools = _load_from_pickle(tools_path)
        else:
            raise mlflow.MlflowException(
                "Missing file for tools which is required to build the AgentExecutor object."
            )

        if os.path.exists(agent_primitive_path):
            kwargs = _load_from_json(agent_primitive_path)

        model = initialize_agent(tools=tools, llm=llm, agent_path=agent_path, **kwargs)
    return model


def patch_langchain_type_to_cls_dict(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def _load_chat_openai():
            from langchain_community.chat_models import ChatOpenAI

            return ChatOpenAI

        def _load_azure_chat_openai():
            from langchain_community.chat_models import AzureChatOpenAI

            return AzureChatOpenAI

        def _load_chat_databricks():
            from databricks_langchain import ChatDatabricks

            return ChatDatabricks

        def _patched_get_type_to_cls_dict(original):
            def _wrapped():
                return {
                    **original(),
                    "openai-chat": _load_chat_openai,
                    "azure-openai-chat": _load_azure_chat_openai,
                    "chat-databricks": _load_chat_databricks,
                }

            return _wrapped

        modules_to_patch = [
            "langchain_databricks",
            "langchain.llms",
            "langchain_community.llms.loading",
        ]
        originals = {}
        for name in modules_to_patch:
            try:
                module = importlib.import_module(name)
                originals[name] = module.get_type_to_cls_dict  # Record original impl for cleanup
            except (ImportError, AttributeError):
                continue
            module.get_type_to_cls_dict = _patched_get_type_to_cls_dict(originals[name])

        try:
            return func(*args, **kwargs)
        finally:
            # Clean up the patch
            for module_name, original_impl in originals.items():
                module = importlib.import_module(module_name)
                module.get_type_to_cls_dict = original_impl

    return wrapper

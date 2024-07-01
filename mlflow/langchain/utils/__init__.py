"""Utility functions for mlflow.langchain."""

import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import types
import warnings
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple

import cloudpickle
import yaml
from packaging import version
from packaging.version import Version

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _validate_and_get_model_code_path
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
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
_UNSUPPORTED_MODEL_ERROR_MESSAGE = (
    "MLflow langchain flavor only supports subclasses of "
    "langchain.chains.base.Chain, langchain.agents.agent.AgentExecutor, "
    "langchain.schema.BaseRetriever, langchain.schema.runnable.RunnableSequence, "
    "langchain.schema.runnable.RunnableLambda, "
    "langchain.schema.runnable.RunnableParallel, "
    "langchain.schema.runnable.RunnablePassthrough, "
    "langchain.schema.runnable.passthrough.RunnableAssign instances, "
    "found {instance_type}"
)
_UNSUPPORTED_MODEL_WARNING_MESSAGE = (
    "MLflow does not guarantee support for Chains outside of the subclasses of LLMChain, found %s"
)
_UNSUPPORTED_LLM_WARNING_MESSAGE = (
    "MLflow does not guarantee support for LLMs outside of HuggingFacePipeline and OpenAI, found %s"
)

# Minimum version of langchain required to support ChatOpenAI and AzureChatOpenAI in MLflow
# Before this version, our hacky patching to support loading ChatOpenAI and AzureChatOpenAI
# will not work.
_LC_MIN_VERSION_SUPPORT_CHAT_OPEN_AI = Version("0.0.307")
_LC_MIN_VERSION_SUPPORT_RUNNABLE = Version("0.0.311")
_CHAT_MODELS_ERROR_MSG = re.compile("Loading (openai-chat|azure-openai-chat) LLM not supported")


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
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.schema

    return (
        langchain.chains.base.Chain,
        langchain.agents.agent.AgentExecutor,
        langchain.schema.BaseRetriever,
    )


@lru_cache
def picklable_runnable_types():
    """
    Runnable types that can be pickled and unpickled by cloudpickle.
    """
    from langchain.chat_models.base import SimpleChatModel
    from langchain.prompts import ChatPromptTemplate

    types = (
        SimpleChatModel,
        ChatPromptTemplate,
    )

    try:
        from langchain.schema.runnable import (
            RunnableLambda,
            RunnablePassthrough,
        )

        types += (RunnableLambda, RunnablePassthrough)
    except ImportError:
        pass

    return types


@lru_cache
def lc_runnable_with_steps_types():
    # import them separately because they are added
    # in different versions of langchain
    try:
        from langchain.schema.runnable import RunnableSequence

        types = (RunnableSequence,)
    except ImportError:
        types = ()

    try:
        from langchain.schema.runnable import RunnableParallel

        types += (RunnableParallel,)
    except ImportError:
        pass

    return types


def lc_runnable_assign_types():
    try:
        from langchain.schema.runnable.passthrough import RunnableAssign

        return (RunnableAssign,)
    except ImportError:
        return ()


def lc_runnable_branch_types():
    try:
        from langchain.schema.runnable import RunnableBranch

        return (RunnableBranch,)
    except ImportError:
        return ()


def lc_runnable_binding_types():
    try:
        from langchain.schema.runnable import RunnableBinding

        return (RunnableBinding,)
    except ImportError:
        return ()


def lc_runnables_types():
    return (
        picklable_runnable_types()
        + lc_runnable_with_steps_types()
        + lc_runnable_branch_types()
        + lc_runnable_assign_types()
        + lc_runnable_binding_types()
    )


def supported_lc_types():
    return base_lc_types() + lc_runnables_types()


@lru_cache
def custom_type_to_loader_dict():
    # helper function to load output_parsers from config
    def _load_output_parser(config: dict) -> dict:
        """Load output parser."""
        from langchain.schema.output_parser import StrOutputParser

        output_parser_type = config.pop("_type", None)
        if output_parser_type == "default":
            return StrOutputParser(**config)
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
    import langchain

    from mlflow.langchain.retriever_chain import _RetrieverChain

    class_name_to_loader_arg = {
        "langchain.chains.RetrievalQA": "retriever",
        "langchain.chains.APIChain": "requests_wrapper",
        "langchain.chains.HypotheticalDocumentEmbedder": "embeddings",
    }
    # NB: SQLDatabaseChain was migrated to langchain_experimental beginning with version 0.0.247
    if version.parse(langchain.__version__) <= version.parse("0.0.246"):
        class_name_to_loader_arg["langchain.chains.SQLDatabaseChain"] = "database"
    else:
        if find_spec("langchain_experimental"):
            # Add this entry only if langchain_experimental is installed
            class_name_to_loader_arg["langchain_experimental.sql.SQLDatabaseChain"] = "database"

    class_to_loader_arg = {
        _RetrieverChain: "retriever",
    }
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
    import langchain.chat_models
    import langchain.llms

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

    for llm_name in ["Databricks", "Mlflow"]:
        try_adding_llm(langchain.llms, llm_name)

    for chat_model_name in [
        "ChatDatabricks",
        "ChatMlflow",
        "ChatOpenAI",
        "AzureChatOpenAI",
    ]:
        try_adding_llm(langchain.chat_models, chat_model_name)

    return supported_llms


# temp_dir is only required when lc_model could be a file path
def _validate_and_prepare_lc_model_or_path(lc_model, loader_fn, temp_dir=None):
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.chains.llm
    import langchain.llms.huggingface_hub
    import langchain.llms.openai
    import langchain.schema

    # lc_model is a file path
    if isinstance(lc_model, str):
        return _validate_and_get_model_code_path(lc_model, temp_dir)

    if not isinstance(lc_model, supported_lc_types()):
        raise mlflow.MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(lc_model).__name__)
        )

    _SUPPORTED_LLMS = _get_supported_llms()
    if isinstance(lc_model, langchain.chains.llm.LLMChain) and not any(
        isinstance(lc_model.llm, supported_llm) for supported_llm in _SUPPORTED_LLMS
    ):
        logger.warning(
            _UNSUPPORTED_LLM_WARNING_MESSAGE,
            type(lc_model.llm).__name__,
        )

    if isinstance(lc_model, langchain.agents.agent.AgentExecutor) and not any(
        isinstance(lc_model.agent.llm_chain.llm, supported_llm) for supported_llm in _SUPPORTED_LLMS
    ):
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
    if isinstance(lc_model, langchain.schema.BaseRetriever):
        from mlflow.langchain.retriever_chain import _RetrieverChain

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
    from langchain.agents.agent import AgentExecutor
    from langchain.chains.base import Chain
    from langchain.chains.llm import LLMChain
    from langchain.chat_models.base import BaseChatModel

    model_data_path = os.path.join(path, _MODEL_DATA_YAML_FILE_NAME)
    model_data_kwargs = {
        _MODEL_DATA_KEY: _MODEL_DATA_YAML_FILE_NAME,
        _MODEL_LOAD_KEY: _BASE_LOAD_KEY,
    }

    if isinstance(model, (LLMChain, BaseChatModel)):
        model.save(model_data_path)
    elif isinstance(model, AgentExecutor):
        if model.agent and model.agent.llm_chain:
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
    elif isinstance(model, Chain):
        logger.warning(
            _UNSUPPORTED_MODEL_WARNING_MESSAGE,
            type(model).__name__,
        )
        model.save(model_data_path)
    else:
        raise mlflow.MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(model).__name__)
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


def _patch_loader(loader_func: Callable) -> Callable:
    """
    Patch LangChain loader function like load_chain() to handle the breaking change introduced in
    LangChain 0.1.12.

    Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization
    requires the `allow_dangerous_deserialization` flag to be set to True, for security reasons.
    However, this flag could not be specified via the LangChain's loading API like load_chain(),
    load_llm(), until LangChain 0.1.14. As a result, such module cannot be loaded with MLflow
    with earlier version of LangChain and we have to tell the user to upgrade LangChain to 0.0.14
    or above.

    Args:
        loader_func: The LangChain loader function to be patched e.g. load_chain().

    Returns:
        The patched loader function.
    """
    if not IS_PICKLE_SERIALIZATION_RESTRICTED:
        return loader_func

    import langchain

    if Version(langchain.__version__) >= Version("0.1.14"):
        # For LangChain 0.1.14 and above, we can pass `allow_dangerous_deserialization` flag
        # via the loader APIs. Since the model is serialized by the user (or someone who has
        # access to the tracking server), it is safe to set this flag to True.
        def patched_loader(*args, **kwargs):
            return loader_func(*args, **kwargs, allow_dangerous_deserialization=True)
    else:

        def patched_loader(*args, **kwargs):
            try:
                return loader_func(*args, **kwargs)
            except ValueError as e:
                if "This code relies on the pickle module" in str(e):
                    raise MlflowException(
                        "Since langchain-community 0.0.27, loading a module that relies on "
                        "the pickle deserialization requires the `allow_dangerous_deserialization` "
                        "flag to be set to True when loading. However, this flag is not supported "
                        "by the installed version of LangChain. Please upgrade LangChain to 0.1.14 "
                        "or above by running `pip install langchain>=0.1.14`.",
                        error_code=INTERNAL_ERROR,
                    ) from e
                else:
                    raise

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

    from langchain.chains.loading import load_chain

    from mlflow.langchain.retriever_chain import _RetrieverChain

    if loader_arg is not None:
        if loader_fn_path is None:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Missing file for loader_fn which is required to build the model."
            )
        loader_fn = _load_from_pickle(loader_fn_path)
        kwargs = {loader_arg: loader_fn(persist_dir)}
        if model_type == _RetrieverChain.__name__:
            model = _RetrieverChain.load(lc_model_path, **kwargs).retriever
        else:
            model = _patch_loader(load_chain)(lc_model_path, **kwargs)
    elif agent_path is None and tools_path is None:
        model = _patch_loader(load_chain)(lc_model_path)
    else:
        from langchain.agents import initialize_agent

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


@contextlib.contextmanager
def patch_langchain_type_to_cls_dict():
    """Patch LangChain's type_to_cls_dict config to handle unsupported types like ChatOpenAI.

    The type_to_cls_dict is a hard-coded dictionary in LangChain code base that defines the mapping
    between the LLM type e.g. "openai" to the loader function for the corresponding LLM class.
    However, this dictionary doesn't contain some chat models like ChatOpenAI, AzureChatOpenAI,
    which makes it unable to save and load chains with these models. Ideally, the config should
    be updated in the LangChain code base, but similar requests have been rejected multiple times
    in the past, because they consider this serde method to be deprecated, and instead prompt
    users to use their new serde method https://github.com/langchain-ai/langchain/pull/8164#issuecomment-1659723157.
    However, we can't simply migrate to the new method because it doesn't support common chains
    like RetrievalQA, AgentExecutor, etc.
    Therefore, we apply a hacky solution to patch the type_to_cls_dict from our side to support
    these models, until a better solution is provided by LangChain.
    """

    def _load_chat_openai():
        from langchain.chat_models import ChatOpenAI

        return ChatOpenAI

    def _load_azure_chat_openai():
        from langchain.chat_models import AzureChatOpenAI

        return AzureChatOpenAI

    def _patched_get_type_to_cls_dict(original):
        def _wrapped():
            return {
                **original(),
                "openai-chat": _load_chat_openai,
                "azure-openai-chat": _load_azure_chat_openai,
            }

        return _wrapped

    # NB: get_type_to_cls_dict() method is defined in the following two modules with the same
    # name but with slight different elements. This is most likely just a mistake in the
    # LangChain codebase, but we patch them separately to avoid any potential issues.
    modules_to_patch = ["langchain.llms", "langchain_community.llms.loading"]
    originals = {}
    for module_name in modules_to_patch:
        try:
            module = importlib.import_module(module_name)
            originals[module_name] = module.get_type_to_cls_dict  # Record original impl for cleanup
        except (ImportError, AttributeError):
            continue
        module.get_type_to_cls_dict = _patched_get_type_to_cls_dict(originals[module_name])

    try:
        yield
    except ValueError as e:
        if m := _CHAT_MODELS_ERROR_MSG.search(str(e)):
            model_name = "ChatOpenAI" if m.group(1) == "openai-chat" else "AzureChatOpenAI"
            raise mlflow.MlflowException(
                f"Loading {model_name} chat model is not supported in MLflow with the "
                "current version of LangChain. Please upgrade LangChain to 0.0.307 or above "
                "by running `pip install langchain>=0.0.307`."
            ) from e
        else:
            raise
    finally:
        # Clean up the patch
        for module_name, original_impl in originals.items():
            module = importlib.import_module(module_name)
            module.get_type_to_cls_dict = original_impl


def register_pydantic_serializer():
    """
    Helper function to pickle pydantic fields for pydantic v1.
    Pydantic's Cython validators are not serializable.
    https://github.com/cloudpipe/cloudpickle/issues/408
    """
    import pydantic

    if Version(pydantic.__version__) >= Version("2.0.0"):
        return

    import pydantic.fields

    def custom_serializer(obj):
        return {
            "name": obj.name,
            # outer_type_ is the original type for ModelFields,
            # while type_ can be updated later with the nested type
            # like int for List[int].
            "type_": obj.outer_type_,
            "class_validators": obj.class_validators,
            "model_config": obj.model_config,
            "default": obj.default,
            "default_factory": obj.default_factory,
            "required": obj.required,
            "final": obj.final,
            "alias": obj.alias,
            "field_info": obj.field_info,
        }

    def custom_deserializer(kwargs):
        return pydantic.fields.ModelField(**kwargs)

    def _CloudPicklerReducer(obj):
        return custom_deserializer, (custom_serializer(obj),)

    warnings.warn(
        "Using custom serializer to pickle pydantic.fields.ModelField classes, "
        "this might miss some fields and validators. To avoid this, "
        "please upgrade pydantic to v2 using `pip install pydantic -U` with "
        "langchain 0.0.267 and above."
    )
    cloudpickle.CloudPickler.dispatch[pydantic.fields.ModelField] = _CloudPicklerReducer


def unregister_pydantic_serializer():
    import pydantic

    if Version(pydantic.__version__) >= Version("2.0.0"):
        return

    cloudpickle.CloudPickler.dispatch.pop(pydantic.fields.ModelField, None)


@contextlib.contextmanager
def register_pydantic_v1_serializer_cm():
    try:
        register_pydantic_serializer()
        yield
    finally:
        unregister_pydantic_serializer()

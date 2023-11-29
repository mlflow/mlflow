"""Utility functions for mlflow.langchain."""
import functools
import json
import logging
import os
import shutil
import types
from importlib.util import find_spec
from typing import NamedTuple

import cloudpickle
from packaging import version

import mlflow
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
_MODEL_DATA_FILE_NAME = "model.yaml"
_MODEL_DATA_KEY = "model_data"
_MODEL_TYPE_KEY = "model_type"
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
    "MLflow does not guarantee support for LLMs outside of HuggingFaceHub and OpenAI, found %s"
)
_UNSUPPORTED_LANGCHAIN_VERSION_ERROR_MESSAGE = (
    "Saving {instance_type} models is only supported in langchain 0.0.194 and above."
)

logger = logging.getLogger(__name__)


def base_lc_types():
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.schema

    return (
        langchain.chains.base.Chain,
        langchain.agents.agent.AgentExecutor,
        langchain.schema.BaseRetriever,
    )


def supported_lc_types():
    return base_lc_types()


class _SpecialChainInfo(NamedTuple):
    loader_arg: str


def _get_special_chain_info_or_none(chain):
    for special_chain_class, loader_arg in _get_map_of_special_chain_class_to_loader_arg().items():
        if isinstance(chain, special_chain_class):
            return _SpecialChainInfo(loader_arg=loader_arg)


@functools.lru_cache
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


def _validate_and_wrap_lc_model(lc_model, loader_fn):
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.chains.llm
    import langchain.llms.huggingface_hub
    import langchain.llms.openai
    import langchain.schema

    if not isinstance(lc_model, supported_lc_types()):
        raise mlflow.MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(lc_model).__name__)
        )

    _SUPPORTED_LLMS = {langchain.llms.openai.OpenAI, langchain.llms.huggingface_hub.HuggingFaceHub}
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
        if isinstance(lc_model, langchain.chains.RetrievalQA) and version.parse(
            langchain.__version__
        ) < version.parse("0.0.194"):
            raise mlflow.MlflowException.invalid_parameter_value(
                _UNSUPPORTED_LANGCHAIN_VERSION_ERROR_MESSAGE.format(
                    instance_type=type(lc_model).__name__
                )
            )
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
    import langchain.agents.agent
    import langchain.chains.base
    import langchain.chains.llm

    model_data_path = os.path.join(path, _MODEL_DATA_FILE_NAME)
    model_data_kwargs = {_MODEL_DATA_KEY: _MODEL_DATA_FILE_NAME}

    if isinstance(model, langchain.chains.llm.LLMChain):
        model.save(model_data_path)
    elif isinstance(model, langchain.agents.agent.AgentExecutor):
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
    elif isinstance(model, langchain.chains.base.Chain):
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


def _get_path_by_key(root_path, key, conf):
    key_path = conf.get(key)
    return os.path.join(root_path, key_path) if key_path else None


def _load_base_lcs(
    local_model_path,
    conf,
):
    lc_model_path = os.path.join(local_model_path, conf.get(_MODEL_DATA_KEY, _MODEL_DATA_FILE_NAME))

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
            model = load_chain(lc_model_path, **kwargs)
    elif agent_path is None and tools_path is None:
        model = load_chain(lc_model_path)
    else:
        from langchain.agents import initialize_agent

        llm = load_chain(lc_model_path)
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

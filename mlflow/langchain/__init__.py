"""
The ``mlflow.langchain`` module provides an API for logging and loading LangChain models.
This module exports multivariate LangChain models in the langchain flavor and univariate
LangChain models in the pyfunc flavor:

LangChain (native) format
    This is the main flavor that can be accessed with LangChain APIs.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch inference.

.. _LangChain:
    https://python.langchain.com/en/latest/index.html
"""
import functools
import json
import logging
import os
import shutil
import types
from typing import Any, Dict, List, NamedTuple, Optional, Union

import cloudpickle
import pandas as pd
import yaml
from packaging import version

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

logger = logging.getLogger(mlflow.__name__)

FLAVOR_NAME = "langchain"
_MODEL_DATA_FILE_NAME = "model.yaml"
_MODEL_DATA_KEY = "model_data"
_AGENT_PRIMITIVES_FILE_NAME = "agent_primitive_args.json"
_AGENT_PRIMITIVES_DATA_KEY = "agent_primitive_data"
_AGENT_DATA_FILE_NAME = "agent.yaml"
_AGENT_DATA_KEY = "agent_data"
_TOOLS_DATA_FILE_NAME = "tools.pkl"
_TOOLS_DATA_KEY = "tools_data"
_MODEL_TYPE_KEY = "model_type"
_LOADER_FN_FILE_NAME = "loader_fn.pkl"
_LOADER_FN_KEY = "loader_fn"
_LOADER_ARG_KEY = "loader_arg"
_PERSIST_DIR_NAME = "persist_dir_data"
_PERSIST_DIR_KEY = "persist_dir"
_UNSUPPORTED_MODEL_ERROR_MESSAGE = (
    "MLflow langchain flavor only supports subclasses of "
    "langchain.chains.base.Chain and langchain.agents.agent.AgentExecutor instances, "
    "found {instance_type}"
)
_UNSUPPORTED_LLM_WARNING_MESSAGE = (
    "MLflow does not guarantee support for LLMs outside of HuggingFaceHub and OpenAI, found %s"
)
_UNSUPPORTED_MODEL_WARNING_MESSAGE = (
    "MLflow does not guarantee support for Chains outside of the subclasses of LLMChain, found %s"
)
_UNSUPPORTED_LANGCHAIN_VERSION_ERROR_MESSAGE = (
    "Saving {instance_type} models is only supported in langchain 0.0.194 and above."
)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("langchain")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


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
        try:
            import langchain.experimental

            class_name_to_loader_arg["langchain_experimental.sql.SQLDatabaseChain"] = "database"
        except ImportError:
            # Users may not have langchain_experimental installed, which is completely normal
            pass

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


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    lc_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    loader_fn=None,
    persist_dir=None,
):
    """
    Save a LangChain model to a path on the local file system.

    :param lc_model: A LangChain model, which could be a
                     `Chain <https://python.langchain.com/docs/modules/chains/>`_,
                     `Agent <https://python.langchain.com/docs/modules/agents/>`_, or
                     `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_.
    :param path: Local path where the serialized model (as YAML) is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `lc_model.input_keys` and `lc_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param loader_fn: A function that's required for models containing objects that aren't natively
                      serialized by LangChain.
                      This function takes a string `persist_dir` as an argument and returns the
                      specific object that the model needs. Depending on the model,
                      this could be a retriever, vectorstore, requests_wrapper, embeddings, or
                      database. For RetrievalQA Chain and retriever models, the object is a
                      (`retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_).
                      For APIChain models, it's a
                      (`requests_wrapper <https://python.langchain.com/docs/modules/agents/tools/integrations/requests>`_).
                      For HypotheticalDocumentEmbedder models, it's an
                      (`embeddings <https://python.langchain.com/docs/modules/data_connection/text_embedding/>`_).
                      For SQLDatabaseChain models, it's a
                      (`database <https://python.langchain.com/docs/modules/agents/toolkits/sql_database>`_).
    :param persist_dir: The directory where the object is stored. The `loader_fn`
                        takes this string as the argument to load the object.
                        This is optional for models containing objects that aren't natively
                        serialized by LangChain. MLflow logs the content in this directory as
                        artifacts in the subdirectory named `persist_dir_data`.

                        Here is the code snippet for logging a RetrievalQA chain with `loader_fn`
                        and `persist_dir`:

                        .. code-block:: python

                            qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())


                            def load_retriever(persist_directory):
                                embeddings = OpenAIEmbeddings()
                                vectorstore = FAISS.load_local(persist_directory, embeddings)
                                return vectorstore.as_retriever()


                            with mlflow.start_run() as run:
                                logged_model = mlflow.langchain.log_model(
                                    qa,
                                    artifact_path="retrieval_qa",
                                    loader_fn=load_retriever,
                                    persist_dir=persist_dir,
                                )

                        See a complete example in examples/langchain/retrieval_qa_chain.py.
    """
    import langchain

    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_kwargs = _save_model(lc_model, path, loader_fn, persist_dir)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.langchain",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_data_kwargs,
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: lc_model.__class__.__name__,
        **model_data_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        langchain_version=langchain.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                str(path), FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _validate_and_wrap_lc_model(lc_model, loader_fn):
    import langchain

    if not isinstance(
        lc_model,
        (
            langchain.chains.base.Chain,
            langchain.agents.agent.AgentExecutor,
            langchain.schema.BaseRetriever,
        ),
    ):
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


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    lc_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    loader_fn=None,
    persist_dir=None,
):
    """
    Log a LangChain model as an MLflow artifact for the current run.

    :param lc_model: A LangChain model, which could be a
                     `Chain <https://python.langchain.com/docs/modules/chains/>`_,
                     `Agent <https://python.langchain.com/docs/modules/agents/>`_, or
                     `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.

    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output
                      :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `lc_model.input_keys` and `lc_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred
                      <mlflow.models.infer_signature>` from datasets with valid model input
                      (e.g. the training dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training dataset),
                      for example:

                      .. code-block:: python

                        from mlflow.models import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to
                          feed the model. The given example will be converted to a
                          Pandas DataFrame and then serialized to json using the
                          Pandas split-oriented format. Bytes are base64-encoded.

    :param await_registration_for: Number of seconds to wait for the model version
                        to finish being created and is in ``READY`` status.
                        By default, the function waits for five minutes.
                        Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param loader_fn: A function that's required for models containing objects that aren't natively
                      serialized by LangChain.
                      This function takes a string `persist_dir` as an argument and returns the
                      specific object that the model needs. Depending on the model,
                      this could be a retriever, vectorstore, requests_wrapper, embeddings, or
                      database. For RetrievalQA Chain and retriever models, the object is a
                      (`retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_).
                      For APIChain models, it's a
                      (`requests_wrapper <https://python.langchain.com/docs/modules/agents/tools/integrations/requests>`_).
                      For HypotheticalDocumentEmbedder models, it's an
                      (`embeddings <https://python.langchain.com/docs/modules/data_connection/text_embedding/>`_).
                      For SQLDatabaseChain models, it's a
                      (`database <https://python.langchain.com/docs/modules/agents/toolkits/sql_database>`_).
    :param persist_dir: The directory where the object is stored. The `loader_fn`
                        takes this string as the argument to load the object.
                        This is optional for models containing objects that aren't natively
                        serialized by LangChain. MLflow logs the content in this directory as
                        artifacts in the subdirectory named `persist_dir_data`.

                        Here is the code snippet for logging a RetrievalQA chain with `loader_fn`
                        and `persist_dir`:

                        .. code-block:: python

                            qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())


                            def load_retriever(persist_directory):
                                embeddings = OpenAIEmbeddings()
                                vectorstore = FAISS.load_local(persist_directory, embeddings)
                                return vectorstore.as_retriever()


                            with mlflow.start_run() as run:
                                logged_model = mlflow.langchain.log_model(
                                    qa,
                                    artifact_path="retrieval_qa",
                                    loader_fn=load_retriever,
                                    persist_dir=persist_dir,
                                )

                        See a complete example in examples/langchain/retrieval_qa_chain.py.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

    # infer signature if signature is not provided
    if signature is None:
        input_columns = [
            ColSpec(type=DataType.string, name=input_key) for input_key in lc_model.input_keys
        ]
        input_schema = Schema(input_columns)
        output_columns = [
            ColSpec(type=DataType.string, name=output_key) for output_key in lc_model.output_keys
        ]
        output_schema = Schema(output_columns)
        signature = ModelSignature(input_schema, output_schema)

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.langchain,
        registered_model_name=registered_model_name,
        lc_model=lc_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        loader_fn=loader_fn,
        persist_dir=persist_dir,
    )


def _save_model(model, path, loader_fn, persist_dir):
    import langchain

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
            with open(tools_data_path, "wb") as f:
                cloudpickle.dump(model.tools, f)
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


def _load_from_pickle(loader_fn_path, persist_dir):
    with open(loader_fn_path, "rb") as f:
        loader_fn = cloudpickle.load(f)
    return loader_fn(persist_dir)


def _load_model(
    path,
    model_type,
    loader_arg=None,
    agent_path=None,
    tools_path=None,
    agent_primitive_path=None,
    loader_fn_path=None,
    persist_dir=None,
):
    from langchain.chains.loading import load_chain

    from mlflow.langchain.retriever_chain import _RetrieverChain

    model = None
    if loader_arg is not None:
        if loader_fn_path is None:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Missing file for loader_fn which is required to build the model."
            )
        kwargs = {loader_arg: _load_from_pickle(loader_fn_path, persist_dir)}
        if model_type == _RetrieverChain.__name__:
            model = _RetrieverChain.load(path, **kwargs).retriever
        else:
            model = load_chain(path, **kwargs)
    elif agent_path is None and tools_path is None:
        model = load_chain(path)
    else:
        from langchain.agents import initialize_agent

        llm = load_chain(path)
        tools = []
        kwargs = {}

        if os.path.exists(tools_path):
            with open(tools_path, "rb") as f:
                tools = cloudpickle.load(f)
        else:
            raise mlflow.MlflowException(
                "Missing file for tools which is required to build the AgentExecutor object."
            )

        if os.path.exists(agent_primitive_path):
            with open(agent_primitive_path) as config_file:
                kwargs = json.load(config_file)

        model = initialize_agent(tools=tools, llm=llm, agent_path=agent_path, **kwargs)
    return model


class _LangChainModelWrapper:
    def __init__(self, lc_model):
        self.lc_model = lc_model

    def predict(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]]],
        params: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> List[str]:
        """
        :param data: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        if isinstance(data, pd.DataFrame):
            messages = data.to_dict(orient="records")
        elif isinstance(data, list) and (
            all(isinstance(d, str) for d in data) or all(isinstance(d, dict) for d in data)
        ):
            messages = data
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Input must be a pandas DataFrame or a list of strings or a list of dictionaries",
            )
        return process_api_requests(lc_model=self.lc_model, requests=messages)


class _TestLangChainWrapper(_LangChainModelWrapper):
    """
    A wrapper class that should be used for testing purposes only.
    """

    def predict(
        self, data, params: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ):
        """
        :param data: Model input data.
        :param params: Additional parameters to pass to the model for inference.

                       .. Note:: Experimental: This parameter may change or be removed in a future
                                               release without warning.

        :return: Model predictions.
        """
        import langchain

        from mlflow.openai.utils import TEST_CONTENT

        from tests.langchain.test_langchain_model_export import _mock_async_request

        if isinstance(
            self.lc_model,
            (
                langchain.chains.llm.LLMChain,
                langchain.chains.RetrievalQA,
                langchain.schema.retriever.BaseRetriever,
            ),
        ):
            mockContent = TEST_CONTENT
        elif isinstance(self.lc_model, langchain.agents.agent.AgentExecutor):
            mockContent = f"Final Answer: {TEST_CONTENT}"

        with _mock_async_request(mockContent):
            return super().predict(data)


def _load_pyfunc(path):
    """
    Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``langchain`` flavor.
    """
    wrapper_cls = _TestLangChainWrapper if _MLFLOW_TESTING.get() else _LangChainModelWrapper
    return wrapper_cls(_load_model_from_local_fs(path))


def _load_model_from_local_fs(local_model_path):
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    lc_model_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_DATA_KEY, _MODEL_DATA_FILE_NAME)
    )

    agent_model_path = tools_model_path = agent_primitive_path = loader_fn_path = persist_dir = None
    if agent_path := flavor_conf.get(_AGENT_DATA_KEY):
        agent_model_path = os.path.join(local_model_path, agent_path)

    if tools_path := flavor_conf.get(_TOOLS_DATA_KEY):
        tools_model_path = os.path.join(local_model_path, tools_path)

    if primitive_path := flavor_conf.get(_AGENT_PRIMITIVES_DATA_KEY):
        agent_primitive_path = os.path.join(local_model_path, primitive_path)

    if loader_fn_file_name := flavor_conf.get(_LOADER_FN_KEY):
        loader_fn_path = os.path.join(local_model_path, loader_fn_file_name)

    if persist_dir_name := flavor_conf.get(_PERSIST_DIR_KEY):
        persist_dir = os.path.join(local_model_path, persist_dir_name)

    model_type = flavor_conf.get(_MODEL_TYPE_KEY)
    loader_arg = flavor_conf.get(_LOADER_ARG_KEY)

    return _load_model(
        lc_model_path,
        model_type,
        loader_arg,
        agent_model_path,
        tools_model_path,
        agent_primitive_path,
        loader_fn_path,
        persist_dir,
    )


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load a LangChain model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A LangChain model instance
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    return _load_model_from_local_fs(local_model_path)

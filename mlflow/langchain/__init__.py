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

import contextlib
import functools
import importlib.util
import logging
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.langchain._langchain_autolog import (
    _update_langchain_model_config,
    patched_inference,
)
from mlflow.langchain._rag_utils import _CODE_CONFIG, _CODE_PATH, _set_config_path
from mlflow.langchain.databricks_dependencies import (
    _DATABRICKS_DEPENDENCY_KEY,
    _detect_databricks_dependencies,
)
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
    _BASE_LOAD_KEY,
    _MODEL_LOAD_KEY,
    _RUNNABLE_LOAD_KEY,
    _get_temp_file_with_content,
    _load_base_lcs,
    _save_base_lcs,
    _validate_and_wrap_lc_model,
    lc_runnables_types,
    patch_langchain_type_to_cls_dict,
    register_pydantic_v1_serializer_cm,
)
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.model_config import _set_model_config
from mlflow.models.resources import _ResourceBuilder
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _convert_llm_input_data, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    autologging_is_disabled,
    safe_patch,
)
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
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    FLAVOR_CONFIG_CODE,
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_copy_model_code_and_config_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

logger = logging.getLogger(mlflow.__name__)

FLAVOR_NAME = "langchain"
_MODEL_TYPE_KEY = "model_type"
_MODEL_CODE_CONFIG = "model_config"
_MODEL_CODE_PATH = "model_code_path"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    # pin pydantic and cloudpickle version as they are used in langchain
    # model saving and loading
    return list(map(_get_pinned_requirement, ["langchain", "pydantic", "cloudpickle"]))


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


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
    example_no_conversion=False,
    model_config=None,
):
    """
    Save a LangChain model to a path on the local file system.

    Args:
        lc_model: A LangChain model, which could be a
            `Chain <https://python.langchain.com/docs/modules/chains/>`_,
            `Agent <https://python.langchain.com/docs/modules/agents/>`_,
            `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_,
            or `RunnableSequence <https://python.langchain.com/docs/modules/chains/foundational/sequential_chains#using-lcel>`_.
        path: Local path where the serialized model (as YAML) is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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

        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        loader_fn: A function that's required for models containing objects that aren't natively
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
        persist_dir: The directory where the object is stored. The `loader_fn`
            takes this string as the argument to load the object.
            This is optional for models containing objects that aren't natively
            serialized by LangChain. MLflow logs the content in this directory as
            artifacts in the subdirectory named `persist_dir_data`.

            Here is the code snippet for logging a RetrievalQA chain with `loader_fn`
            and `persist_dir`:

            .. Note:: In langchain_community >= 0.0.27, loading pickled data requires providing the
                ``allow_dangerous_deserialization`` argument.

            .. code-block:: python

                qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())


                def load_retriever(persist_directory):
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.load_local(
                        persist_directory,
                        embeddings,
                        # you may need to add the line below
                        # for langchain_community >= 0.0.27
                        allow_dangerous_deserialization=True,
                    )
                    return vectorstore.as_retriever()


                with mlflow.start_run() as run:
                    logged_model = mlflow.langchain.log_model(
                        qa,
                        artifact_path="retrieval_qa",
                        loader_fn=load_retriever,
                        persist_dir=persist_dir,
                    )

            See a complete example in examples/langchain/retrieval_qa_chain.py.
        example_no_conversion: {{ example_no_conversion }}
        model_config: The model configuration to apply to the model if saving model as code. This
            configuration is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
    """
    import langchain
    from langchain.schema import BaseRetriever

    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    model_config_path = None
    model_code_path = None
    if isinstance(lc_model, str):
        # The LangChain model is defined as Python code located in the file at the path
        # specified by `lc_model`. Verify that the path exists and, if so, copy it to the
        # model directory along with any other specified code modules

        if os.path.exists(lc_model):
            model_code_path = lc_model
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"If the provided model '{lc_model}' is a string, it must be a valid python "
                "file path or a databricks notebook file path containing the code for defining "
                "the chain instance."
            )

        if isinstance(model_config, dict):
            model_config_path = _get_temp_file_with_content(
                "config.yml", yaml.dump(model_config), "w"
            )
        elif isinstance(model_config, str):
            if os.path.exists(model_config):
                model_config_path = model_config
            else:
                raise mlflow.MlflowException.invalid_parameter_value(
                    f"Model config path '{model_config}' provided is not a valid file path. "
                    "Please provide a valid model configuration."
                )
        elif not model_config:
            # If the model_config is not provided we fallback to getting the config path
            # from code_paths so that is backwards compatible.
            if code_paths and len(code_paths) == 1 and os.path.exists(code_paths[0]):
                model_config_path = code_paths[0]

        lc_model = (
            _load_model_code_path(model_code_path, model_config_path)
            if model_config_path
            else _load_model_code_path(model_code_path)
        )
        _validate_and_copy_model_code_and_config_paths(model_code_path, model_config_path, path)

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None:
        if input_example is not None:
            wrapped_model = _LangChainModelWrapper(lc_model)
            signature = _infer_signature_from_input_example(input_example, wrapped_model)
        else:
            if hasattr(lc_model, "input_keys"):
                input_columns = [
                    ColSpec(type=DataType.string, name=input_key)
                    for input_key in lc_model.input_keys
                ]
                input_schema = Schema(input_columns)
            else:
                input_schema = None
            if (
                hasattr(lc_model, "output_keys")
                and len(lc_model.output_keys) == 1
                and not isinstance(lc_model, BaseRetriever)
            ):
                output_columns = [
                    ColSpec(type=DataType.string, name=output_key)
                    for output_key in lc_model.output_keys
                ]
                output_schema = Schema(output_columns)
            else:
                # TODO: empty output schema if multiple output_keys or is a retriever. fix later!
                # https://databricks.atlassian.net/browse/ML-34706
                output_schema = None

            signature = (
                ModelSignature(input_schema, output_schema)
                if input_schema or output_schema
                else None
            )

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature

    if input_example is not None:
        _save_example(mlflow_model, input_example, path, example_no_conversion)
    if metadata is not None:
        mlflow_model.metadata = metadata

    streamable = isinstance(lc_model, lc_runnables_types())

    if not isinstance(model_code_path, str):
        model_data_kwargs = _save_model(lc_model, path, loader_fn, persist_dir)
        flavor_conf = {
            _MODEL_TYPE_KEY: lc_model.__class__.__name__,
            **model_data_kwargs,
        }
    else:
        # If the model is a string, we expect the code_path which is ideally config.yml
        # would be used in the model. We set the code_path here so it can be set
        # globally when the model is loaded with the local path. So the consumer
        # can use that path instead of the config.yml path when the model is loaded
        # TODO: what if model_config is not a string / file path?
        flavor_conf = (
            {_MODEL_CODE_CONFIG: model_config_path, _MODEL_CODE_PATH: model_code_path}
            if model_config_path
            else {_MODEL_CODE_CONFIG: None, _MODEL_CODE_PATH: model_code_path}
        )
        model_data_kwargs = {}

    # TODO: Pass model_config to pyfunc
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.langchain",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        predict_stream_fn="predict_stream",
        streamable=streamable,
        model_code_path=model_code_path,
        **model_data_kwargs,
    )

    if Version(langchain.__version__) >= Version("0.0.311"):
        (databricks_dependency, databricks_resources) = _detect_databricks_dependencies(lc_model)
        if databricks_dependency:
            flavor_conf[_DATABRICKS_DEPENDENCY_KEY] = databricks_dependency
        if databricks_resources:
            serialized_databricks_resources = _ResourceBuilder.from_resources(databricks_resources)
            mlflow_model.resources = serialized_databricks_resources

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        langchain_version=langchain.__version__,
        code=code_dir_subpath,
        streamable=streamable,
        **flavor_conf,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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
    example_no_conversion=False,
    run_id=None,
    model_config=None,
):
    """
    Log a LangChain model as an MLflow artifact for the current run.

    Args:
        lc_model: A LangChain model, which could be a
            `Chain <https://python.langchain.com/docs/modules/chains/>`_,
            `Agent <https://python.langchain.com/docs/modules/agents/>`_, or
            `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        loader_fn: A function that's required for models containing objects that aren't natively
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
        persist_dir: The directory where the object is stored. The `loader_fn`
            takes this string as the argument to load the object.
            This is optional for models containing objects that aren't natively
            serialized by LangChain. MLflow logs the content in this directory as
            artifacts in the subdirectory named `persist_dir_data`.

            Here is the code snippet for logging a RetrievalQA chain with `loader_fn`
            and `persist_dir`:

            .. Note:: In langchain_community >= 0.0.27, loading pickled data requires providing the
                ``allow_dangerous_deserialization`` argument.

            .. code-block:: python

                qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=db.as_retriever())


                def load_retriever(persist_directory):
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.load_local(
                        persist_directory,
                        embeddings,
                        # you may need to add the line below
                        # for langchain_community >= 0.0.27
                        allow_dangerous_deserialization=True,
                    )
                    return vectorstore.as_retriever()


                with mlflow.start_run() as run:
                    logged_model = mlflow.langchain.log_model(
                        qa,
                        artifact_path="retrieval_qa",
                        loader_fn=load_retriever,
                        persist_dir=persist_dir,
                    )

            See a complete example in examples/langchain/retrieval_qa_chain.py.
        example_no_conversion: {{ example_no_conversion }}
        run_id: run_id to associate with this model version. If specified, we resume the
                run and log the model to that run. Otherwise, a new run is created.
                Default to None.
        model_config: The model configuration to apply to the model if saving model as code. This
            configuration is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

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
        example_no_conversion=example_no_conversion,
        run_id=run_id,
        model_config=model_config,
    )


def _save_model(model, path, loader_fn, persist_dir):
    if Version(cloudpickle.__version__) < Version("2.1.0"):
        warnings.warn(
            "If you are constructing a custom LangChain model, "
            "please upgrade cloudpickle to version 2.1.0 or later "
            "using `pip install cloudpickle>=2.1.0` "
            "to ensure the model can be loaded correctly."
        )
    # patch_langchain_type_to_cls_dict here as we attempt to load model
    # if it's saved by `dict` method
    with register_pydantic_v1_serializer_cm(), patch_langchain_type_to_cls_dict():
        if isinstance(model, lc_runnables_types()):
            return _save_runnables(model, path, loader_fn=loader_fn, persist_dir=persist_dir)
        else:
            return _save_base_lcs(model, path, loader_fn, persist_dir)


def _load_model(local_model_path, flavor_conf):
    # model_type is not accurate as the class can be subclass
    # of supported types, we define _MODEL_LOAD_KEY to ensure
    # which load function to use
    model_load_fn = flavor_conf.get(_MODEL_LOAD_KEY)
    with register_pydantic_v1_serializer_cm():
        if model_load_fn == _RUNNABLE_LOAD_KEY:
            model = _load_runnables(local_model_path, flavor_conf)
        elif model_load_fn == _BASE_LOAD_KEY:
            model = _load_base_lcs(local_model_path, flavor_conf)
        else:
            raise mlflow.MlflowException(
                "Failed to load LangChain model. Unknown model type: "
                f"{flavor_conf.get(_MODEL_TYPE_KEY)}"
            )
    # To avoid double logging, we set model_logged to True
    # when the model is loaded
    if not autologging_is_disabled(FLAVOR_NAME):
        if _update_langchain_model_config(model):
            model.model_logged = True
            model.run_id = get_model_info(local_model_path).run_id
    return model


class _LangChainModelWrapper:
    def __init__(self, lc_model):
        self.lc_model = lc_model

    def predict(
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        messages, return_first_element = self._prepare_predict_messages(data)
        results = process_api_requests(lc_model=self.lc_model, requests=messages)
        return results[0] if return_first_element else results

    @experimental
    def _predict_with_callbacks(
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any],
        params: Optional[Dict[str, Any]] = None,
        callback_handlers=None,
        convert_chat_responses=False,
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.
            callback_handlers: Callback handlers to pass to LangChain.
            convert_chat_responses: If true, forcibly convert response to chat model
                response format.

        Returns:
            Model predictions.
        """
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        messages, return_first_element = self._prepare_predict_messages(data)
        results = process_api_requests(
            lc_model=self.lc_model,
            requests=messages,
            callback_handlers=callback_handlers,
            convert_chat_responses=convert_chat_responses,
        )
        return results[0] if return_first_element else results

    def _prepare_predict_messages(self, data):
        """
        Return a tuple of (preprocessed_data, return_first_element)
        `preprocessed_data` is always a list,
        and `return_first_element` means if True, we should return the first element
        of inference result, otherwise we should return the whole inference result.
        """
        data = _convert_llm_input_data(data)

        if not isinstance(data, list):
            # if the input data is not a list (i.e. single input),
            # we still need to convert it to a one-element list `[data]`
            # because `process_api_requests` only accepts list as valid input.
            # and in this case,
            # we should return the first element of the inference result
            # because we change input `data` to `[data]`
            return [data], True
        if isinstance(data, list):
            return data, False
        raise mlflow.MlflowException.invalid_parameter_value(
            "Input must be a pandas DataFrame or a list "
            f"for model {self.lc_model.__class__.__name__}"
        )

    def _prepare_predict_stream_messages(self, data):
        data = _convert_llm_input_data(data)

        if isinstance(data, list):
            # `predict_stream` only accepts single input.
            # but `enforce_schema` might convert single input into a list like `[single_input]`
            # so extract the first element in the list.
            if len(data) != 1:
                raise MlflowException(
                    f"'predict_stream' requires single input, but it got input data {data}"
                )
            return data[0]
        return data

    def predict_stream(
        self,
        data: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data, only single input is allowed.
            params: Additional parameters to pass to the model for inference.

        Returns:
            An iterator of model prediction chunks.
        """
        from mlflow.langchain.api_request_parallel_processor import (
            process_stream_request,
        )

        data = self._prepare_predict_stream_messages(data)
        return process_stream_request(
            lc_model=self.lc_model,
            request_json=data,
        )

    def _predict_stream_with_callbacks(
        self,
        data: Any,
        params: Optional[Dict[str, Any]] = None,
        callback_handlers=None,
        convert_chat_responses=False,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Args:
            data: Model input data, only single input is allowed.
            params: Additional parameters to pass to the model for inference.
            callback_handlers: Callback handlers to pass to LangChain.
            convert_chat_responses: If true, forcibly convert response to chat model
                response format.

        Returns:
            An iterator of model prediction chunks.
        """
        from mlflow.langchain.api_request_parallel_processor import (
            process_stream_request,
        )

        data = self._prepare_predict_stream_messages(data)
        return process_stream_request(
            lc_model=self.lc_model,
            request_json=data,
            callback_handlers=callback_handlers,
            convert_chat_responses=convert_chat_responses,
        )


class _TestLangChainWrapper(_LangChainModelWrapper):
    """
    A wrapper class that should be used for testing purposes only.
    """

    def predict(
        self,
        data,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Model input data and additional parameters.

        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        import langchain
        from langchain.schema.retriever import BaseRetriever

        from mlflow.utils.openai_utils import (
            TEST_CONTENT,
            TEST_INTERMEDIATE_STEPS,
            TEST_SOURCE_DOCUMENTS,
        )

        from tests.langchain.test_langchain_model_export import _mock_async_request

        if isinstance(
            self.lc_model,
            (
                langchain.chains.llm.LLMChain,
                langchain.chains.RetrievalQA,
                BaseRetriever,
            ),
        ):
            mockContent = TEST_CONTENT
        elif isinstance(self.lc_model, langchain.agents.agent.AgentExecutor):
            mockContent = f"Final Answer: {TEST_CONTENT}"
        else:
            mockContent = TEST_CONTENT

        with _mock_async_request(mockContent):
            result = super().predict(data)
        if (
            hasattr(self.lc_model, "return_source_documents")
            and self.lc_model.return_source_documents
        ):
            for res in result:
                res["source_documents"] = TEST_SOURCE_DOCUMENTS
        if (
            hasattr(self.lc_model, "return_intermediate_steps")
            and self.lc_model.return_intermediate_steps
        ):
            for res in result:
                res["intermediate_steps"] = TEST_INTERMEDIATE_STEPS

        return result


def _load_pyfunc(path):
    """Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``langchain`` flavor.
    """
    wrapper_cls = _TestLangChainWrapper if _MLFLOW_TESTING.get() else _LangChainModelWrapper
    return wrapper_cls(_load_model_from_local_fs(path))


def _load_model_from_local_fs(local_model_path):
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    if _MODEL_CODE_PATH in flavor_conf:
        flavor_config_path = flavor_conf.get(_MODEL_CODE_CONFIG, None)
        if flavor_config_path is not None:
            config_path = os.path.join(
                local_model_path,
                os.path.basename(flavor_config_path),
            )
        else:
            config_path = None

        flavor_code_path = flavor_conf.get(_MODEL_CODE_PATH)
        code_path = os.path.join(
            local_model_path,
            os.path.basename(flavor_code_path),
        )

        return _load_model_code_path(code_path, config_path)
    # Code for backwards compatibility, relies on RAG utils - remove in the future
    elif _CODE_CONFIG in flavor_conf:
        path = flavor_conf.get(_CODE_CONFIG)
        flavor_code_config = flavor_conf.get(FLAVOR_CONFIG_CODE)
        if path is not None:
            config_path = os.path.join(
                local_model_path,
                flavor_code_config,
                os.path.basename(path),
            )
        else:
            config_path = None

        flavor_code_path = flavor_conf.get(_CODE_PATH, "chain.py")
        code_path = os.path.join(
            local_model_path,
            flavor_code_config,
            os.path.basename(flavor_code_path),
        )

        return _load_model_code_path(code_path, config_path)
    else:
        _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
        with patch_langchain_type_to_cls_dict():
            return _load_model(local_model_path, flavor_conf)


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load a LangChain model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A LangChain model instance.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    return _load_model_from_local_fs(local_model_path)


@contextmanager
def _config_path_context(config_path: Optional[str] = None):
    # Check if config_path is None and set it to "" so when loading the model
    # the config_path is set to "" so the ModelConfig can correctly check if the
    # config is set or not
    if config_path is None:
        config_path = ""

    _set_model_config(config_path)
    # set rag utils global for backwards compatibility
    _set_config_path(config_path)
    try:
        yield
    finally:
        _set_model_config(None)
        # unset rag utils global for backwards compatibility
        _set_config_path(None)


# In the Python's module caching mechanism, which by default, prevents the
# re-importation of previously loaded modules. This is particularly
# problematic in contexts where it's necessary to reload a module (in this case,
# the `model code path` module) multiple times within the same Python
# runtime environment.
# The issue at hand arises from the desire to import the `model code path` module
# multiple times during a single runtime session. Normally, once a module is
# imported, it's added to `sys.modules`, and subsequent import attempts retrieve
# the cached module rather than re-importing it.
# To address this, the function dynamically imports the `model code path` module
# under unique, dynamically generated module names. This is achieved by creating
# a unique name for each import using a combination of the original module name
# and a randomly generated UUID. This approach effectively bypasses the caching
# mechanism, as each import is considered as a separate module by the Python interpreter.
def _load_model_code_path(code_path: str, config_path: Optional[str] = None):
    with _config_path_context(config_path):
        try:
            new_module_name = f"code_model_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(new_module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[new_module_name] = module
            spec.loader.exec_module(module)
        except ImportError as e:
            raise mlflow.MlflowException("Failed to import LangChain model.") from e

    return (
        mlflow.models.model.__mlflow_model__ or mlflow.langchain._rag_utils.__databricks_rag_chain__
    )


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    log_datasets=False,
    log_inputs_outputs=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=True,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
):
    """
    Enables (or disables) and configures autologging from Langchain to MLflow.

    Args:
        log_input_examples: If ``True``, input examples from inference data are collected and
            logged along with Langchain model artifacts during inference. If
            ``False``, input examples are not logged.
            Note: Input examples are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with Langchain model artifacts during inference. If ``False``,
            signatures are not logged.
            Note: Model signatures are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_models: If ``True``, langchain models are logged as MLflow model artifacts.
            If ``False``, langchain models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, dataset information is logged to MLflow Tracking
            if applicable. If ``False``, dataset information is not logged.
        log_inputs_outputs: If ``True``, inference data and results are combined into a single
            pandas DataFrame and logged to MLflow Tracking as an artifact.
            If ``False``, inference data and results are not logged.
            Default to ``True``.
        disable: If ``True``, disables the Langchain autologging integration. If ``False``,
            enables the Langchain autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            langchain that have not been tested against this version of the MLflow
            client or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Langchain
            autologging. If ``False``, show all events and warnings during Langchain
            autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """

    with contextlib.suppress(ImportError):
        from langchain.agents.agent import AgentExecutor
        from langchain.chains.base import Chain
        from langchain.schema import BaseRetriever

        classes = lc_runnables_types() + (AgentExecutor, Chain)
        for cls in classes:
            # If runnable also contains loader_fn and persist_dir, warn
            # BaseRetrievalQA, BaseRetriever, ...
            safe_patch(
                FLAVOR_NAME,
                cls,
                "invoke",
                functools.partial(patched_inference, "invoke"),
            )

            safe_patch(
                FLAVOR_NAME,
                cls,
                "batch",
                functools.partial(patched_inference, "batch"),
            )

            safe_patch(
                FLAVOR_NAME,
                cls,
                "stream",
                functools.partial(patched_inference, "stream"),
            )

        for cls in [AgentExecutor, Chain]:
            safe_patch(
                FLAVOR_NAME,
                cls,
                "__call__",
                functools.partial(patched_inference, "__call__"),
            )

        safe_patch(
            FLAVOR_NAME,
            BaseRetriever,
            "get_relevant_documents",
            functools.partial(patched_inference, "get_relevant_documents"),
        )

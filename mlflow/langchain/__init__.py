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
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.langchain._langchain_autolog import (
    _update_langchain_model_config,
    patched_inference,
)
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
    _BASE_LOAD_KEY,
    _MODEL_LOAD_KEY,
    _RUNNABLE_LOAD_KEY,
    _load_base_lcs,
    _save_base_lcs,
    _validate_and_wrap_lc_model,
    lc_runnables_types,
)
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
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
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

logger = logging.getLogger(mlflow.__name__)

FLAVOR_NAME = "langchain"
_MODEL_TYPE_KEY = "model_type"


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("langchain")]


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
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
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
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
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
        example_no_conversion: {{ example_no_conversion }}
    """
    import langchain
    from langchain.schema import BaseRetriever

    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
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
):
    """
    Log a LangChain model as an MLflow artifact for the current run.

    Args:
        lc_model: A LangChain model, which could be a
            `Chain <https://python.langchain.com/docs/modules/chains/>`_,
            `Agent <https://python.langchain.com/docs/modules/agents/>`, or
            `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
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
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
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
        example_no_conversion: {{ example_no_conversion }}
        run_id: run_id to associate with this model version. If specified, we resume the
                run and log the model to that run. Otherwise, a new run is created.
                Default to None.

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
    )


def _save_model(model, path, loader_fn, persist_dir):
    if isinstance(model, lc_runnables_types()):
        return _save_runnables(model, path, loader_fn=loader_fn, persist_dir=persist_dir)
    else:
        return _save_base_lcs(model, path, loader_fn, persist_dir)


def _load_model(local_model_path, flavor_conf):
    # model_type is not accurate as the class can be subclass
    # of supported types, we define _MODEL_LOAD_KEY to ensure
    # which load function to use
    model_load_fn = flavor_conf.get(_MODEL_LOAD_KEY)
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

    def predict(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any],
        params: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> List[str]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        messages = self._prepare_messages(data)
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        return_first_element = isinstance(self.lc_model, lc_runnables_types()) and not isinstance(
            data, pd.DataFrame
        )
        results = process_api_requests(lc_model=self.lc_model, requests=messages)
        return results[0] if return_first_element else results

    @experimental
    def _predict_with_callbacks(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any],
        params: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        callback_handlers=None,
        convert_chat_responses=False,
    ) -> List[str]:
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.
            data: Callback handlers to pass to LangChain.

        Returns:
            Model predictions.
        """
        messages = self._prepare_messages(data)
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        return_first_element = isinstance(self.lc_model, lc_runnables_types()) and not isinstance(
            data, pd.DataFrame
        )
        results = process_api_requests(
            lc_model=self.lc_model,
            requests=messages,
            callback_handlers=callback_handlers,
            convert_chat_responses=convert_chat_responses,
        )
        return results[0] if return_first_element else results

    def _prepare_messages(self, data):
        # numpy array is not json serializable, so we convert it to list
        # then send it to the model
        def _convert_ndarray_to_list(data):
            import numpy as np

            if isinstance(data, np.ndarray):
                return data.tolist()
            if isinstance(data, list):
                return [_convert_ndarray_to_list(d) for d in data]
            if isinstance(data, dict):
                return {k: _convert_ndarray_to_list(v) for k, v in data.items()}
            return data

        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")

        data = _convert_ndarray_to_list(data)
        if isinstance(self.lc_model, lc_runnables_types()):
            return [data]
        if isinstance(data, list) and (
            all(isinstance(d, str) for d in data) or all(isinstance(d, dict) for d in data)
        ):
            return data
        raise mlflow.MlflowException.invalid_parameter_value(
            "Input must be a pandas DataFrame or a list of strings "
            "or a list of dictionaries "
            f"for model {self.lc_model.__class__.__name__}"
        )


class _TestLangChainWrapper(_LangChainModelWrapper):
    """
    A wrapper class that should be used for testing purposes only.
    """

    def predict(
        self, data, params: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ):
        """
        Model input data and additional parameters.

        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

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
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
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
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from Langchain to MLflow.

    :param log_input_examples: If ``True``, input examples from inference data are collected and
                               logged along with Langchain model artifacts during inference. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with Langchain model artifacts during inference. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, langchain models are logged as MLflow model artifacts.
                       If ``False``, langchain models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param log_datasets: If ``True``, dataset information is logged to MLflow Tracking
                         if applicable. If ``False``, dataset information is not logged.
    :param log_inputs_outputs: If ``True``, inference data and results are combined into a single
                                  pandas DataFrame and logged to MLflow Tracking as an artifact.
                                  If ``False``, inference data and results are not logged.
                                  Default to ``True``.
    :param disable: If ``True``, disables the Langchain autologging integration. If ``False``,
                    enables the Langchain autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      langchain that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Langchain
                   autologging. If ``False``, show all events and warnings during Langchain
                   autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    :param extra_tags: A dictionary of extra tags to set on each managed run created by autologging.
    """

    with contextlib.suppress(ImportError):
        from langchain.agents.agent import AgentExecutor
        from langchain.chains.base import Chain
        from langchain.schema import BaseRetriever

        classes = lc_runnables_types() + (AgentExecutor, Chain)
        for cls in classes:
            # If runnable also contains loader_fn and persist_dir, warn
            # BaseRetrievalQA, BaseRetriever, ...
            safe_patch(FLAVOR_NAME, cls, "invoke", functools.partial(patched_inference, "invoke"))

        for cls in [AgentExecutor, Chain]:
            safe_patch(
                FLAVOR_NAME, cls, "__call__", functools.partial(patched_inference, "__call__")
            )

        safe_patch(
            FLAVOR_NAME,
            BaseRetriever,
            "get_relevant_documents",
            functools.partial(patched_inference, "get_relevant_documents"),
        )

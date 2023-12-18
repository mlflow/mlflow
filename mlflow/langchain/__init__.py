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
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
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
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
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
                     `Agent <https://python.langchain.com/docs/modules/agents/>`_,
                     `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_,
                     or `RunnableSequence <https://python.langchain.com/docs/modules/chains/foundational/sequential_chains#using-lcel>`_.
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

    :param input_example: {{ input_example }}
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

    :param input_example: {{ input_example }}

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
    from langchain.schema import BaseRetriever

    lc_model = _validate_and_wrap_lc_model(lc_model, loader_fn)

    # infer signature if signature is not provided
    if signature is None:
        if hasattr(lc_model, "input_keys") and hasattr(lc_model, "output_keys"):
            input_columns = [
                ColSpec(type=DataType.string, name=input_key) for input_key in lc_model.input_keys
            ]
            input_schema = Schema(input_columns)

            output_columns = [
                ColSpec(type=DataType.string, name=output_key)
                for output_key in lc_model.output_keys
            ]
            output_schema = Schema(output_columns)

            # TODO: empty output schema if multiple output_keys or is a retriever. fix later!
            # https://databricks.atlassian.net/browse/ML-34706
            if len(lc_model.output_keys) > 1 or isinstance(lc_model, BaseRetriever):
                output_schema = None

            signature = ModelSignature(input_schema, output_schema)
        # TODO: support signature for other runnables

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
        return _load_runnables(local_model_path, flavor_conf)
    if model_load_fn == _BASE_LOAD_KEY:
        return _load_base_lcs(local_model_path, flavor_conf)
    raise mlflow.MlflowException(
        f"Failed to load LangChain model. Unknown model type: {flavor_conf.get(_MODEL_TYPE_KEY)}"
    )


class _LangChainModelWrapper:
    def __init__(self, lc_model):
        self.lc_model = lc_model

    def predict(  # pylint: disable=unused-argument
        self,
        data: Union[pd.DataFrame, List[Union[str, Dict[str, Any]]], Any],
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
        elif isinstance(self.lc_model, lc_runnables_types()):
            messages = [data]
            return process_api_requests(lc_model=self.lc_model, requests=messages)[0]
        elif isinstance(data, list) and (
            all(isinstance(d, str) for d in data) or all(isinstance(d, dict) for d in data)
        ):
            messages = data
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Input must be a pandas DataFrame or a list of strings or a list of dictionaries "
                f"for model {self.lc_model.__class__.__name__}"
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
    """
    Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``langchain`` flavor.
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

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
import inspect
import json
import logging
import os
import tempfile
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.langchain.databricks_dependencies import _detect_databricks_dependencies
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
    _BASE_LOAD_KEY,
    _MODEL_LOAD_KEY,
    _RUNNABLE_LOAD_KEY,
    _load_base_lcs,
    _save_base_lcs,
    _validate_and_prepare_lc_model_or_path,
    lc_runnables_types,
    patch_langchain_type_to_cls_dict,
    register_pydantic_v1_serializer_cm,
)
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.dependencies_schemas import (
    _clear_dependencies_schemas,
    _get_dependencies_schemas,
)
from mlflow.models.model import MLMODEL_FILE_NAME, MODEL_CODE_PATH, MODEL_CONFIG
from mlflow.models.resources import _ResourceBuilder
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import (
    _convert_llm_input_data,
    _load_model_code_path,
    _save_example,
)
from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
from mlflow.pyfunc.context import get_prediction_context
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    autologging_is_disabled,
    safe_patch,
)
from mlflow.utils.databricks_utils import (
    is_in_databricks_model_serving_environment,
    is_mlflow_tracing_enabled_in_model_serving,
)
from mlflow.utils.docstring_utils import (
    LOG_MODEL_PARAM_DOCS,
    docstring_version_compatibility_warning,
    format_docstring,
)
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
    _validate_and_copy_file_to_directory,
    _validate_and_get_model_config_from_file,
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
@docstring_version_compatibility_warning(FLAVOR_NAME)
@trace_disabled  # Suppress traces for internal predict calls while saving model
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
    example_no_conversion=None,
    model_config=None,
    streamable: Optional[bool] = None,
):
    """
    Save a LangChain model to a path on the local file system.

    Args:
        lc_model: A LangChain model, which could be a
            `Chain <https://python.langchain.com/docs/modules/chains/>`_,
            `Agent <https://python.langchain.com/docs/modules/agents/>`_,
            `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_,
            or `RunnableSequence <https://python.langchain.com/docs/modules/chains/foundational/sequential_chains#using-lcel>`_,
            or a path containing the `LangChain model code <https://github.com/mlflow/mlflow/blob/master/examples/langchain/chain_as_code_driver.py>`
            for the above types. When using model as path, make sure to set the model
            by using :func:`mlflow.models.set_model()`.

            .. Note:: Experimental: Using model as path may change or be removed in a future
                        release without warning.
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
        example_no_conversion: This parameter is deprecated and will be removed in a future
                release. It's no longer used and can be safely removed. Input examples are
                not converted anymore.
        model_config: The model configuration to apply to the model if saving model from code. This
            configuration is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        streamable: A boolean value indicating if the model supports streaming prediction. If
            True, the model must implement `stream` method. If None, streamable is
            set to True if the model implements `stream` method. Default to `None`.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        import langchain
        from langchain.schema import BaseRetriever

        lc_model_or_path = _validate_and_prepare_lc_model_or_path(lc_model, loader_fn, temp_dir)

        _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)

        if isinstance(model_config, str):
            model_config = _validate_and_get_model_config_from_file(model_config)

        model_code_path = None
        if isinstance(lc_model_or_path, str):
            # The LangChain model is defined as Python code located in the file at the path
            # specified by `lc_model`. Verify that the path exists and, if so, copy it to the
            # model directory along with any other specified code modules
            model_code_path = lc_model_or_path

            lc_model = _load_model_code_path(model_code_path, model_config)
            _validate_and_copy_file_to_directory(model_code_path, path, "code")
        else:
            lc_model = lc_model_or_path

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path, example_no_conversion)

    if signature is None:
        if saved_example is not None:
            wrapped_model = _LangChainModelWrapper(lc_model)
            signature = _infer_signature_from_input_example(saved_example, wrapped_model)
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

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    with _get_dependencies_schemas() as dependencies_schemas:
        schema = dependencies_schemas.to_dict()
        if schema is not None:
            if mlflow_model.metadata is None:
                mlflow_model.metadata = {}
            mlflow_model.metadata.update(schema)

    if streamable is None:
        streamable = hasattr(lc_model, "stream")

    model_data_kwargs = {}
    flavor_conf = {}
    if not isinstance(model_code_path, str):
        model_data_kwargs = _save_model(lc_model, path, loader_fn, persist_dir)
        flavor_conf = {
            _MODEL_TYPE_KEY: lc_model.__class__.__name__,
            **model_data_kwargs,
        }

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.langchain",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        predict_stream_fn="predict_stream",
        streamable=streamable,
        model_code_path=model_code_path,
        model_config=model_config,
        **model_data_kwargs,
    )

    if Version(langchain.__version__) >= Version("0.0.311"):
        if databricks_resources := _detect_databricks_dependencies(lc_model):
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
@docstring_version_compatibility_warning(FLAVOR_NAME)
@trace_disabled  # Suppress traces for internal predict calls while logging model
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
    example_no_conversion=None,
    run_id=None,
    model_config=None,
    streamable=None,
):
    """
    Log a LangChain model as an MLflow artifact for the current run.

    Args:
        lc_model: A LangChain model, which could be a
            `Chain <https://python.langchain.com/docs/modules/chains/>`_,
            `Agent <https://python.langchain.com/docs/modules/agents/>`_, or
            `retriever <https://python.langchain.com/docs/modules/data_connection/retrievers/>`_
            or a path containing the `LangChain model code <https://github.com/mlflow/mlflow/blob/master/examples/langchain/chain_as_code_driver.py>`
            for the above types. When using model as path, make sure to set the model
            by using :func:`mlflow.models.set_model()`.

            .. Note:: Experimental: Using model as path may change or be removed in a future
                                    release without warning.
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
        example_no_conversion: This parameter is deprecated and will be removed in a future
                release. It's no longer used and can be safely removed. Input examples are
                not converted anymore.
        run_id: run_id to associate with this model version. If specified, we resume the
                run and log the model to that run. Otherwise, a new run is created.
                Default to None.
        model_config: The model configuration to apply to the model if saving model from code. This
            configuration is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        streamable: A boolean value indicating if the model supports streaming prediction. If
            True, the model must implement `stream` method. If None, If None, streamable is
            set to True if the model implements `stream` method. Default to `None`.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
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
        streamable=streamable,
    )


# patch_langchain_type_to_cls_dict here as we attempt to load model
# if it's saved by `dict` method
@patch_langchain_type_to_cls_dict
def _save_model(model, path, loader_fn, persist_dir):
    if Version(cloudpickle.__version__) < Version("2.1.0"):
        warnings.warn(
            "If you are constructing a custom LangChain model, "
            "please upgrade cloudpickle to version 2.1.0 or later "
            "using `pip install cloudpickle>=2.1.0` "
            "to ensure the model can be loaded correctly."
        )

    with register_pydantic_v1_serializer_cm():
        if isinstance(model, lc_runnables_types()):
            return _save_runnables(model, path, loader_fn=loader_fn, persist_dir=persist_dir)
        else:
            return _save_base_lcs(model, path, loader_fn, persist_dir)


@patch_langchain_type_to_cls_dict
def _load_model(local_model_path, flavor_conf):
    from mlflow.langchain._langchain_autolog import _update_langchain_model_config

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
    # To avoid double logging, we set _mlflow_model_logged to True
    # when the model is loaded
    if not autologging_is_disabled(FLAVOR_NAME):
        if _update_langchain_model_config(model):
            model._mlflow_model_logged = True
            model.run_id = get_model_info(local_model_path).run_id
    return model


class _LangChainModelWrapper:
    def __init__(self, lc_model, model_path=None):
        self.lc_model = lc_model
        self.model_path = model_path

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.lc_model

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
        # TODO: We don't automatically turn tracing on in OSS model serving, because we haven't
        # implemented storage option for traces in OSS model serving (counterpart to the
        # Inference Table in Databricks model serving).
        if (
            is_in_databricks_model_serving_environment()
            # TODO: This env var was once used for controlling whether or not to inject the
            #   tracer in Databricks model serving. However, now we have the new env var
            #   `ENABLE_MLFLOW_TRACING` to control that. We don't remove this condition
            #   right now in the interest of caution, but we should remove this condition
            #   after making sure that the functionality is stable.
            and os.environ.get("MLFLOW_ENABLE_TRACE_IN_SERVING", "false").lower() == "true"
            # if this is False, tracing is disabled and we shouldn't inject the tracer
            and is_mlflow_tracing_enabled_in_model_serving()
        ):
            from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

            callbacks = [MlflowLangchainTracer()]
        elif (context := get_prediction_context()) and context.is_evaluate:
            # NB: We enable traces automatically for the model evaluation. Note that we have to
            #   manually pass the context instance to callback, because LangChain callback may be
            #   invoked asynchronously and it doesn't correctly propagate the thread-local context.
            from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

            callbacks = [MlflowLangchainTracer(prediction_context=context)]
        else:
            callbacks = None

        return self._predict_with_callbacks(data, params, callback_handlers=callbacks)

    def _update_dependencies_schemas_in_prediction_context(self, callback_handlers):
        from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

        if (
            callback_handlers
            and (
                tracer := next(
                    (c for c in callback_handlers if isinstance(c, MlflowLangchainTracer)), None
                )
            )
            and self.model_path
        ):
            model = Model.load(self.model_path)
            context = tracer._prediction_context
            if model.metadata and context:
                dependencies_schemas = model.metadata.get("dependencies_schemas", {})
                context.update(
                    dependencies_schemas={
                        dependency: json.dumps(schema)
                        for dependency, schema in dependencies_schemas.items()
                    }
                )

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

        self._update_dependencies_schemas_in_prediction_context(callback_handlers)
        messages, return_first_element = self._prepare_predict_messages(data)
        results = process_api_requests(
            lc_model=self.lc_model,
            requests=messages,
            callback_handlers=callback_handlers,
            convert_chat_responses=convert_chat_responses,
            params=params or {},
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
            params=params or {},
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

        self._update_dependencies_schemas_in_prediction_context(callback_handlers)
        data = self._prepare_predict_stream_messages(data)
        return process_stream_request(
            lc_model=self.lc_model,
            request_json=data,
            callback_handlers=callback_handlers,
            convert_chat_responses=convert_chat_responses,
            params=params or {},
        )


def _load_pyfunc(path: str, model_config: Optional[Dict[str, Any]] = None):
    """Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``langchain`` flavor.
    """
    return _LangChainModelWrapper(_load_model_from_local_fs(path, model_config), path)


def _load_model_from_local_fs(local_model_path, model_config_overrides=None):
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    pyfunc_flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=PYFUNC_FLAVOR_NAME
    )
    # Add code from the langchain flavor to the system path
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    # The model_code_path and the model_config were previously saved langchain flavor but now we
    # also save them inside the pyfunc flavor. For backwards compatibility of previous models,
    # we need to check both places.
    if MODEL_CODE_PATH in pyfunc_flavor_conf or MODEL_CODE_PATH in flavor_conf:
        model_config = pyfunc_flavor_conf.get(MODEL_CONFIG, flavor_conf.get(MODEL_CONFIG, None))
        if isinstance(model_config, str):
            config_path = os.path.join(
                local_model_path,
                os.path.basename(model_config),
            )
            model_config = _validate_and_get_model_config_from_file(config_path)

        flavor_code_path = pyfunc_flavor_conf.get(
            MODEL_CODE_PATH, flavor_conf.get(MODEL_CODE_PATH, None)
        )
        model_code_path = os.path.join(
            local_model_path,
            os.path.basename(flavor_code_path),
        )
        try:
            model = _load_model_code_path(
                model_code_path, {**(model_config or {}), **(model_config_overrides or {})}
            )
        finally:
            # We would like to clean up the dependencies schema which is set to global
            # after loading the mode to avoid the schema being used in the next model loading
            _clear_dependencies_schemas()
        return model
    else:
        return _load_model(local_model_path, flavor_conf)


@experimental
@docstring_version_compatibility_warning(FLAVOR_NAME)
@trace_disabled  # Suppress traces while loading model
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


def _patch_runnable_cls(cls):
    """
    For classes that are subclasses of Runnable, we patch the `invoke`, `batch`, `stream` and
    `ainvoke`, `abatch`, `astream` methods for autologging.

    Args:
        cls: The class to patch.
    """
    from mlflow.langchain._langchain_autolog import patched_inference

    patch_functions = ["invoke", "batch", "stream", "ainvoke", "abatch", "astream"]
    for func_name in patch_functions:
        if hasattr(cls, func_name):
            safe_patch(
                FLAVOR_NAME,
                cls,
                func_name,
                functools.partial(patched_inference, func_name),
            )


def _inspect_module_and_patch_cls(module, inspected_modules, patched_classes):
    """
    Internal method to inspect the module and patch classes that are
    subclasses of Runnable for autologging.
    """
    from langchain.schema.runnable import Runnable

    if module.__name__ not in inspected_modules:
        inspected_modules.add(module.__name__)
        for _, obj in inspect.getmembers(module):
            if inspect.ismodule(obj) and (obj.__name__.startswith("langchain")):
                _inspect_module_and_patch_cls(obj, inspected_modules, patched_classes)
            elif (
                inspect.isclass(obj)
                and obj.__name__ not in patched_classes
                and issubclass(obj, Runnable)
            ):
                _patch_runnable_cls(obj)
                patched_classes.add(obj.__name__)


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    log_datasets=False,
    log_inputs_outputs=None,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
    extra_model_classes=None,
    log_traces=True,
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
        log_inputs_outputs: **Deprecated** The legacy parameter used for logging inference
            inputs and outputs. This argument will be removed in a future version of MLflow.
            The alternative is to use ``log_traces`` which logs traces for Langchain models,
            including inputs and outputs for each stage.
            If ``True``, inference data and results are combined into a single
            pandas DataFrame and logged to MLflow Tracking as an artifact.
            If ``False``, inference data and results are not logged.
            Default to ``False``.
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
        extra_model_classes: A list of langchain classes to log in addition to the default classes.
            We do not guarantee classes specified in this list can be logged as a model, but tracing
            will be supported. Note that all classes within the list must be subclasses of Runnable,
            and we only patch `invoke`, `batch`, and `stream` methods for tracing.
        log_traces: If ``True``, traces are logged for Langchain models by using
            MlflowLangchainTracer as a callback during inference. If ``False``, no traces are
            collected during inference. Default to ``True``.
    """
    with contextlib.suppress(ImportError):
        import langchain
        import langchain_community
        from langchain.agents.agent import AgentExecutor
        from langchain.chains.base import Chain
        from langchain.schema import BaseRetriever
        from langchain.schema.runnable import Runnable

        from mlflow.langchain._langchain_autolog import patched_inference

        # avoid duplicate patching
        patched_classes = set()
        # avoid infinite recursion
        inspected_modules = set()

        for module in [langchain, langchain_community]:
            _inspect_module_and_patch_cls(module, inspected_modules, patched_classes)

        if extra_model_classes:
            unsupported_classes = []
            for cls in extra_model_classes:
                if cls.__name__ in patched_classes:
                    continue
                elif inspect.isclass(cls) and issubclass(cls, Runnable):
                    _patch_runnable_cls(cls)
                    patched_classes.add(cls.__name__)
                else:
                    unsupported_classes.append(cls.__name__)
            if unsupported_classes:
                logger.warning(
                    f"Unsupported classes found in extra_model_classes: {unsupported_classes}. "
                    "Only subclasses of Runnable are supported."
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

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
import tempfile
import warnings
from typing import Any, Iterator, Optional, Union

import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.exceptions import MlflowException
from mlflow.langchain.constants import FLAVOR_NAME
from mlflow.langchain.databricks_dependencies import _detect_databricks_dependencies
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils.logging import (
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
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.dependencies_schemas import (
    _clear_dependencies_schemas,
    _get_dependencies_schema_from_model,
    _get_dependencies_schemas,
)
from mlflow.models.model import MLMODEL_FILE_NAME, MODEL_CODE_PATH, MODEL_CONFIG
from mlflow.models.resources import DatabricksFunction, Resource, _ResourceBuilder
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import (
    _convert_llm_input_data,
    _load_model_code_path,
    _save_example,
)
from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.fluent import (
    _get_active_model_context,
    _set_active_model,
)
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    disable_autologging_globally,
)
from mlflow.utils.databricks_utils import (
    _get_databricks_serverless_env_vars,
    is_in_databricks_model_serving_environment,
    is_in_databricks_serverless_runtime,
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
                        name="retrieval_qa",
                        loader_fn=load_retriever,
                        persist_dir=persist_dir,
                    )

            See a complete example in examples/langchain/retrieval_qa_chain.py.
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
    saved_example = _save_example(mlflow_model, input_example, path)

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

    needs_databricks_auth = False
    if Version(langchain.__version__) >= Version("0.0.311") and mlflow_model.resources is None:
        if databricks_resources := _detect_databricks_dependencies(lc_model):
            logger.info(
                "Attempting to auto-detect Databricks resource dependencies for the "
                "current langchain model. Dependency auto-detection is "
                "best-effort and may not capture all dependencies of your langchain "
                "model, resulting in authorization errors when serving or querying "
                "your model. We recommend that you explicitly pass `resources` "
                "to mlflow.langchain.log_model() to ensure authorization to "
                "dependent resources succeeds when the model is deployed."
            )
            serialized_databricks_resources = _ResourceBuilder.from_resources(databricks_resources)
            mlflow_model.resources = serialized_databricks_resources
            needs_databricks_auth = any(
                isinstance(r, DatabricksFunction) for r in databricks_resources
            )

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
            extra_env_vars = (
                _get_databricks_serverless_env_vars()
                if needs_databricks_auth and is_in_databricks_serverless_runtime()
                else None
            )
            inferred_reqs = mlflow.models.infer_pip_requirements(
                str(path), FLAVOR_NAME, fallback=default_reqs, extra_env_vars=extra_env_vars
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
@disable_autologging_globally  # Avoid side-effect of autologging while logging model
def log_model(
    lc_model,
    artifact_path: Optional[str] = None,
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
    run_id=None,
    model_config=None,
    streamable=None,
    resources: Optional[Union[list[Resource], str]] = None,
    prompts: Optional[list[Union[str, Prompt]]] = None,
    name: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, Any]] = None,
    model_type: Optional[str] = None,
    step: int = 0,
    model_id: Optional[str] = None,
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
        artifact_path: Deprecated. Use `name` instead.
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
                        name="retrieval_qa",
                        loader_fn=load_retriever,
                        persist_dir=persist_dir,
                    )

            See a complete example in examples/langchain/retrieval_qa_chain.py.
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
        resources: A list of model resources or a resources.yaml file containing a list of
            resources required to serve the model. If logging a LangChain model with dependencies
            (e.g. on LLM model serving endpoints), we encourage explicitly passing dependencies
            via this parameter. Otherwise, ``log_model`` will attempt to infer dependencies,
            but dependency auto-inference is best-effort and may miss some dependencies.
        prompts: {{ prompts }}

        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
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
        run_id=run_id,
        model_config=model_config,
        streamable=streamable,
        resources=resources,
        prompts=prompts,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
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
        data: Union[pd.DataFrame, list[Union[str, dict[str, Any]]], Any],
        params: Optional[dict[str, Any]] = None,
    ) -> list[Union[str, dict[str, Any]]]:
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
            if schema := _get_dependencies_schema_from_model(model):
                context.update(**schema)

    @experimental
    def _predict_with_callbacks(
        self,
        data: Union[pd.DataFrame, list[Union[str, dict[str, Any]]], Any],
        params: Optional[dict[str, Any]] = None,
        callback_handlers=None,
        convert_chat_responses=False,
    ) -> list[Union[str, dict[str, Any]]]:
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
        params: Optional[dict[str, Any]] = None,
    ) -> Iterator[Union[str, dict[str, Any]]]:
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
        params: Optional[dict[str, Any]] = None,
        callback_handlers=None,
        convert_chat_responses=False,
    ) -> Iterator[Union[str, dict[str, Any]]]:
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


def _load_pyfunc(path: str, model_config: Optional[dict[str, Any]] = None):  # noqa: D417
    """Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``langchain`` flavor.
    """
    return _LangChainModelWrapper(_load_model_from_local_fs(path, model_config), path)


def _load_model_from_local_fs(local_model_path, model_config_overrides=None):
    mlflow_model = Model.load(local_model_path)
    if (
        mlflow_model.model_id
        and (amc := _get_active_model_context())
        # only set the active model if the model is not set by the user
        and not amc.set_by_user
        and amc.model_id != mlflow_model.model_id
    ):
        _set_active_model(model_id=mlflow_model.model_id)
        logger.info(
            "Use `mlflow.set_active_model` to set the active model to a different one if needed."
        )
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
@disable_autologging_globally  # Avoid side-effect of autologging while loading model
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
    model_uri = str(model_uri)
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    return _load_model_from_local_fs(local_model_path)

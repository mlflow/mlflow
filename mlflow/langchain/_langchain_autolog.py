import contextlib
import inspect
import logging
import os
import random
import string
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from packaging.version import Version

import mlflow
from mlflow.entities import RunTag
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.autologging_utils import (
    AUTOLOGGING_INTEGRATIONS,
    disable_autologging,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

MIN_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["minimum"])
MAX_REQ_VERSION = Version(_ML_PACKAGE_VERSIONS["langchain"]["autologging"]["maximum"])

_logger = logging.getLogger(__name__)

UNSUPPORT_LOG_MODEL_MESSAGE = (
    "MLflow autologging does not support logging models containing BaseRetriever because "
    "logging the model requires `loader_fn` and `persist_dir`. Please log the model manually "
    "using `mlflow.langchain.log_model(model, artifact_path, loader_fn=..., persist_dir=...)`"
)
INFERENCE_FILE_NAME = "inference_inputs_outputs.json"


@dataclass
class AutoLoggingConfig:
    log_models: bool
    log_input_examples: bool
    log_model_signatures: bool
    log_inputs_outputs: bool
    extra_tags: Optional[dict] = None

    def should_log_optional_artifacts(self):
        """
        Check if any optional artifacts should be logged to MLflow.
        """
        return (
            self.log_models
            or self.log_input_examples
            or self.log_model_signatures
            or self.log_inputs_outputs
        )

    @staticmethod
    def init():
        config_dict = AUTOLOGGING_INTEGRATIONS.get(mlflow.langchain.FLAVOR_NAME, {})
        return AutoLoggingConfig(
            log_models=config_dict.get("log_models", False),
            log_input_examples=config_dict.get("log_input_examples", False),
            log_model_signatures=config_dict.get("log_model_signatures", False),
            log_inputs_outputs=config_dict.get("log_inputs_outputs", False),
            extra_tags=config_dict.get("extra_tags", None),
        )


def patched_inference(func_name, original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables
    logging the traces, and other optional artifacts like model, input examples, etc.

    We patch either `invoke` or `__call__` function for different models
    based on their usage.
    """
    # Inject MLflow tracer into the model
    # TODO: the legacy LangChain callback is removed as its functionality largely
    #  overlaps with the tracer while increasing the latency due to synchronous
    #  artifact logging. However, complex text metrics are not logged by the tracer.
    #  We should recover this functionality either in the tracer or a separate
    #  callback before merging the LangChain autologging change into the main branch.
    #  https://github.com/langchain-ai/langchain/blob/38fb1429fe5a955a863fb91b626c2c9f85efb703/libs/community/langchain_community/callbacks/mlflow_callback.py#L62-L81
    mlflow_tracer = _set_up_langchain_tracer()
    args, kwargs = _inject_mlflow_callbacks(func_name, [mlflow_tracer], args, kwargs)

    autolog_config = AutoLoggingConfig.init()

    if autolog_config.should_log_optional_artifacts():
        # This warning is only valid for the legacy LangChain autologging, and the new tracer
        # should work outside the version range.
        import langchain

        if not MIN_REQ_VERSION <= Version(langchain.__version__) <= MAX_REQ_VERSION:
            warnings.warn(
                "Autologging is known to be compatible with langchain versions between "
                f"{MIN_REQ_VERSION} and {MAX_REQ_VERSION} and may not succeed with packages "
                "outside this range."
            )

        # Logging Traces does not require an active MLflow run, while the other optional artifacts
        # needs to be logged into an active run, so we need to start a run if it's not already set.
        run_id = getattr(self, "run_id", _setup_autolog_run(autolog_config))

    with disable_autologging():
        result = original(self, *args, **kwargs)

    if autolog_config.should_log_optional_artifacts():
        _log_optional_artifacts(autolog_config, run_id, result, self, func_name, *args, **kwargs)

        # Terminate the run if it is not managed by the user
        if mlflow.active_run() is None or mlflow.active_run().info.run_id != run_id:
            mlflow.MlflowClient().set_terminated(run_id)

    return result


def _set_up_langchain_tracer():
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    mlflow_tracer = MlflowLangchainTracer()

    try:
        mlflow_tracer.flush_tracker()
    except Exception as e:
        if _MLFLOW_TESTING.get():
            raise
        _logger.warning(f"Failed to flush mlflow tracer due to error {e}.")

    return mlflow_tracer


def _setup_autolog_run(config):
    active_run = mlflow.active_run()

    # include run context tags
    resolved_tags = context_registry.resolve_tags(config.extra_tags)
    tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, resolved_tags)

    if active_run:
        # If there is an active run, log the autologging-related tags to the run
        run_id = active_run.info.run_id
        mlflow.MlflowClient().log_batch(
            run_id=run_id,
            tags=[RunTag(key, str(value)) for key, value in tags.items()],
        )
        return run_id
    else:
        experiment_id = _get_experiment_id()
        run_name = "langchain-" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=7)
        )
        run = mlflow.MlflowClient().create_run(experiment_id, run_name=run_name, tags=tags)
        run_id = run.info.run_id

    return run_id


# TODO: Experiment ID should be set for any Trace object not only for the LangChain autologging.
#  We should move this to a shared utility and call it within the trace exporter.
def _get_experiment_id():
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        mlflow.set_tracking_uri("databricks")
        experiment_id = mlflow.tracking.fluent._get_experiment_id()
    else:
        tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        experiment_id = mlflow.tracking.fluent._get_experiment_id_from_env()

    return experiment_id


def _get_input_data_from_function(func_name, model, args, kwargs):
    func_param_name_mapping = {
        "__call__": "inputs",
        "invoke": "input",
        "get_relevant_documents": "query",
    }
    input_example_exc = None
    if param_name := func_param_name_mapping.get(func_name):
        inference_func = getattr(model, func_name)
        # A guard to make sure `param_name` is the first argument of inference function
        if next(iter(inspect.signature(inference_func).parameters.keys())) != param_name:
            input_example_exc = MlflowException(
                "Inference function signature changes, please contact MLflow team to "
                "fix langchain autologging.",
            )
        else:
            return args[0] if len(args) > 0 else kwargs.get(param_name)
    else:
        input_example_exc = MlflowException(
            f"Unsupported inference function. Only support {list(func_param_name_mapping.keys())}."
        )
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__} "
        f"due to {input_example_exc}."
    )


def _convert_data_to_dict(data, key):
    if isinstance(data, dict):
        return {f"{key}-{k}": v for k, v in data.items()}
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, str):
        return {key: [data]}
    raise MlflowException("Unsupported data type.")


def _combine_input_and_output(input, output, session_id, func_name):
    """
    Combine input and output into a single dictionary
    """
    if func_name == "get_relevant_documents" and output is not None:
        output = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in output]
        # to make sure output is inside a single row when converted into pandas DataFrame
        output = [output]
    result = {"session_id": [session_id]}
    if input:
        result.update(_convert_data_to_dict(input, "input"))
    if output:
        result.update(_convert_data_to_dict(output, "output"))
    return result


def _update_langchain_model_config(model):
    try:
        from langchain_core.pydantic_v1 import Extra
    except ImportError as e:
        warnings.warn(
            "MLflow langchain autologging might log model several "
            "times due to the pydantic.config.Extra import error. "
            f"Error: {e}"
        )
        return False
    else:
        # Langchain models are Pydantic models, and the value for extra is
        # ignored, we need to set it to allow so as to set attributes on
        # the model to keep track of logging status
        if hasattr(model, "__config__"):
            model.__config__.extra = Extra.allow
        return True


def _inject_mlflow_callbacks(func_name, mlflow_callbacks, args, kwargs):
    if func_name == "invoke":
        from langchain.schema.runnable.config import RunnableConfig

        in_args = False
        # `config` is the second positional argument of runnable.invoke function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/runnables/base.py#L468
        if len(args) >= 2:
            config = args[1]
            in_args = True
        else:
            config = kwargs.get("config", None)
        if config is None:
            callbacks = mlflow_callbacks
            config = RunnableConfig(callbacks=callbacks)
        else:
            callbacks = config.get("callbacks") or []
            callbacks.extend(mlflow_callbacks)
            config["callbacks"] = callbacks
        if in_args:
            args = (args[0], config) + args[2:]
        else:
            kwargs["config"] = config
        return args, kwargs

    if func_name == "__call__":
        # `callbacks` is the third positional argument of chain.__call__ function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/langchain/langchain/chains/base.py#L320
        if len(args) >= 3:
            callbacks = args[2] or []
            callbacks.extend(mlflow_callbacks)
            args = args[:2] + (callbacks,) + args[3:]
        else:
            callbacks = kwargs.get("callbacks") or []
            callbacks.extend(mlflow_callbacks)
            kwargs["callbacks"] = callbacks
        return args, kwargs

    # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/retrievers.py#L173
    if func_name == "get_relevant_documents":
        callbacks = kwargs.get("callbacks") or []
        callbacks.extend(mlflow_callbacks)
        kwargs["callbacks"] = callbacks
        return args, kwargs


def _runnable_with_retriever(model):
    from langchain.schema import BaseRetriever

    with contextlib.suppress(ImportError):
        from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
        from langchain.schema.runnable.passthrough import RunnableAssign

        if isinstance(model, RunnableBranch):
            return any(_runnable_with_retriever(runnable) for _, runnable in model.branches)

        if isinstance(model, RunnableParallel):
            return any(_runnable_with_retriever(runnable) for runnable in model.steps.values())

        if isinstance(model, RunnableSequence):
            return any(_runnable_with_retriever(runnable) for runnable in model.steps)

        if isinstance(model, RunnableAssign):
            return _runnable_with_retriever(model.mapper)

    return isinstance(model, BaseRetriever)


def _chain_with_retriever(model):
    with contextlib.suppress(ImportError):
        from langchain.chains import RetrievalQA

        return isinstance(model, RetrievalQA)
    return False


def _log_optional_artifacts(autolog_config, run_id, result, self, func_name, *args, **kwargs):
    input_example = None
    if autolog_config.log_models and not hasattr(self, "model_logged"):
        if (
            (func_name == "get_relevant_documents")
            or _runnable_with_retriever(self)
            or _chain_with_retriever(self)
        ):
            _logger.info(UNSUPPORT_LOG_MODEL_MESSAGE)
        else:
            # warn user in case we did't capture some cases where retriever is used
            warnings.warn(UNSUPPORT_LOG_MODEL_MESSAGE)
            if autolog_config.log_input_examples:
                input_example = deepcopy(
                    _get_input_data_from_function(func_name, self, args, kwargs)
                )
                if not autolog_config.log_model_signatures:
                    _logger.info(
                        "Signature is automatically generated for logged model if "
                        "input_example is provided. To disable log_model_signatures, "
                        "please also disable log_input_examples."
                    )

            registered_model_name = get_autologging_config(
                mlflow.langchain.FLAVOR_NAME, "registered_model_name", None
            )
            try:
                with disable_autologging():
                    mlflow.langchain.log_model(
                        self,
                        "model",
                        input_example=input_example,
                        registered_model_name=registered_model_name,
                        run_id=run_id,
                    )
            except Exception as e:
                _logger.warning(f"Failed to log model due to error {e}.")
            if _update_langchain_model_config(self):
                self.model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if _update_langchain_model_config(self):
        # NB: We have to set these attributes AFTER the model is logged, otherwise those extra
        # attributes will be logged as a part of the pickled model and pollute the loaded model.
        if not hasattr(self, "run_id"):
            self.run_id = run_id
        if not hasattr(self, "session_id"):
            self.session_id = uuid.uuid4().hex
        self.inference_id = getattr(self, "inference_id", 0) + 1

    if autolog_config.log_inputs_outputs:
        if input_example is None:
            input_data = deepcopy(_get_input_data_from_function(func_name, self, args, kwargs))
            if input_data is None:
                _logger.info("Input data gathering failed, only log inference results.")
        else:
            input_data = input_example
        try:
            data_dict = _combine_input_and_output(input_data, result, self.session_id, func_name)
        except Exception as e:
            _logger.warning(
                f"Failed to log inputs and outputs into `{INFERENCE_FILE_NAME}` "
                f"file due to error {e}."
            )
        mlflow.log_table(data_dict, INFERENCE_FILE_NAME, run_id=self.run_id)

    return result

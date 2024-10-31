import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from typing import Dict, List, Union

from langchain_core.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from packaging.version import Version

import mlflow
from mlflow.entities import RunTag
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from mlflow.langchain.runnables import get_runnable_steps
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import name_utils
from mlflow.utils.autologging_utils import disable_autologging, get_autologging_config
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

_logger = logging.getLogger(__name__)

UNSUPPORT_LOG_MODEL_MESSAGE = (
    "MLflow autologging does not support logging models containing BaseRetriever because "
    "logging the model requires `loader_fn` and `persist_dir`. Please log the model manually "
    "using `mlflow.langchain.log_model(model, artifact_path, loader_fn=..., persist_dir=...)`"
)
INFERENCE_FILE_NAME = "inference_inputs_outputs.json"


def patched_inference(func_name, original, self, *args, **kwargs):
    """
    A patched implementation of langchain models inference process which enables
    logging the traces, and other optional artifacts like model, input examples, etc.

    We patch inference functions for different models based on their usage.
    """

    # NB: Running the original inference with disabling autologging, so we only patch the top-level
    # component and avoid duplicate logging for child components.
    def _invoke(self, *args, **kwargs):
        with disable_autologging():
            return original(self, *args, **kwargs)

    config = AutoLoggingConfig.init(mlflow.langchain.FLAVOR_NAME)

    if config.log_traces:
        args, kwargs = _get_args_with_mlflow_tracer(func_name, args, kwargs)
        _logger.debug("Injected MLflow callbacks into the model.")

    # Traces does not require an MLflow run, only the other optional artifacts require it.
    if config.should_log_optional_artifacts():
        with _setup_autolog_run(config, self) as run_id:
            result = _invoke(self, *args, **kwargs)
            _log_optional_artifacts(config, run_id, result, self, func_name, *args, **kwargs)
    else:
        result = _invoke(self, *args, **kwargs)
    return result


def _get_args_with_mlflow_tracer(func_name, args, kwargs):
    """
    Get the patched arguments with MLflow tracer injected.
    """
    mlflow_tracer = MlflowLangchainTracer()

    if func_name in ["invoke", "batch", "stream", "ainvoke", "abatch", "astream"]:
        # `config` is the second positional argument of runnable APIs such as
        # invoke, batch, stream, ainvoke, abatch, and astream
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/runnables/base.py
        if len(args) >= 2:
            config = args[1]
            config = _get_runnable_config_with_callback(config, mlflow_tracer)
            return (args[0], config, *args[2:]), kwargs
        else:
            config = kwargs.get("config")
            kwargs["config"] = _get_runnable_config_with_callback(config, mlflow_tracer)
        return args, kwargs

    elif func_name == "__call__":
        # `callbacks` is the third positional argument of chain.__call__ function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/langchain/langchain/chains/base.py#L320
        if len(args) >= 3:
            callbacks = args[2] or []
            callbacks = _inject_callback(callbacks, mlflow_tracer)
            return (*args[:2], callbacks, *args[3:]), kwargs
        else:
            callbacks = kwargs.get("callbacks") or []
            kwargs["callbacks"] = _inject_callback(callbacks, mlflow_tracer)
            return args, kwargs

    elif func_name == "get_relevant_documents":
        # callbacks is only available as kwargs in get_relevant_documents function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/retrievers.py#L173
        callbacks = kwargs.get("callbacks") or []
        kwargs["callbacks"] = _inject_callback(callbacks, mlflow_tracer)
        return args, kwargs

    else:
        _logger.warning(f"Unsupported function `{func_name}`. Skipping injecting MLflow callbacks.")
        return args, kwargs


def _get_runnable_config_with_callback(
    original_config: Union[None, Dict, List[Dict]],
    new_callback: BaseCallbackHandler,
) -> Union[Dict, List[Dict]]:
    """
    Create a new RunnableConfig (or a list of them) with the new callback injected.

    This function MUST return a new RunnableConfig instance, instead of mutating the original
    config. This is because the original config may be shared across different calls and in-place
    modification may cause unexpected behaviors, e.g. double injection.

    Args:
        original_config: the original RunnableConfig passed by the user
        new_callback: a new callback to be injected
    """
    from langchain.schema.runnable.config import RunnableConfig

    if original_config is None:
        return RunnableConfig(callbacks=[new_callback])
    elif isinstance(original_config, list):
        return [_get_runnable_config_with_callback(c, new_callback) for c in original_config]
    # Here we expect RunnableConfig, but it is a TypedDict so cannot be used for `isinstance`
    # check. At runtime, it will merely be a dict.
    elif isinstance(original_config, dict):
        config_copy = original_config.copy()
        callbacks = config_copy.pop("callbacks", None) or []
        callbacks = _inject_callback(callbacks, new_callback)
        return RunnableConfig(callbacks=callbacks, **config_copy)
    else:
        _logger.warning(
            f"Unsupported config type `{original_config}` for autologging with tracing."
            "Skipping injecting MLflow callbacks."
        )
        return original_config


def _inject_callback(
    original_callbacks: Union[List[BaseCallbackHandler], BaseCallbackManager],
    new_callback: MlflowLangchainTracer,
) -> Union[List, BaseCallbackManager]:
    """
    Inject a callback into the original callbacks.

    This function MUST return a new list or new callback manager instance, instead of
    mutating the original callbacks. This is because the original callbacks may be shared across
    different calls and in-place modification may cause unexpected behaviors, e.g. double injection.

    Args:
        original_callbacks: the original callbacks passed by the user
        new_callback: a new callback to be injected
    """
    if isinstance(original_callbacks, BaseCallbackManager):
        callback_manager_copy = original_callbacks.copy()
        if not any(isinstance(cb, type(new_callback)) for cb in callback_manager_copy.handlers):
            # Create a copy of the handlers list to avoid modifying the original handlers list,
            # while the callback instance itself is shallow copied.
            handlers = [*callback_manager_copy.handlers, new_callback]
            callback_manager_copy.handlers = handlers
        return callback_manager_copy

    elif _is_list_of_base_callback_handlers(original_callbacks):
        callback_list_copy = list(original_callbacks)
        if not any(isinstance(cb, type(new_callback)) for cb in callback_list_copy):
            callback_list_copy.append(new_callback)
        return callback_list_copy

    else:
        _logger.warning(
            f"Unsupported callbacks type `{original_callbacks}` for autologging with tracing."
            "Skipping injecting MLflow callbacks."
        )
        return original_callbacks


def _is_list_of_base_callback_handlers(callbacks) -> bool:
    return isinstance(callbacks, list) and all(
        isinstance(cb, BaseCallbackHandler) for cb in callbacks
    )


@contextlib.contextmanager
def _setup_autolog_run(config, model):
    """Set up autologging run and return the run ID.

    This function only creates a run when there is no active run and the model does not have
    a run ID attribute propagated from the previous call. Iff it creates a new run, MLflow should
    terminate the run at the end of the inference.

    Args:
        config: AutoLoggingConfig: The autologging configuration.
        model: Any: The LangChain model instance that runs the inference.

    Returns: yields the run IDs
    """
    if propagated_run_id := getattr(model, "run_id", None):
        # When model has "run_id" attribute, it means the model is already invoked once with autolog
        # enabled and the run_id is propagated from the previous call, so we don't create a new run.
        run_id = propagated_run_id
        # The run should be already terminated at the end of the previous call.
        should_terminate_run = False

    elif active_run := mlflow.active_run():
        run_id = active_run.info.run_id
        tags = _resolve_tags(config.extra_tags, active_run)
        mlflow.MlflowClient().log_batch(run_id, tags=[RunTag(k, str(v)) for k, v in tags.items()])
        should_terminate_run = False
    else:
        from mlflow.tracking.fluent import _get_experiment_id

        run = mlflow.MlflowClient().create_run(
            experiment_id=_get_experiment_id(),
            run_name="langchain-" + name_utils._generate_random_name(),
            tags=_resolve_tags(config.extra_tags),
        )
        run_id = run.info.run_id
        should_terminate_run = True

    run_status = None
    try:
        yield run_id
    except Exception:
        run_status = RunStatus.to_string(RunStatus.FAILED)
        raise
    finally:
        if should_terminate_run:
            mlflow.MlflowClient().set_terminated(run_id, status=run_status)


def _resolve_tags(extra_tags, active_run=None):
    resolved_tags = context_registry.resolve_tags(extra_tags)
    tags = _resolve_extra_tags(mlflow.langchain.FLAVOR_NAME, resolved_tags)
    if active_run:
        # Some context tags like mlflow.runName are immutable once logged, but they might be already
        # set when the run is created, then we should avoid updating them.
        excluded_tags = {tag for tag in active_run.data.tags.keys() if tag.startswith("mlflow.")}
        tags = {k: v for k, v in tags.items() if k not in excluded_tags}
    return tags


def _get_input_data_from_function(func_name, model, args, kwargs):
    func_param_name_mapping = {
        "__call__": "inputs",
        "invoke": "input",
        "batch": "inputs",
        "stream": "input",
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
    # Langchain models are Pydantic models, and the value for extra is
    # ignored, we need to set it to allow so as to set attributes on
    # the model to keep track of logging status
    import langchain

    try:
        # LangChain 0.3.0 and above is fully migrated to Pydantic v2
        if Version(langchain.__version__) >= Version("0.3.0"):
            if hasattr(model, "model_config") and model.model_config is not None:
                model.model_config["extra"] = "allow"
                model.__pydantic_extra__ = {}
                return True
        else:
            from langchain_core.pydantic_v1 import Extra

            if hasattr(model, "__config__"):
                model.__config__.extra = Extra.allow
            return True
    except Exception as e:
        warnings.warn(
            "Failed to set extra attribute on the model for keeping track of logging status. "
            f"MLflow langchain autologging might log model several times. Error: {e}"
        )
        return False


def _runnable_with_retriever(model):
    from langchain.schema import BaseRetriever

    with contextlib.suppress(ImportError):
        from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableSequence
        from langchain.schema.runnable.passthrough import RunnableAssign

        if isinstance(model, RunnableBranch):
            return any(_runnable_with_retriever(runnable) for _, runnable in model.branches)

        if isinstance(model, RunnableParallel):
            return any(
                _runnable_with_retriever(runnable)
                for runnable in get_runnable_steps(model).values()
            )

        if isinstance(model, RunnableSequence):
            return any(_runnable_with_retriever(runnable) for runnable in get_runnable_steps(model))

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
    if autolog_config.log_models and not hasattr(self, "_mlflow_model_logged"):
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
        # only try logging model once, even if it can't be logged
        # we don't want to spam the user with warnings/infos
        if _update_langchain_model_config(self):
            self._mlflow_model_logged = True

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
        # we do not convert stream output iterator for logging as tracing
        # will provide much more information than this function, we might drop
        # this in the future
        if func_name == "stream":
            _logger.warning(f"Unsupported function {func_name} for logging inputs and outputs.")
        else:
            try:
                data_dict = _combine_input_and_output(
                    input_data, result, self.session_id, func_name
                )
                mlflow.log_table(data_dict, INFERENCE_FILE_NAME, run_id=self.run_id)
            except Exception as e:
                _logger.warning(
                    f"Failed to log inputs and outputs into `{INFERENCE_FILE_NAME}` "
                    f"file due to error {e}."
                )

    return result

import contextlib
import inspect
import logging
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.entities import RunTag
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.langchain.runnables import get_runnable_steps
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import name_utils
from mlflow.utils.autologging_utils import (
    AUTOLOGGING_INTEGRATIONS,
    disable_autologging,
    get_autologging_config,
)
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

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
    log_traces: bool
    log_inputs_outputs: Optional[bool] = None
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

    @classmethod
    def init(cls):
        config_dict = AUTOLOGGING_INTEGRATIONS.get(mlflow.langchain.FLAVOR_NAME, {})
        if config_dict.get("log_inputs_outputs"):
            _logger.warning(
                "The log_inputs_outputs option is deprecated and will be removed in a future "
                "release. Please use the log_traces option in `mlflow.langchain.autolog` "
                "to log traces (including inputs and outputs) of the model."
            )
        return cls(
            log_models=config_dict.get("log_models", False),
            log_input_examples=config_dict.get("log_input_examples", False),
            log_model_signatures=config_dict.get("log_model_signatures", False),
            log_traces=config_dict.get("log_traces", True),
            log_inputs_outputs=config_dict.get("log_inputs_outputs", False),
            extra_tags=config_dict.get("extra_tags", None),
        )


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

    config = AutoLoggingConfig.init()
    with patch_args_and_kwargs(func_name, config, args, kwargs) as (args, kwargs):
        # Traces does not require an MLflow run, only the other optional artifacts require it.
        if config.should_log_optional_artifacts():
            with _setup_autolog_run(config, self) as run_id:
                result = _invoke(self, *args, **kwargs)
            _log_optional_artifacts(config, run_id, result, self, func_name, *args, **kwargs)
        else:
            result = _invoke(self, *args, **kwargs)
    return result


@contextlib.contextmanager
def patch_args_and_kwargs(func_name, autologging_config, args, kwargs):
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    if autologging_config.log_traces:
        # Inject MLflow tracer into the callbacks argument
        mlflow_tracer = MlflowLangchainTracer()
        try:
            with _inject_mlflow_callbacks(func_name, [mlflow_tracer], args, kwargs) as (
                args,
                kwargs,
            ):
                yield args, kwargs
            _logger.debug("Injected MLflow callbacks into the model.")
        finally:
            mlflow_tracer.flush_tracker()
    else:
        yield args, kwargs


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


def _is_new_callback_already_in_original_callbacks(new_callback, original_callbacks):
    return any(
        isinstance(new_callback, type(original_callback))
        for original_callback in original_callbacks
    )


def _is_base_callback_manager(callback):
    from langchain_core.callbacks.base import BaseCallbackManager

    return isinstance(callback, BaseCallbackManager)


def _inject_callbacks_into_base_callback_manager(original_callbacks, new_callbacks):
    added_callbacks = []
    for callback in new_callbacks:
        if not _is_new_callback_already_in_original_callbacks(
            callback, original_callbacks.handlers
        ):
            added_callbacks.append(callback)
            original_callbacks.add_handler(callback)
    return original_callbacks, added_callbacks


def _is_list_of_base_callback_handlers(callback):
    from langchain_core.callbacks.base import BaseCallbackHandler

    return isinstance(callback, list) and all(
        isinstance(callback_handler, BaseCallbackHandler) for callback_handler in callback
    )


def _inject_callbacks_into_list_of_base_callback_handlers(original_callbacks, new_callbacks):
    # Make a copy of the original callbacks to avoid modifying the original list
    updated_callbacks = list(original_callbacks)
    for new_callback in new_callbacks:
        if not _is_new_callback_already_in_original_callbacks(new_callback, original_callbacks):
            updated_callbacks.append(new_callback)
    return updated_callbacks


@contextlib.contextmanager
def _inject_callbacks(original_callbacks, new_callbacks):
    """
    Inject list of callbacks into the original callbacks.
    For RunnableConfig, the callbacks is defined as List[BaseCallbackHandler] or BaseCallbackManager
    https://github.com/langchain-ai/langchain/blob/ed980601e1c630f996aabf85df5cb26178e53099/libs/core/langchain_core/callbacks/base.py#L636
    We shouldn't modify user's original callbacks, so we need to revert the changes after the
    inference is done.

    Args:
        original_callbacks: List[BaseCallbackHandler] or BaseCallbackManager
        new_callbacks: List[BaseCallbackHandler]
    """
    if _is_base_callback_manager(original_callbacks):
        added_callbacks = []
        try:
            original_callbacks, added_callbacks = _inject_callbacks_into_base_callback_manager(
                original_callbacks, new_callbacks
            )
            yield original_callbacks
        finally:
            # avoid modifying original callbackmanager
            for callback in added_callbacks:
                original_callbacks.remove_handler(callback)

    elif _is_list_of_base_callback_handlers(original_callbacks):
        yield _inject_callbacks_into_list_of_base_callback_handlers(
            original_callbacks, new_callbacks
        )

    else:
        _logger.warning(
            f"Unsupported callbacks type `{original_callbacks}` for autologging with tracing."
        )
        yield original_callbacks


@contextlib.contextmanager
def _inject_callbacks_to_config_list(config_list, new_callbacks):
    """
    Inject new_callbacks into each config in the config_list.
    """
    added_callbacks_list = {}
    original_callbacks_list = {}
    for i, config in enumerate(config_list):
        original_callbacks = config.get("callbacks") or []
        if _is_base_callback_manager(original_callbacks):
            original_callbacks, added_callbacks = _inject_callbacks_into_base_callback_manager(
                original_callbacks, new_callbacks
            )
            config["callbacks"] = original_callbacks
            added_callbacks_list[i] = added_callbacks
        elif _is_list_of_base_callback_handlers(original_callbacks):
            config["callbacks"] = _inject_callbacks_into_list_of_base_callback_handlers(
                original_callbacks, new_callbacks
            )
            original_callbacks_list[i] = original_callbacks
    try:
        yield config_list
    finally:
        for i, config in enumerate(config_list):
            # original callback must be baseCallbackManager
            if i in added_callbacks_list:
                for callback in added_callbacks_list[i]:
                    config["callbacks"].remove_handler(callback)
            # original callback must be list of baseCallbackHandlers
            else:
                config["callbacks"] = original_callbacks_list[i] or None


@contextlib.contextmanager
def _inject_callbacks_for_runnable(mlflow_callbacks, args, kwargs):
    """
    `config` is the second positional argument of runnable invoke, batch, stream,
    ainvoke, abatch, astream functions
    https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/runnables/base.py#L468
    https://github.com/langchain-ai/langchain/blob/ed980601e1c630f996aabf85df5cb26178e53099/libs/core/langchain_core/runnables/base.py#L600-L607
    https://github.com/langchain-ai/langchain/blob/ed980601e1c630f996aabf85df5cb26178e53099/libs/core/langchain_core/runnables/base.py#L801
    https://github.com/langchain-ai/langchain/blob/25ba7332185e0c6624a2b02b72030f073755d716/libs/core/langchain_core/runnables/base.py#L588
    https://github.com/langchain-ai/langchain/blob/25ba7332185e0c6624a2b02b72030f073755d716/libs/core/langchain_core/runnables/base.py#L706
    https://github.com/langchain-ai/langchain/blob/25ba7332185e0c6624a2b02b72030f073755d716/libs/core/langchain_core/runnables/base.py#L810

    """
    from langchain.schema.runnable.config import RunnableConfig

    def _update_args_or_kwargs(in_args, config, args, kwargs):
        if in_args:
            yield (args[0], config) + args[2:], kwargs
        else:
            kwargs["config"] = config
            yield args, kwargs

    in_args = False
    if len(args) >= 2:
        config = args[1]
        in_args = True
    else:
        config = kwargs.get("config")
    if config is None:
        config = RunnableConfig(callbacks=mlflow_callbacks)
        yield from _update_args_or_kwargs(in_args, config, args, kwargs)
    else:
        # for `invoke`, `ainvoke`, `stream` and `astream`, config type is RunnableConfig
        # for `batch` and `abatch`, config type is Union[RunnableConfig, List[RunnableConfig]]
        if isinstance(config, list):
            with _inject_callbacks_to_config_list(config, mlflow_callbacks) as updated_config:
                yield from _update_args_or_kwargs(in_args, updated_config, args, kwargs)
        else:
            original_callbacks = config.get("callbacks") or []
            with _inject_callbacks(original_callbacks, mlflow_callbacks) as updated_callbacks:
                try:
                    config["callbacks"] = updated_callbacks
                    yield from _update_args_or_kwargs(in_args, config, args, kwargs)
                finally:
                    config["callbacks"] = original_callbacks or None


@contextlib.contextmanager
def _inject_mlflow_callbacks(func_name, mlflow_callbacks, args, kwargs):
    """
    Inject list of callbacks into the function named `func_name` of the model.
    """
    if func_name in ["invoke", "batch", "stream", "ainvoke", "abatch", "astream"]:
        with _inject_callbacks_for_runnable(mlflow_callbacks, args, kwargs) as (
            args,
            kwargs,
        ):
            yield args, kwargs

    elif func_name == "__call__":
        # `callbacks` is the third positional argument of chain.__call__ function
        # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/langchain/langchain/chains/base.py#L320
        if len(args) >= 3:
            with _inject_callbacks(args[2] or [], mlflow_callbacks) as callbacks_arg:
                yield args[:2] + (callbacks_arg,) + args[3:], kwargs
        else:
            with _inject_callbacks(
                kwargs.get("callbacks") or [], mlflow_callbacks
            ) as callbacks_arg:
                kwargs["callbacks"] = callbacks_arg
                yield args, kwargs

    # callbacks is only available as kwargs in get_relevant_documents function
    # https://github.com/langchain-ai/langchain/blob/7d444724d7582386de347fb928619c2243bd0e55/libs/core/langchain_core/retrievers.py#L173
    elif func_name == "get_relevant_documents":
        with _inject_callbacks(kwargs.get("callbacks") or [], mlflow_callbacks) as callbacks_arg:
            kwargs["callbacks"] = callbacks_arg
            yield args, kwargs

    else:
        _logger.warning(f"Unsupported function `{func_name}` for injecting MLflow callbacks.")
        yield args, kwargs


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
